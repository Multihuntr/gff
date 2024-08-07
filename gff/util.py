import datetime
import functools
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence, Union

import affine
import geopandas
import pyproj
import rasterio
import shapely
import numpy as np
import pandas as pd
import torch
import xarray


# Generic geometry utilities


def rounded_bounds(arg, outer=True):
    if isinstance(arg, tuple) or (isinstance(arg, np.ndarray) and arg.shape == (4,)):
        xlo, ylo, xhi, yhi = arg
        if outer:
            return math.floor(xlo), math.floor(ylo), math.ceil(xhi), math.ceil(yhi)
        else:
            return math.ceil(xlo), math.ceil(ylo), math.floor(xhi), math.floor(yhi)
    elif isinstance(arg, shapely.Geometry):
        return rounded_bounds(shapely.bounds(arg))
    else:
        raise ValueError("Bounds must be 4-tuple, 1D np.array or a shapely.Geometry")


def shapely_bounds_to_rasterio_window(bounds, transform=None, align=True):
    """
    Converts a shapely bounds (xlo, ylo, xhi, yhi)
    into a rasterio window tuple ((ylo, yhi), (xlo, xhi)).

    If transform is provided, the bounds are assumed to be in a CRS, and it
    will convert them to pixel-space before returning.

    If aligned, will enforce that it's rounded to the nearest pixel.
    Generally you want to align for writing, but not align for reading.
    This is because rasterio's write always rounds down, but it's read doesn't.
    So, if you try to write to ((0, 9.9999999), (0, 10)),
    it will only write to ((0, 9), (0, 10)).
    """
    xlo, ylo, xhi, yhi = bounds

    # If transform is provided, assume bounds are in CRS and convert to pixel-space
    if transform is not None:
        coords = np.array(((xlo, ylo), (xhi, yhi)))
        (xlo, ylo), (xhi, yhi) = np.array((~transform) * coords.T).T

    # Sometimes bounds are flipped after transform; ensure lo < hi
    if xhi < xlo:
        xlo, xhi = xhi, xlo
    if yhi < ylo:
        ylo, yhi = yhi, ylo

    # Rasterio write doesn't use resampling nearest, so floating-point errors will cause off-by-one
    if align:
        xlo, ylo, xhi, yhi = round(xlo), round(ylo), round(xhi), round(yhi)

    return ((ylo, yhi), (xlo, xhi))


def image_footprint(tif_path: Union[Path, rasterio.DatasetReader]) -> shapely.Geometry:
    """Create shape of image bounds in image CRS"""
    if isinstance(tif_path, Path):
        with rasterio.open(tif_path) as tif:
            coords = np.array([[0, 0], [tif.width, tif.height]])
            (xlo, ylo), (xhi, yhi) = np.array(tif.transform * coords.T).T
    else:
        tif = tif_path
        coords = np.array([[0, 0], [tif.width, tif.height]])
        (xlo, ylo), (xhi, yhi) = np.array(tif.transform * coords.T).T
    return shapely.box(xlo, ylo, xhi, yhi)


def mk_box_grid(width, height, x_offset=0, y_offset=0, box_width=1, box_height=1):
    """
    Create a grid of box geometries, stored in a vectorised Shapely array.
    """
    xs = np.arange(width // box_width) * box_width
    ys = np.arange(height // box_height) * box_height
    yss, xss = np.meshgrid(ys, xs)
    # fmt: off
    coords = np.array([ # Clockwise squares
        [xss+x_offset,           yss+y_offset],
        [xss+x_offset+box_width, yss+y_offset],
        [xss+x_offset+box_width, yss+y_offset+box_height],
        [xss+x_offset,           yss+y_offset+box_height],
    ]).transpose((2,3,0,1)) # shapes [4, 2, W, H] -> [W, H, 4, 2]
    # fmt: on
    return shapely.polygons(coords)


def mk_pixel_overlap_mask(geom, bounds):
    """
    Create a mask of shape specified by bounds, where each grid cell / pixel is
    the overlap proportion between `geom` and that grid cell / pixel.

    Note that the returned array starts at the (xlo, ylo) specified in `bounds`.

    Args:
        geom (shapely.Geometry): Geometry in pixel-space to find overlap
        bounds (tuple(int)): Standard bounds

    Returns:
        np.ndarray: Shaped [W, H], proportion of overlap as an array.
    """
    xlo, ylo, xhi, yhi = bounds
    pixel_geoms = mk_box_grid(xhi - xlo, yhi - ylo, x_offset=xlo, y_offset=ylo)
    intersections = geom.intersection(pixel_geoms)

    # note: pixels are defined as being unit sized, so area(intersection) == proportional overlap
    return shapely.area(intersections)


def convert_crs(shp: Union[shapely.Geometry, np.ndarray], _from: str, _to: str):
    if _from == _to:
        return shp
    project = pyproj.Transformer.from_crs(_from, _to, always_xy=True).transform
    if isinstance(shp, shapely.Geometry):
        return shapely.ops.transform(project, shp)
    else:
        op = lambda xy: np.stack(project(xy[..., 0], xy[..., 1]), axis=-1)
        return convert_shp(shp, op)


def convert_affine(shp, transform: affine.Affine, dtype=np.float64):
    op = lambda coords: np.array(transform * coords.T, dtype=dtype).T
    return convert_shp(shp, op)


def convert_shp(shp, op: callable):
    if isinstance(shp, np.ndarray):
        shp = shp.copy()
    coords = shapely.get_coordinates(shp).astype(np.float64)
    coords_transformed = op(coords)
    shp = shapely.set_coordinates(shp, coords_transformed)
    return shp


def count_group(df, column, index):
    # Group by the column, count and reindex to catch rows with a count of 0
    return df.groupby(column).count().reindex(index=index).HYBAS_ID.fillna(0).astype(int).values


# Raster geometry utilities


def resample_xr(
    arr: xarray.Dataset,
    bounds: tuple[int, int, int, int],
    size: tuple[int, int],
    method: str = "nearest",
    x_key: str = "x",
    y_key: str = "y",
):
    xlo, ylo, xhi, yhi = bounds
    # NOTE: linspace is inclusive of hi (endpoint=True is default), and
    #       xarray treats coordinates as pixel centers.
    xs = np.linspace(xlo, xhi, size[1])
    ys = np.linspace(yhi, ylo, size[0])
    kwargs = {x_key: xs, y_key: ys}
    return arr.interp(**kwargs, method=method, kwargs={"fill_value": "extrapolate"})


def resample_bilinear_subpixel(arr, bounds):
    """The special case for bilinear interpolation where you only want sub-pixel resampling"""
    xlo, ylo, xhi, yhi = bounds
    xoff = xlo % 1
    yoff = ylo % 1
    w = round(xhi - xlo)
    h = round(yhi - ylo)
    x = int(xlo)
    y = int(ylo)

    xlin = (
        arr[..., y : y + h + 1, x : x + w] * (1 - xoff)
        + arr[..., y : y + h + 1, x + 1 : x + w + 1] * xoff
    )
    ylin = xlin[..., :h, :] * (1 - yoff) + xlin[..., 1 : h + 1, :] * yoff
    # This is most similar I could come up with to rasterio's resampling;
    # When including the y would cause a nan, just use x resample.
    out = np.where(np.isnan(ylin), xlin[..., 1 : h + 1, :], ylin)
    return out


@functools.cache
def tif_data_ram(fpath: Path):
    memfile = rasterio.MemoryFile()
    with open(fpath, "rb") as tif:
        memfile.write(tif.read())
    tif = rasterio.open(memfile, "r+")
    return tif


@functools.cache
def nc_data_ram(fpath: Path, start: datetime.datetime = None, end: datetime.datetime = None):
    ds = xarray.open_dataset(fpath)
    if start is not None:
        ds = ds.sel(time=slice(start, end))
    return ds.compute()


def _get_tile_cached(p, *args, **kwargs):
    if isinstance(p, Path):
        tif = tif_data_ram(p)
    else:
        tif = tif_data_ram(p.name)
    return _get_tile_uncached(tif, *args, **kwargs)


def _get_tile_uncached(
    p: Union[Path, rasterio.DatasetReader],
    bounds: tuple[float, float, float, float] = None,
    bounds_px: tuple[int, int, int, int] = None,
    bounds_in_px: bool = False,
    align: bool = False,
):
    if isinstance(p, Path):
        tif = rasterio.open(p)
    else:
        tif = p

    if bounds is not None and not bounds_in_px:
        window = shapely_bounds_to_rasterio_window(bounds, tif.transform, align=align)
    elif (bounds is not None and bounds_in_px) or bounds_px is not None:
        window = shapely_bounds_to_rasterio_window(bounds_px, align=align)
    else:
        raise Exception("Either bounds or bounds_px must have a value")

    result = tif.read(window=window, resampling=rasterio.enums.Resampling.bilinear)

    if isinstance(p, Path):
        tif.close()

    return result


def get_tile(*args, cache: bool = False, **kwargs):
    if cache:
        return _get_tile_cached(*args, **kwargs)
    else:
        return _get_tile_uncached(*args, **kwargs)


if os.environ.get("CACHE_LOADED_TILES_IN_RAM", "no")[0].lower() == "y":
    # TODO: Fix this mess of caching. In one case we need to put a limit, in the other we don't.
    #       os.environ sets a max, but passing the cache argument does not.
    get_tile = functools.lru_cache(maxsize=16384)(get_tile)


def get_tiles_single(imgs, geom: shapely.Geometry, geom_in_px: bool = False):
    """Get window from a list of images for a single model run"""
    inps = []
    for p in imgs:
        inp = get_tile(p, geom.bounds, bounds_in_px=geom_in_px)
        inps.append(torch.tensor(inp)[None].cuda())
    return inps


def get_tiles_batched(imgs, geoms: shapely.Geometry, geom_in_px: bool = False):
    """Get windows from a list of images for a batched model run"""
    inps = []
    for p in imgs:
        img_windows = [get_tile(p, geom.bounds, bounds_in_px=geom_in_px) for geom in geoms]
        img_batch = np.stack(img_windows)
        inps.append(torch.tensor(img_batch).cuda())
    return inps


# Basin utilities


def majority_tile_mask_for_basin(
    in_tiles: Sequence[Union[np.ndarray, shapely.Geometry]], basins_df: geopandas.GeoDataFrame
):
    """
    Create a mask to exclude tiles outside the majority basin
    (which basin is the majority basin is also calculated here)
    """
    # Extract shapes
    tiles = np.array([shapely.Polygon(t) for t in in_tiles])
    tile_centers = np.array([shapely.centroid(p) for p in tiles])
    basin_geoms = np.array(basins_df.geometry.values)

    # Identify basin by majority voting across tiles
    tiles_in_basins = shapely.within(tile_centers[None], basin_geoms[:, None])
    basin_idx = np.argmax(tiles_in_basins.sum(axis=1)).item()
    hybas_id = basins_df.iloc[basin_idx].HYBAS_ID

    # Calculate a mask to remove tiles
    # Remove tiles in another basin, but keep a few tiles at the ocean's edge
    other_basins = np.concatenate(
        [tiles_in_basins[:basin_idx], tiles_in_basins[(basin_idx + 1) :]]
    )
    exclude_mask = np.any(other_basins, axis=0)
    basin_geom = basin_geoms[basin_idx]
    just_a_bit_of_ocean = shapely.buffer(basin_geom, 0.04).simplify(0.01)
    exclude_mask |= ~shapely.within(tile_centers, just_a_bit_of_ocean)

    return hybas_id, exclude_mask


def get_upstream_basins(basins, basin_id):
    """
    Create a geopandas dataset containing just the basins which feed into
    the basin_row

    Args:
        basins (geopandas.GeoDataFrame): All basin shapes
        basin_id (int): Row from basins of the basin in question

    Returns:
        geopandas.GeoDataFrame: All basin shapes that share a river
    """
    # Quickly filter to basins in the same river system
    basin_row = basins.loc[basins["HYBAS_ID"] == basin_id]
    basins_w_same_sink = basins[basins["NEXT_SINK"] == basin_row["NEXT_SINK"].item()]

    # Collect all upstream. Scans for upstream basins in waves.
    # Start with the basin in question
    incl_ids = {basin_id}
    include_list = [basins_w_same_sink[basins_w_same_sink["HYBAS_ID"] == basin_id]]
    while True:
        # Add new basins that feed into the latest layer of basins
        new_included = basins_w_same_sink[basins_w_same_sink["NEXT_DOWN"].isin(incl_ids)]
        include_list.append(new_included)

        # Update ids to check for the next layer of upstream basins.
        incl_ids = set(new_included["HYBAS_ID"].to_numpy().tolist())
        if len(incl_ids) == 0:
            break

    # Concat the geodataframes back into a single dataframe
    return geopandas.GeoDataFrame(pd.concat(include_list, ignore_index=True))


# Torch utils


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_todevice(c, device) for c in x]
    else:
        return x


def seed_packages(seed):
    # Delay until needed 'cause expensive
    import torch.cuda

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


# Torch nanops - https://github.com/pytorch/pytorch/issues/61474#issuecomment-1735537507
def torch_nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).amax(dim=dim, keepdim=keepdim)
    return output


def torch_nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).amin(dim=dim, keepdim=keepdim)
    return output


def torch_nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def torch_nanstd(tensor, dim=None, keepdim=False):
    output = torch_nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


# The ol' misc. functions


def pair(str):
    """For commandline argument parsing"""
    k, v = str.split("=")
    if v.isdigit():
        v = int(v)
        # TODO: parse as yml for datatypes, maybe?
    elif v == "True" or v == "true":
        v = True
    elif v == "False" or v == "false":
        v = False
    else:
        try:
            v = float(v)
        except ValueError:
            pass

    return k, v


def save_tiles(tiles, tile_stats, fpath, crs):
    n_bg, n_pw, n_fl = list(zip(*tile_stats))
    df = geopandas.GeoDataFrame(
        {
            "geometry": tiles,
            "n_background": n_bg,
            "n_permanent_water": n_pw,
            "n_flooded": n_fl,
        },
        geometry="geometry",
        crs=crs,
    )
    df.to_file(fpath)


def get_s1_stem_from_meta(meta: dict):
    date_str = datetime.datetime.fromisoformat(meta["pre1_date"]).strftime("%Y-%m-%d")
    return f"{meta['key']}-{date_str}"


def meta_in_date_range(meta: dict, valid_date_range: tuple, window: int):
    start, end = valid_date_range
    post_d = datetime.datetime.fromisoformat(meta["post_date"])
    window_start = post_d - datetime.timedelta(days=window)
    return (window_start.timestamp() >= start.timestamp()) and (
        post_d.timestamp() <= end.timestamp()
    )


def get_valid_date_range(era5_folder, era5L_folder, all_fnames, weather_window: int):
    era5_date_fmt = "%Y-%m"
    weather_delta = datetime.timedelta(days=weather_window)
    # filenames like *-YYYY-mm.tif
    era5_filenames = list((era5_folder).glob("*"))
    era5_filenames.sort()
    era5_start_str = "-".join(era5_filenames[0].stem.split("-")[-2:])
    era5_start = datetime.datetime.strptime(era5_start_str, era5_date_fmt)
    era5_start += weather_delta
    era5_end_str = "-".join(era5_filenames[-1].stem.split("-")[-2:])
    era5_end = datetime.datetime.strptime(era5_end_str, era5_date_fmt)

    era5L_filenames = list((era5L_folder).glob("*"))
    era5L_filenames.sort()
    era5L_start_str = "-".join(era5L_filenames[0].stem.split("-")[-2:])
    era5L_start = datetime.datetime.strptime(era5L_start_str, era5_date_fmt)
    era5L_start += weather_delta
    era5L_end_str = "-".join(era5L_filenames[-1].stem.split("-")[-2:])
    era5L_end = datetime.datetime.strptime(era5L_end_str, era5_date_fmt)

    fname_dates = []
    for fname in all_fnames:
        # fnames like '*-YYYY-MM-DD-meta.json'
        date_str = "-".join(fname.split("-")[-4:-1])
        fname_dates.append(datetime.datetime.fromisoformat(date_str))
    fname_dates.sort()
    if len(all_fnames) > 0:
        fname_start = fname_dates[0]
        fname_end = fname_dates[-1]
    else:
        # If no fnames provided, limit date range to just era5 files
        fname_start = era5L_start
        fname_end = era5L_end

    start = max([era5_start, era5L_start, fname_start])
    end = min([era5_end, era5L_end, fname_end])
    return start, end


def flatten_classes(tile: np.ndarray, n_classes: int):
    tile[tile > 2] = -100  # This index will be ignored
    if n_classes == 2:
        # Flatten "flooded"/"permanent water" to just "water"
        tile[tile == 2] = 1
    return tile
