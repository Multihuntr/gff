import datetime
import math

import geopandas
import pyproj
import rasterio
import scipy.interpolate
import shapely
import numpy as np
import pandas as pd
import xarray


# Generic geometry utilities


def rounded_bounds(arg):
    if isinstance(arg, tuple) or (isinstance(arg, np.ndarray) and arg.shape == (4,)):
        xlo, ylo, xhi, yhi = arg
        return math.floor(xlo), math.floor(ylo), math.ceil(xhi), math.ceil(yhi)
    elif isinstance(arg, shapely.Geometry):
        return rounded_bounds(shapely.bounds(arg))
    else:
        raise ValueError("Bounds must be 4-tuple, 1D np.array or a shapely.Geometry")


def shapely_bounds_to_rasterio_window(bounds, transform=None):
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

    return ((ylo, yhi), (xlo, xhi))


def image_footprint(tif_path):
    with rasterio.open(tif_path) as tif:
        coords = np.array([[0, 0], [tif.width, tif.height]])
        (xlo, ylo), (xhi, yhi) = np.array(tif.transform * coords.T).T
    return shapely.box(xlo, ylo, xhi, yhi)


def mk_box_grid(width, height, x_offset=0, y_offset=0, box_width=1, box_height=1):
    """
    Create a grid of box geometries, stored in a vectorised Shapely array.
    """
    xs = np.arange(width) * box_height
    ys = np.arange(height) * box_width
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


def convert_crs(shp: shapely.Geometry, _from: str, _to: str):
    project = pyproj.Transformer.from_crs(_from, _to, always_xy=True).transform
    return shapely.ops.transform(project, shp)


# Raster geometry utilities


def resample(arr: xarray.Dataset, bounds: tuple[int, int, int, int], size: tuple[int, int]):
    xlo, ylo, xhi, yhi = bounds
    xs = np.linspace(xlo, xhi, size[1])
    ys = np.linspace(yhi, ylo, size[0])
    return arr.interp(x=xs, y=ys, method="linear", kwargs={"fill_value": "extrapolate"})


# Basin selection utilities


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


# The ol' misc. functions


def parse_date(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")
