import collections
import datetime
import functools
import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib
import tarfile
import zipfile

import asf_search as asf
import einops
import geopandas
import numpy as np
import pandas
import rasterio
import shapely
import xarray

from . import util
from . import constants


class URLNotAvailable(Exception):
    pass


_tried = {}


def download_url(url, max_tries=3, verbose=False):
    if url in _tried:
        # Trying to be a good denizen of the internet and not spam the endpoint
        # if we know that it's not available
        raise URLNotAvailable()
    if verbose:
        print("Downloading", url)
    for i in range(max_tries):
        try:
            with urllib.request.urlopen(url) as f:
                return f.read()
        except Exception as e:
            if verbose:
                print("Error", e)
            elif i < (max_tries - 1):
                print("Server returned error. Trying again in a few seconds.")
                time.sleep(3)
    _tried[url] = True
    if verbose:
        print("URL not available.")
    raise URLNotAvailable()


def degrees_to_char(north: int, east: int):
    ns = "N"
    if north < 0:
        ns = "S"
        north = -north
    ew = "E"
    if east < 0:
        ew = "W"
        east = -east
    return ns, ew, north, east


def degrees_to_north_east(north: int, east: int):
    ns, ew, north, east = degrees_to_char(north, east)
    return f"{ns}{north:02d}", f"{ew}{east:03d}"


def get_cop_dem_file(
    north: int, east: int, folder: Path, name: str, download_fnc: callable = download_url
):
    """
    Allowed names:
        COP-DEM_GLO-30-DGED/2021_1
        COP-DEM_GLO-90-DGED/2021_1
        COP-DEM_GLO-30-DGED/2022_1
        COP-DEM_GLO-90-DGED/2022_1
        COP-DEM_GLO-30-DTED/2023_1
        COP-DEM_GLO-90-DGED/2023_1
        COP-DEM_GLO-30-DGED/2023_1
        COP-DEM_GLO-90-DTED/2023_1
    """
    # Check already downloaded
    north_str, east_str = degrees_to_north_east(north, east)
    tile_name = f"Copernicus_DSM_10_{north_str}_00_{east_str}_00"
    final_fpath = folder / "dem" / name / f"{tile_name}.tif"
    if final_fpath.exists():
        return final_fpath
    final_fpath.parent.mkdir(parents=True, exist_ok=True)

    # Download tar
    base_url = "https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload"
    url = f"{base_url}/{name}/{tile_name}.tar"
    bin_data = download_fnc(url)

    # Unpack tar and convert to tif (Subprocess to GDAL because nothing else knows what it is)
    fp = io.BytesIO(bin_data)
    tmp_fpath = final_fpath.with_suffix(".dt2")
    with tarfile.TarFile(fileobj=fp) as unzipped:
        dt2_fname_in_zip = next((m for m in unzipped.getmembers() if ".dt2" in m.name))
        dt_fp = unzipped.extractfile(dt2_fname_in_zip)
    with tmp_fpath.open("wb") as tmp_fp:
        tmp_fp.write(dt_fp.read())
    fp.close()
    dt_fp.close()
    tile_opts = ["-co", "TILED=YES", "-co", "blockxsize=256", "-co", "blockysize=256"]
    comp_opts = ["-co", "COMPRESS=PACKBITS"]
    subprocess.run(["gdal_translate", tmp_fpath, final_fpath, *tile_opts, *comp_opts])
    tmp_fpath.unlink()
    print(f"Downloaded {tile_name}.tif to {final_fpath}")
    return final_fpath


def get_srtm_dem_file(north: int, east: int, folder: Path):
    # SRTM is in 5x5 degree rectangles, lat in [60, -60], long in [-180, 180]
    n = (90 - (north + 30)) // 5 + 1
    e = (east + 180) // 5 + 1
    tile_name = f"srtm_{e:02d}_{n:02d}"
    # Check already downloaded
    final_fpath = folder / "dem" / "SRTM 3Sec" / f"{tile_name}.tif"
    if final_fpath.exists():
        return final_fpath

    # Download zip
    base_url = "https://download.esa.int/step/auxdata/dem/SRTM90/tiff"
    url = f"{base_url}/{tile_name}.zip"
    bin_data = download_url(url)

    # Save as tiled tif
    fp = io.BytesIO(bin_data)
    tmp_folder = final_fpath.parent / "tmp"
    tmp_folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(fp) as unzipped:
        fname_in_zip = next((m for m in unzipped.filelist if ".tif" in m.filename))
        unzipped.extract(fname_in_zip, tmp_folder)
    fp.close()
    tmp_fpath = tmp_folder / fname_in_zip.filename
    tile_opts = ["-co", "TILED=YES", "-co", "blockxsize=256", "-co", "blockysize=256"]
    comp_opts = ["-co", "COMPRESS=PACKBITS"]
    subprocess.run(["gdal_translate", tmp_fpath, final_fpath, *tile_opts, *comp_opts])
    tmp_fpath.unlink()
    print(f"Downloaded {tile_name}.tif to {final_fpath}")
    return final_fpath


def get_world_cover_file(north: int, east: int, folder: Path):
    base_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    year = 2021
    ver = "v200"

    north_str, east_str = degrees_to_north_east(north, east)
    tile = f"{north_str}{east_str}"
    fname = f"ESA_WorldCover_10m_{year}_{ver}_{tile}_Map.tif"
    fpath = folder / fname
    if fpath.exists():
        return fpath

    url = f"{base_url}/{ver}/{year}/map/{fname}"
    bin_data = download_url(url)
    folder.mkdir(exist_ok=True)
    with open(fpath, "wb") as f:
        f.write(bin_data)
    return fpath


def get_global_surface_water_file(
    north: int,
    east: int,
    folder: Path,
    date: datetime.datetime,
    previous_year: bool = False,
    ver: str = "LATEST",
):
    if previous_year:
        # To get to the previous year, go to the first day of the year and subtract a day.
        date = date.replace(month=1, day=1) - datetime.timedelta(days=1)
    year = date.year
    base_url = (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/"
        + f"YearlyClassification/{ver}/tiles/yearlyClassification{year}"
    )

    # GSWE observations numbers tiles with f'000{num:03d}0000'
    # From num=0 to num=52 represent 80d to -50d latitude, in increments of 4 per 10d
    # From num=0 to num=140 represent -180d to 170d longitude, in increments of 4 per 10d
    # Lat, lon are top-left of tile, but provided north is bottom of required latitude.
    north += 10
    n = round((80 - north) * 4 / 10)
    e = round((east + 180) * 4 / 10)
    fname = f"yearlyClassification{year}-000{n:03d}0000-000{e:03d}0000.tif"
    fpath = folder / "gswe" / fname
    if fpath.exists():
        return fpath

    url = f"{base_url}/{fname}"
    bin_data = download_url(url)
    fpath.parent.mkdir(exist_ok=True)
    with open(fpath, "wb") as f:
        f.write(bin_data)

    return fpath


def get_hand_file(north: int, east: int, folder: Path):
    north_str, east_str = degrees_to_north_east(north, east)
    base_url = "https://glo-30-hand.s3.amazonaws.com/v1/2021"
    fname = f"Copernicus_DSM_COG_10_{north_str}_00_{east_str}_00_HAND.tif"

    hand_folder = folder / "dem" / "HAND"
    final_fpath = hand_folder / f"{fname}"
    if final_fpath.exists():
        return final_fpath

    bin_data = download_url(f"{base_url}/{fname}")
    hand_folder.mkdir(exist_ok=True)
    with open(final_fpath, "wb") as f:
        f.write(bin_data)
    return final_fpath


def get_nbyn_product(shp, shp_crs, folder, getter, n=1, preprocess=None, **kwargs):
    """
    Get a product tiled with n-by-n degree squares.

    Saves downloaded product to folder.

    Returns (computed) xarray in minimal bounding box around shp.
    Automatically handles the case that shp goes over multiple product tile boundaries
    """
    if shp_crs != "EPSG:4326":
        shp4326 = util.convert_crs(shp, shp_crs, "EPSG:4326")
    else:
        shp4326 = shp

    xlo, ylo, xhi, yhi = shp4326.bounds
    ew_ran = range(math.floor(xlo / n) * n, math.ceil(xhi / n) * n, n)
    ns_ran = range(math.floor(ylo / n) * n, math.ceil(yhi / n) * n, n)
    paths = [getter(ns, ew, folder, **kwargs) for ew, ns in itertools.product(ew_ran, ns_ran)]
    data = xarray.open_mfdataset(paths, preprocess=preprocess)
    box = data.sel(x=slice(xlo, xhi), y=slice(yhi, ylo))

    return box.compute()


def get_world_cover(shp, shp_crs, folder):
    return get_nbyn_product(shp, shp_crs, folder, get_world_cover_file, n=3)


def get_global_surface_water(shp, shp_crs, folder, date):
    return get_nbyn_product(shp, shp_crs, folder, get_global_surface_water_file, n=10, date=date)


def _drop_last_pixel(ds):
    return ds.isel(x=slice(len(ds.x) - 1), y=slice(len(ds.y) - 1))


def get_hand(shp, shp_crs, folder):
    return get_nbyn_product(shp, shp_crs, folder, get_hand_file, n=1)


def get_dem(shp, shp_crs, folder, name="COP-DEM_GLO-30-DTED__2023_1"):
    if "COP-DEM" in name:
        return get_nbyn_product(
            shp, shp_crs, folder, get_cop_dem_file, n=1, preprocess=_drop_last_pixel, name=name
        )
    elif name == "SRTM 3Sec":
        return get_nbyn_product(shp, shp_crs, folder, get_srtm_dem_file, n=5, name=name)
    else:
        raise ValueError(f"Unknown DEM: '{name}'")


if os.environ.get("CACHE_LOADED_TILES_IN_RAM", "no")[0].lower() == "y":
    get_dem = functools.lru_cache(maxsize=3000)(get_dem)


def download_s1(img_folder, asf_result, cred_fname=".asf_auth"):
    if not isinstance(asf_result, dict):
        asf_result = asf_result.geojson()
    zip_fpath = img_folder / asf_result["properties"]["fileName"]
    if not (zip_fpath.exists()):
        print("Downloading S1 image: ", zip_fpath)
        with open(cred_fname) as f:
            cred = json.load(f)
            username, password = cred["user"], cred["pass"]
        session = asf.ASFSession().auth_with_creds(username, password)
        url = asf_result["properties"]["url"]
        asf.download_url(url, path=img_folder, session=session)
    return zip_fpath


def preprocess_s1(data_folder: Path, s1_fname: Path, out_fname: Path):
    img_folder = data_folder / "s1"
    assert (img_folder / s1_fname).exists(), "Sentinel 1 file not downloaded"
    if (img_folder / "tmp" / out_fname).exists():
        return
    tmp_folder = Path("tmp")
    (img_folder / tmp_folder).mkdir(exist_ok=True)

    def mk_args(graph_name):
        return [
            "docker",
            "run",
            "--rm",
            "-it",
            "-v",
            f"{str(data_folder)}/dem:/root/.snap/auxdata/dem",
            "-v",
            f"{str(data_folder)}/Orbits:/root/.snap/auxdata/Orbits",
            "-v",
            f"{str(img_folder)}:/data",
            "esa-snappy",
            "bash",
            "-c",
            f"gpt {graph_name} -c 12G -Ssource=/data/{s1_fname} -t /data/tmp/{out_fname} "
            f"&& chown -R {os.getuid()} /data/tmp/{out_fname}"
            f"&& chown -R {os.getuid()} /data/tmp/{out_fname.with_suffix('.data')}",
        ]

    print(f"Preprocessing {img_folder / s1_fname} to {img_folder}/{tmp_folder}/{out_fname}")
    args = mk_args("graph.xml")
    process_result = subprocess.run(args)

    if not (img_folder / tmp_folder / out_fname).exists():
        # At least one can't have noise removal applied, so retry with no noise removal
        args = mk_args("graph_nonoise.xml")
        process_result = subprocess.run(args)

    if not (img_folder / tmp_folder / out_fname).exists():
        raise Exception("Docker run failed for some reason.")


def load_dfo(path: Path, for_s1: bool = False):
    dfo = geopandas.read_file(path / "FloodArchive_region.shp")
    dfo.geometry = dfo.geometry.buffer(0)
    classification_types = pandas.read_csv(path / "classifications.csv", quotechar="'")

    # Filter by known flood types we care about
    dfo = dfo.copy(deep=True)
    ct = classification_types
    care_groups = ct[ct["group"].isin(constants.DFO_INCLUDE_TYPES)]
    care_names = ";".join(care_groups["all_names"]).split(";")
    dfo = dfo[dfo["MAINCAUSE"].isin(care_names)]

    if for_s1:
        # Only include 2014 onwards
        dfo = dfo[dfo["BEGAN"] >= "2014-01-01"]

    return dfo


@functools.lru_cache(maxsize=None)
def _era5_band_index(p: Path):
    """
    Internally, the tif file has a flat list of bands which represent
    the different keys on different days.
    This function returns all of the key idxs in a dict structure so
    that it's easier to pull out just the parts you need.

    Access like band_idxs['YYYY-MM-DD'][key]

    In retrospect, perhaps a netcdf file would have been better,
    since it can have a third indexing dimension of time. TODO.
    """
    band_index = {}
    with rasterio.open(p) as tif:
        for i, name in enumerate(tif.descriptions):
            parts = name.split("-")
            day = "-".join(parts[:3])
            k = parts[-1]
            if day not in band_index:
                band_index[day] = {}
            band_index[day][k] = i + 1
    return band_index


def load_era5(
    folder: Path,
    geom: shapely.Geometry,
    res: int | tuple[int, int],
    start: datetime.datetime,
    end: datetime.datetime,
    keys: list[str] = None,
    era5_land: bool = True,
):
    """For reading from raw data exported from GEE"""
    if isinstance(res, int):
        res = (res, res)
    if keys is None:
        if era5_land:
            keys = constants.ERA5L_BANDS
        else:
            keys = constants.ERA5_BANDS
    # Calculate the indices to read
    n_bands = len(keys)
    to_get = collections.defaultdict(lambda: [])
    current = start
    while current <= end:
        if era5_land:
            fname = f"era5-land-{current.year}-{current.month:02d}.tif"
        else:
            fname = f"era5-{current.year}-{current.month:02d}.tif"

        band_index = _era5_band_index(folder / fname)
        day_str = current.strftime("%Y-%m-%d")
        band_idxs = [band_index[day_str][k] for k in keys]
        to_get[fname].append(band_idxs)
        current += datetime.timedelta(days=1)

    results = []
    out_keys = []
    for fname, all_day_idxs in to_get.items():
        with rasterio.open(folder / fname) as tif:
            window = util.shapely_bounds_to_rasterio_window(
                geom.bounds, tif.transform, align=False
            )
            band_idxs = [idx for day_idxs in all_day_idxs for idx in day_idxs]
            data = tif.read(
                band_idxs,
                window=window,
                out_shape=res,
                resampling=rasterio.enums.Resampling.bilinear,
            )
            band_idxs0 = np.array(band_idxs) - 1  # Make them 0-indexed
            for i in band_idxs0:
                out_keys.append(tif.descriptions[i])

            scales = np.array(tif.scales)[band_idxs0, None, None]
            offsets = np.array(tif.offsets)[band_idxs0, None, None]
            nan_mask = data == tif.nodata
            data = data * scales + offsets
            data[nan_mask] = np.nan
            n_img = len(all_day_idxs)
            data = einops.rearrange(data, "(I B) H W -> I B H W", I=n_img, B=n_bands)
            results.extend(data)

    return out_keys, results


@functools.cache
def _era5_exported_band_idxs(
    fpath: Path, start: datetime.datetime, end: datetime.datetime, keys: tuple[str]
):
    # Calculate the indices to read
    band_index = _era5_band_index(fpath)
    day_strs = []
    current = start
    while current <= end:
        day_strs.append(current.strftime("%Y-%m-%d"))
        current += datetime.timedelta(days=1)

    band_idxs = tuple([band_index[day_str][k] for day_str in day_strs for k in keys])
    return day_strs, band_idxs


def load_exported_era5(
    fpath: Path,
    geom: shapely.Geometry,
    res: tuple[int, int],
    start: datetime.datetime,
    end: datetime.datetime,
    keys: list[str],
    cache_in_ram: bool = False,
):
    """For reading an exported raster already at the right resolution"""
    day_strs, band_idxs = _era5_exported_band_idxs(fpath, start, end, tuple(keys))

    if cache_in_ram:
        tif = util.tif_data_ram(fpath)
    else:
        tif = rasterio.open(fpath)

    T = tif.transform
    window = util.shapely_bounds_to_rasterio_window(geom.bounds, T, align=False)
    method = rasterio.enums.Resampling.bilinear
    data = tif.read(band_idxs, window=window, out_shape=res, resampling=method)

    if not cache_in_ram:
        tif.close()

    data = einops.rearrange(data, "(I B) H W -> I B H W", I=len(day_strs), B=len(keys))

    return data


def load_exported_era5_nc(
    fpath: Path,
    geom: shapely.Geometry,
    res: tuple[int, int],
    start: datetime.datetime,
    end: datetime.datetime,
    keys: list[str],
    cache_in_ram: bool = False,
):
    """For reading an exported raster already at the right resolution"""
    # The end date is whenever the S1 image was taken (middle of the day)
    # But the dates in the data are the first second of the day.
    if not (start.hour == 0 and start.minute == 0 and start.second == 0):
        start_incl_first = start - datetime.timedelta(days=1)
    else:
        start_incl_first = start

    if cache_in_ram:
        ds = util.nc_data_ram(fpath, start_incl_first, end)
    else:
        ds = xarray.open_dataset(fpath)
        ds = ds.sel(time=slice(start_incl_first, end))

    # Get image data - offsets to match whatever rasterio was doing
    xlo, xhi, ylo, yhi = geom.bounds
    xlo += constants.ERA5L_DEGREES_PER_PIXEL / 4
    ylo -= constants.ERA5L_DEGREES_PER_PIXEL / 4
    xhi += constants.ERA5L_DEGREES_PER_PIXEL / 4
    yhi -= constants.ERA5L_DEGREES_PER_PIXEL / 4
    resample_ds = util.resample_xr(
        ds, (xlo, xhi, ylo, yhi), res, "linear", "longitude", "latitude"
    )
    subset_np = np.stack(
        [getattr(resample_ds, band).values for band in keys], axis=1, dtype=np.float32
    )

    return subset_np


def load_pregenerated_raster(
    fpath: Path,
    geom: shapely.Geometry,
    res: tuple[int, int],
    band_idxs: tuple[int] = (1,),
    cache_in_ram: bool = False,
):
    if cache_in_ram:
        tif = util.tif_data_ram(fpath)
    else:
        tif = rasterio.open(fpath)

    window = util.shapely_bounds_to_rasterio_window(geom.bounds, tif.transform, align=False)

    data = tif.read(
        band_idxs,
        window=window,
        out_shape=res,
        resampling=rasterio.enums.Resampling.bilinear,
    )

    if not cache_in_ram:
        tif.close()
    return data


def get_climate_zone_from_static(folder: Path, geom: shapely.Geometry):
    fpath = folder / constants.HYDROATLAS_RASTER_FNAME
    return get_climate_zone(fpath)


def get_climate_zone(tif: rasterio.DatasetReader, geom: shapely.Geometry, geom_crs: str):
    geom_in_crs = util.convert_crs(geom, geom_crs, tif.crs)
    px = util.convert_affine(geom_in_crs, ~tif.transform)
    x, y = shapely.get_coordinates(px)[0]
    window = ((y - 0.5, y + 0.5), (x - 0.5, x + 0.5))
    band_idx = 1 + tif.descriptions.index(constants.HYDROATLAS_CLIMATE_ZONE_BAND_NAME)
    val = tif.read(band_idx, window=window).item()
    # There's a weird problem where nans are sometimes being read as a very small value
    # Sooo.... uh.....
    if np.isnan(val) or val < 0.001:
        return None
    else:
        return int(val)


def ks_water_stats(tile: np.ndarray):
    flood = tile == constants.KUROSIWO_FLOOD_CLASS
    pwater = tile == constants.KUROSIWO_PW_CLASS
    bg = tile == constants.KUROSIWO_BG_CLASS
    return bg.sum(), pwater.sum(), flood.sum()


def load_glofas(
    fpath: Path,
    geom: shapely.Geometry,
    res: tuple[int, int],
    start: datetime.datetime,
    end: datetime.datetime,
    bands: list[str] = ["dis24", "rowe", "swir"],
    cache_in_ram: bool = False,
):
    if cache_in_ram:
        ds = util.nc_data_ram(fpath)
    else:
        ds = xarray.open_dataset(fpath)

    # GloFAS is sometimes in in 0-360, but geom is always in -180-180
    # (When the area contained within the site crosses the 0-360 boundary,
    #   GloFAS uses negatives to ensure contiguous indices)
    xlo, ylo, xhi, yhi = geom.bounds
    lon_lo, lon_hi = ds.longitude.min().item(), ds.longitude.max().item()
    if lon_lo > 180:
        xlo, ylo, xhi, yhi = (xlo + 360, ylo, xhi + 360, yhi)

    # The end date is whenever the S1 image was taken (middle of the day)
    # But the dates in glofas are set at the first second of the day
    # So to include the correct number of days, we need to add a one-day buffer.
    # E.g. start is day 54.3 and end is day 74.3, and glofas "days" are at:
    #   54.0, 55.0, 56.0, 57.0, etc.
    # We need to include a date range of [53.3, 74.3]
    # (Note I've used fractional days here for illustration only, really they're datetime objects)
    start_incl_first = start - datetime.timedelta(days=1)
    subset_ds = ds.sel(
        time=slice(start_incl_first, end),
        longitude=slice(xlo, xhi),
        latitude=(slice(yhi, ylo)),
    )

    # Normal image resampling doesn't work for GloFAS because it's actually a single-pixel-width
    # path represented in an image. Consider a 2x2 influence for each output pixel. The magnitude
    # of the output pixel is then dependent on whether it includes 1, 2 or 3 pixels from the path.
    # Instead, we'll run a naive majority filter on each 2x2.
    width = getattr(subset_ds, bands[0]).values.shape[-1]
    subset_np = np.stack([getattr(subset_ds, band).values for band in bands], axis=1)
    width = subset_np.shape[-1]
    assert (
        res[1] == width or res[1] == width / 2
    ), "GloFAS custom resampling only works on full or half resolution"
    if res[1] == width / 2:
        c00 = subset_np[:, :, ::2, ::2]
        c01 = subset_np[:, :, ::2, 1::2]
        c10 = subset_np[:, :, 1::2, ::2]
        c11 = subset_np[:, :, 1::2, 1::2]
        subset_np = np.stack([c00, c01, c10, c11], axis=0).max(axis=0)

    return subset_np
