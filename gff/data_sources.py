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
        except:
            if i < (max_tries - 1):
                print("Server returned error. Trying again in a few seconds.")
                time.sleep(3)
    _tried[url] = True
    if verbose:
        print("URL not available.")
    raise URLNotAvailable()


def degrees_to_north_east(north: int, east: int):
    ns = "N"
    if north < 0:
        ns = "S"
        north = -north
    ew = "E"
    if east < 0:
        ew = "W"
        east = -east
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
    dem = xarray.open_mfdataset(paths, preprocess=preprocess)
    box = dem.sel(x=slice(xlo, xhi), y=slice(yhi, ylo))

    return box.compute()


def get_world_cover(shp, shp_crs, folder):
    return get_nbyn_product(shp, shp_crs, folder, get_world_cover_file, n=3)


def _drop_last_pixel(ds):
    return ds.isel(x=slice(len(ds.x) - 1), y=slice(len(ds.y) - 1))


def get_dem(shp, shp_crs, folder, name="COP-DEM_GLO-30-DTED__2023_1"):
    if "COP-DEM" in name:
        return get_nbyn_product(
            shp, shp_crs, folder, get_cop_dem_file, n=1, preprocess=_drop_last_pixel, name=name
        )
    elif name == "SRTM 3Sec":
        return get_nbyn_product(shp, shp_crs, folder, get_srtm_dem_file, n=5, name=name)
    else:
        raise ValueError(f"Unknown DEM: '{name}'")


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
    res: int,
    start: datetime.datetime,
    end: datetime.datetime,
    keys: list[str] = None,
    era5_land: bool = True,
):
    if keys is None:
        if era5_land:
            keys = constants.ERA5L_BANDS
        else:
            keys = constants.ERA5_BANDS
    results = []
    current = start
    while current <= end:
        if era5_land:
            fname = f"era5-land-{current.year}-{current.month:02d}.tif"
        else:
            fname = f"era5-{current.year}-{current.month:02d}.tif"
        band_index = _era5_band_index(folder / fname)
        day_str = current.strftime("%Y-%m-%d")
        band_idxs = [band_index[day_str][k] for k in keys]
        with rasterio.open(folder / fname) as tif:
            window = util.shapely_bounds_to_rasterio_window(
                geom.bounds, tif.transform, align=False
            )
            data = tif.read(
                band_idxs,
                window=window,
                out_shape=(res, res),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            band_idxs0 = np.array(band_idxs) - 1  # Make them 0-indexed
            scales = np.array(tif.scales)[band_idxs0, None, None]
            offsets = np.array(tif.offsets)[band_idxs0, None, None]
            data = data * scales + offsets
        results.append(data)
        current += datetime.timedelta(days=1)

    return results


def load_pregenerated_raster(
    fpath: Path, geom: shapely.Geometry, res: int, keys: list[str] | list[int] = None
):
    with rasterio.open(fpath) as tif:
        if keys is None:
            band_idxs = list(range(1, len(tif.descriptions) + 1))
        elif isinstance(keys[0], int):
            band_idxs = keys
        else:
            band_idxs = [tif.descriptions.index(k) + 1 for k in keys]
        window = util.shapely_bounds_to_rasterio_window(geom.bounds, tif.transform, align=False)
        return tif.read(
            band_idxs,
            window=window,
            out_shape=(res, res),
            resampling=rasterio.enums.Resampling.bilinear,
        )


def get_climate_zone(folder: Path, geom: shapely.Geometry):
    fpath = folder / constants.HYDROATLAS_RASTER_FNAME
    with rasterio.open(fpath) as tif:
        px = util.convert_affine_inplace(geom.centroid, ~tif.transform)
        x, y = shapely.get_coordinates(px)[0]
        window = ((y, y + 1), (x, x + 1))
        band_idx = 1 + tif.descriptions.index(constants.HYDROATLAS_CLIMATE_ZONE_BAND_NAME)
        return int(tif.read(band_idx, window=window).item())
