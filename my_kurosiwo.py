import io
import itertools
import math
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Union
import urllib
import tarfile
import zipfile

import affine
import asf_search as asf
import numpy as np
import pyproj
import rasterio
import shapely
from torch import nn
import torch
import tqdm

import xarray

import constants
import util


def download_url(url):
    with urllib.request.urlopen(url) as f:
        return f.read()


def get_cop_dem_file(north: int, east: int, name: str):
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
    ns = "N"
    if north < 0:
        ns = "S"
        north = -north
    if east < 0:
        ew = "W"
        east = -east
    tile_name = f"Copernicus_DSM_10_{ns}{north:02d}_00_{ew}{east:03d}_00"
    final_fpath = Path("preprocessing") / "dem" / name / f"{tile_name}.tif"
    if final_fpath.exists():
        return final_fpath
    print(f"Downloading {tile_name}.tif to {final_fpath}")

    # Download tar
    base_url = "https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload"
    url = f"{base_url}/{name}/{tile_name}.tar"
    bin_data = download_url(url)

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
    return final_fpath


def get_srtm_dem_file(north: int, east: int):
    # SRTM is in 5x5 degree rectangles, lat in [60, -60], long in [-180, 180]
    n = (90 - (north + 30)) // 5 + 1
    e = (east + 180) // 5 + 1
    tile_name = f"srtm_{e:02d}_{n:02d}"
    # Check already downloaded
    final_fpath = Path("preprocessing") / "dem" / "SRTM 3Sec" / f"{tile_name}.tif"
    if final_fpath.exists():
        return final_fpath
    print(f"Downloading {tile_name}.tif to {final_fpath}")

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
    return final_fpath


def get_dem_file(north: int, east: int, name: str = "COP-DEM_GLO-30-DTED__2023_1"):
    if "COP-DEM" in name:
        return get_cop_dem_file(north, east, name)
    elif name == "SRTM 3Sec":
        return get_srtm_dem_file(north, east)
    else:
        raise ValueError(f"Unknown DEM: '{name}'")


def get_dem(shp, name="COP-DEM_GLO-30-DTED__2023_1", shp_crs="EPSG:4326"):
    """
    Get a dem for the shape with a minimal bounding box around it
    shp must be in xy (lon, lat)

    Return xarray in minimal bounding box around shp
    """
    if shp_crs != "EPSG:4326":
        util.convert_crs(shp, shp_crs, "EPSG:4326")
    else:
        shp4326 = shp

    xlo, ylo, xhi, yhi = shp4326.bounds
    ew_ran = range(math.floor(xlo), math.ceil(xhi))
    ns_ran = range(math.floor(ylo), math.ceil(yhi))
    dem_paths = [get_dem_file(ns, ew, name) for ew, ns in itertools.product(ew_ran, ns_ran)]
    dem = xarray.open_mfdataset(dem_paths)
    box = dem.sel(x=slice(xlo, xhi), y=slice(yhi, ylo))

    return box.compute()


def download_s1(img_folder, asf_result, cred_fname=".asf_auth"):
    zip_fpath = img_folder / asf_result.properties["fileName"]
    if not (zip_fpath.exists()):
        with open(cred_fname) as f:
            username, password = f.read().strip().split(";")
        session = asf.ASFSession().auth_with_creds(username, password)
        asf_result.download(path=img_folder, session=session)


def preprocess(
    img_folder: Path, s1_fname: Path, out_fname: Path, wkt: str = None, reprocess: bool = False
):
    assert (img_folder / s1_fname).exists(), "Sentinel 1 file not downloaded"
    if (img_folder / out_fname).exists():
        return
    if reprocess:
        tmp_str = "".join(random.choice("0123456789ABCDEF") for n in range(6))
        tmp_fname = f"{tmp_str}.tif"
    args = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{str(Path.cwd())}/preprocessing/dem:/root/.snap/auxdata/dem",
        "-v",
        f"{str(Path.cwd())}/preprocessing/Orbits:/root/.snap/auxdata/Orbits",
        "-v",
        f"{str(img_folder)}:/data",
        "esa-snappy",
        "gpt",
        "graph.xml",
        "-c",
        "12G",
        f"-Ssource=/data/{s1_fname}",
        "-t",
        f"/data/{tmp_fname}",
    ]
    if wkt is not None:
        args.append("--aoi_wkt")
        args.append(wkt)
    process_result = subprocess.run(args, capture_output=True)

    if not (img_folder / tmp_fname).exists():
        raise Exception("Docker run failed for some reason.")

    if reprocess:
        # Rewrite with rasterio; esp. compress/tile
        in_tif = rasterio.open(img_folder / tmp_fname)
        new_profile = {
            **in_tif.profile,
            "count": 2,
            "compress": "packbits",
            "nodata": np.nan,  # TODO: check
            "blockxsize": constants.FLOODMAP_BLOCK_SIZE,
            "blockysize": constants.FLOODMAP_BLOCK_SIZE,
            "tiled": True,
            "descriptions": ["vv", "vh"],
            "bigtiff": "YES",
        }

        with rasterio.open(img_folder / out_fname, "w", **new_profile) as out_tif:
            for (y, x), window in tqdm.tqdm(out_tif.block_windows(1)):
                data_block = in_tif.read(window=window)
                out_tif.write(data_block, window=window)


def run_snunet_once(
    imgs,
    geom: shapely.Geometry,
    model: nn.Module,
    geom_in_px: bool = False,
    geom_crs: str = "EPSG:3857",
):
    inps = util.get_tiles_single(imgs, geom, geom_in_px)

    geom_4326 = util.convert_crs(geom, geom_crs, "EPSG:4326")
    dem_coarse = get_dem(geom_4326)
    dem_fine = util.resample(dem_coarse, geom_4326.bounds, inps[0].shape[2:])
    dem_th = dem_fine.band_data.values
    out = model(inps, dem=dem_th)[0].cpu().numpy()
    return inps, dem_th, out


def run_flood_vit_once(imgs, geom: shapely.Geometry, model: nn.Module, geom_in_px: bool = False):
    inps = util.get_tiles_single(imgs, geom, geom_in_px)
    out = model(inps)[0].cpu().numpy()
    return inps, out


def run_flood_vit_batched(imgs, geoms: np.ndarray, model: nn.Module, geoms_in_px: bool = False):
    inps = util.get_tiles_batched(imgs, geoms, geoms_in_px)
    out = model(inps).cpu().numpy()
    return inps, out
