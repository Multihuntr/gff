import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import urllib
import tarfile
import zipfile

import asf_search as asf

import xarray

from . import util


def download_url(url):
    with urllib.request.urlopen(url) as f:
        return f.read()


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


def get_cop_dem_file(north: int, east: int, name: str, folder: Path):
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
    print(f"Downloading {tile_name}.tif to {final_fpath}")
    final_fpath.mkdir(parents=True, exist_ok=True)

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


def get_srtm_dem_file(north: int, east: int, folder: Path):
    # SRTM is in 5x5 degree rectangles, lat in [60, -60], long in [-180, 180]
    n = (90 - (north + 30)) // 5 + 1
    e = (east + 180) // 5 + 1
    tile_name = f"srtm_{e:02d}_{n:02d}"
    # Check already downloaded
    final_fpath = folder / "dem" / "SRTM 3Sec" / f"{tile_name}.tif"
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


def get_nbyn_product(shp, shp_crs, folder, getter, n=1, **kwargs):
    """
    Get a product tiled with n-by-n degree squares.

    Saves downloaded product to folder.

    Returns (computed) xarray in minimal bounding box around shp.
    Automatically handles the case that shp goes over multiple product tile boundaries
    """
    if shp_crs != "EPSG:4326":
        util.convert_crs(shp, shp_crs, "EPSG:4326")
    else:
        shp4326 = shp

    xlo, ylo, xhi, yhi = shp4326.bounds
    ew_ran = range(math.floor(xlo / n) * n, math.ceil(xhi / n) * n, n)
    ns_ran = range(math.floor(ylo / n) * n, math.ceil(yhi / n) * n, n)
    paths = [getter(ns, ew, folder, **kwargs) for ew, ns in itertools.product(ew_ran, ns_ran)]
    dem = xarray.open_mfdataset(paths)
    box = dem.sel(x=slice(xlo, xhi), y=slice(yhi, ylo))

    return box.compute()


def get_world_cover(shp, shp_crs, folder):
    return get_nbyn_product(shp, shp_crs, folder, get_world_cover_file, n=3)


def get_dem(shp, shp_crs, folder, name="COP-DEM_GLO-30-DTED__2023_1"):
    if "COP-DEM" in name:
        return get_nbyn_product(shp, shp_crs, folder, get_cop_dem_file, n=1, name=name)
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
    args = [
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
        f"gpt graph.xml -c 12G -Ssource=/data/{s1_fname} -t /data/tmp/{out_fname} "
        f"&& chown -R {os.getuid()} /data/tmp/{out_fname}"
        f"&& chown -R {os.getuid()} /data/tmp/{out_fname.with_suffix('.data')}",
    ]
    print(f"Preprocessing {img_folder / s1_fname} to {img_folder}/{tmp_folder}/{out_fname}")
    process_result = subprocess.run(args)

    if not (img_folder / tmp_folder / out_fname).exists():
        raise Exception("Docker run failed for some reason.")
