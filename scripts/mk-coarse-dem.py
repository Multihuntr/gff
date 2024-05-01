import argparse
import itertools
import math
from pathlib import Path
import sys

import affine
import geopandas
import numpy as np
import rasterio
import shapely
import tqdm
import xarray

import gff.constants
import gff.data_sources
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Cycles through all floodmaps and creates a dem file at the resolution matching a provided reference tif"
    )

    parser.add_argument("data_path", type=Path)
    parser.add_argument("ref_tif_path", type=Path)
    parser.add_argument(
        "context", type=int, help="Number of pixels in ref_tif as buffer around each floodmap"
    )

    return parser.parse_args(argv)


def download_url_partial(url):
    return gff.data_sources.download_url(url, max_tries=1, verbose=True)


def get_available_cop_dem(shp, shp_crs, folder, n=1, name="COP-DEM_GLO-30-DTED__2023_1"):
    if shp_crs != "EPSG:4326":
        shp4326 = gff.util.convert_crs(shp, shp_crs, "EPSG:4326")
    else:
        shp4326 = shp

    xlo, ylo, xhi, yhi = shp4326.bounds
    ew_ran = range(math.floor(xlo / n) * n, math.ceil(xhi / n) * n, n)
    ns_ran = range(math.floor(ylo / n) * n, math.ceil(yhi / n) * n, n)
    paths = []
    for ew, ns in itertools.product(ew_ran, ns_ran):
        try:
            fpath = gff.data_sources.get_cop_dem_file(
                ns, ew, folder, name=name, download_fnc=download_url_partial
            )
            paths.append(fpath)
        except gff.data_sources.URLNotAvailable:
            pass

    dem = xarray.open_mfdataset(paths, preprocess=gff.data_sources._drop_last_pixel)
    box = dem.sel(x=slice(xlo, xhi), y=slice(yhi, ylo))
    return box


INT16MIN = -32768


def main(args):
    ref_tif = rasterio.open(args.ref_tif_path)
    width, height = ref_tif.width * 2 - 1, ref_tif.height * 2 - 1
    Q = ref_tif.transform
    if Q[1] != 0:
        raise NotImplementedError("Rotated reference tifs not supported")

    res = (Q[0] / 2, Q[4] / 2)
    origin_x = Q[2] + Q[0] / 2
    origin_y = Q[5] + Q[4] / 2
    transform = rasterio.transform.from_origin(origin_x, origin_y, res[0], -res[1])

    out_fpath = args.data_path / gff.constants.COARSE_DEM_FNAME
    profile = {
        **ref_tif.profile,
        "width": width,
        "height": height,
        "transform": transform,
        "count": 1,
        "nodata": INT16MIN,
        "dtype": np.int16,
        "tiled": "yes",
        "blockxsize": (args.context * 3 // 16) * 16,
        "blockysize": (args.context * 3 // 16) * 16,
        "COMPRESS": "LERC",
        "MAX_Z_ERROR": 0,
        "INTERLEAVE": "BAND",
        "BIGTIFF": "YES",
    }
    with rasterio.open(out_fpath, "w", **profile) as out_tif:

        floodmap_path = args.data_path / "floodmaps"
        tile_geom_paths = list(floodmap_path.glob("*/????-??????????-*-visit.gpkg"))
        for i, path in enumerate(tqdm.tqdm(tile_geom_paths)):
            visit_tiles = geopandas.read_file(path)
            # In original CRS
            hull = shapely.convex_hull(shapely.union_all(visit_tiles.geometry))
            # In ref tif CRS
            hull_crs = gff.util.convert_crs(hull, visit_tiles.crs, out_tif.crs)
            # In resulting tif pixel space
            hull_px = gff.util.convert_affine_inplace(hull_crs, ~transform)
            hull_w_context = hull_px.buffer(args.context)
            pxlo, pylo, pxhi, pyhi = bounds = gff.util.rounded_bounds(hull_w_context)
            size_pix = (pyhi - pylo, pxhi - pxlo)
            box = shapely.box(*bounds)
            # Back into ref tif CRS
            box_crs = gff.util.convert_affine_inplace(box, transform)
            fine_dem_xr = get_available_cop_dem(box_crs, out_tif.crs, args.data_path)
            coarse_dem_xr = gff.util.resample_xr(
                fine_dem_xr, box_crs.bounds, size_pix, method="linear"
            )
            dem_raster = coarse_dem_xr.band_data.values
            window = gff.util.shapely_bounds_to_rasterio_window(bounds)

            dem_raster[np.isnan(dem_raster)] = INT16MIN
            dem_raster = dem_raster.astype(np.int16)
            out_tif.write(dem_raster, window=window)
            coarse_dem_xr.close()

    ref_tif.close()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
