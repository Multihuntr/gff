import argparse
import datetime
import itertools
import json
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
import gff.dataloaders
import gff.data_sources
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Cycles through all floodmaps and creates a dem file at the resolution matching a provided reference tif"
    )

    parser.add_argument("data_path", type=Path)
    parser.add_argument(
        "context", type=int, help="Number of pixels in era5_land as buffer around each floodmap"
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
    ref_era5l_path = list((args.data_path / gff.constants.ERA5L_FOLDER).glob("*.tif"))[0]
    ref_era5l_tif = rasterio.open(ref_era5l_path)
    hydroatlas_tif = rasterio.open(args.data_path / gff.constants.HYDROATLAS_RASTER_FNAME)

    assert hydroatlas_tif.crs == ref_era5l_tif.crs
    ref_crs = hydroatlas_tif.crs

    # The DEM and Hydroatlas are stored at a finer resolution because they
    # themselves are downsampled products. This reduces resampling errors downstream.
    downsampled_res = hydroatlas_tif.transform[0], -hydroatlas_tif.transform[4]
    era5l_res = ref_era5l_tif.transform[0], -ref_era5l_tif.transform[4]
    dem_profile = {
        **hydroatlas_tif.profile,
        "count": 1,
        "nodata": INT16MIN,
        "dtype": np.int16,
        "COMPRESS": "LERC",
        "MAX_Z_ERROR": 0,
        "INTERLEAVE": "BAND",
    }

    # Get valid date range as the minimum across both
    max_days = 365
    era5_path = args.data_path / gff.constants.ERA5_FOLDER
    era5L_path = args.data_path / gff.constants.ERA5L_FOLDER
    valid_date_range = gff.util.get_valid_date_range(era5_path, era5L_path, [], max_days)

    floodmap_path = args.data_path / "rois"
    meta_paths = list(floodmap_path.glob("*-meta.json"))
    for i, path in enumerate(tqdm.tqdm(meta_paths)):
        with open(path) as f:
            meta = json.load(f)

        visit_tiles = geopandas.read_file(path.parent / meta["visit_tiles"])
        # In original CRS
        hull = shapely.convex_hull(shapely.union_all(visit_tiles.geometry))
        # In ref tif CRS
        hull_crs = gff.util.convert_crs(hull, visit_tiles.crs, ref_crs)
        # In era5_land tif pixel space
        hull_px = gff.util.convert_affine(hull_crs, ~ref_era5l_tif.transform)
        hull_w_context = hull_px.buffer(args.context)
        exlo, eylo, exhi, eyhi = bounds = gff.util.rounded_bounds(hull_w_context)
        box = shapely.box(*bounds)
        # Back into ref tif CRS
        context_box = gff.util.convert_affine(box, ref_era5l_tif.transform)
        cxlo, cylo, cxhi, cyhi = context_box.bounds
        # And then back into downsampled pixel space
        context_box_ds_px = gff.util.convert_affine(context_box, ~hydroatlas_tif.transform)
        dxlo, dylo, dxhi, dyhi = gff.util.rounded_bounds(context_box_ds_px)

        k = Path(meta["floodmap"]).stem
        dem_out_fpath = floodmap_path / f"{k}-dem-context.tif"
        hydroatlas_out_fpath = floodmap_path / f"{k}-hydroatlas.tif"
        era5_out_fpath = floodmap_path / f"{k}-era5.tif"
        era5L_out_fpath: Path = floodmap_path / f"{k}-era5-land.tif"
        era5L_out_fpath.parent.mkdir(exist_ok=True)

        ds_spatial_profile = {
            "width": dxhi - dxlo,
            "height": dyhi - dylo,
            "transform": rasterio.transform.from_origin(cxlo, cyhi, *downsampled_res),
        }
        era5_spatial_profile = {
            "width": exhi - exlo,
            "height": eyhi - eylo,
            "transform": rasterio.transform.from_origin(cxlo, cyhi, *era5l_res),
        }
        # Output dem file
        d_size_px = (dyhi - dylo, dxhi - dxlo)
        this_dem_profile = {**dem_profile, **ds_spatial_profile}
        with rasterio.open(dem_out_fpath, "w", **this_dem_profile) as out_tif:
            fine_dem_xr = get_available_cop_dem(context_box, ref_crs, args.data_path)
            coarse_dem_xr = gff.util.resample_xr(
                fine_dem_xr, context_box.bounds, d_size_px, method="linear"
            )
            dem_raster = coarse_dem_xr.band_data.values

            dem_raster[np.isnan(dem_raster)] = INT16MIN
            dem_raster = dem_raster.astype(np.int16)
            out_tif.write(dem_raster)
            coarse_dem_xr.close()

        # Output HydroATLAS
        this_hydroatlas_profile = {**hydroatlas_tif.profile, **ds_spatial_profile}
        with rasterio.open(hydroatlas_out_fpath, "w", **this_hydroatlas_profile) as out_tif:
            out_tif.descriptions = hydroatlas_tif.descriptions
            data = gff.util.get_tile(hydroatlas_tif, context_box.bounds)
            out_tif.write(data)

        # Output ERA5

        # Don't output if there's no valid dates
        if not gff.util.meta_in_date_range(meta, valid_date_range, 1):
            continue

        era5L_size_pix = (eyhi - eylo, exhi - exlo)
        end_date = datetime.datetime.fromisoformat(meta["post_date"])
        start_date = end_date - datetime.timedelta(days=max_days)
        # Note, resampling done on load
        era5_keys, era5_data = gff.data_sources.load_era5(
            era5_path,
            context_box,
            era5L_size_pix,
            start_date,
            end_date,
            era5_land=False,
        )
        # Between the beginning of the project and the end, I changed my mind about how
        # tifs should be stored. To begin with, it was ints with explicit scale/offsets.
        # But then I found out about LERC, which automatically does that, but better.
        new_era5_profile = {
            "dtype": np.float32,
            "nodata": np.nan,
            "COMPRESS": "LERC",
            "MAX_Z_ERROR": 0.00001,
            "INTERLEAVE": "BAND",
        }
        era5_profile = {
            **ref_era5l_tif.profile,
            **new_era5_profile,
            **era5_spatial_profile,
            "count": len(era5_keys),
        }
        with rasterio.open(era5_out_fpath, "w", **era5_profile) as out_tif:
            out_tif.descriptions = era5_keys
            out_tif.scales = (1,) * len(era5_keys)
            out_tif.offsets = (0,) * len(era5_keys)
            out_tif.write(np.concatenate(era5_data, axis=0))

        # Output ERA5 Land
        era5L_keys, era5L_data = gff.data_sources.load_era5(
            era5L_path,
            context_box,
            era5L_size_pix,
            start_date,
            end_date,
            era5_land=True,
        )
        era5L_profile = {
            **ref_era5l_tif.profile,
            **new_era5_profile,
            **era5_spatial_profile,
            "count": len(era5L_keys),
        }
        with rasterio.open(era5L_out_fpath, "w", **era5L_profile) as out_tif:
            out_tif.descriptions = era5L_keys
            out_tif.scales = (1,) * len(era5L_keys)
            out_tif.offsets = (0,) * len(era5L_keys)
            # TODO: Volumetric total precipitation, snowdepth water equivalent, potential evaporation sum are all 0
            # TODO: Surface Net solar and Thermal, surface pressure span the whole int16 range
            out_tif.write(np.concatenate(era5L_data, axis=0))

    ref_era5l_tif.close()
    hydroatlas_tif.close()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
