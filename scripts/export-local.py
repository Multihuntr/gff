import argparse
import datetime
import json
import sys
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import tqdm

import gff.constants
import gff.data_sources
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("--export", "-e", action="append", default=[])

    return parser.parse_args(argv)


def export_resample_by_fnc(
    data_path,
    out_fpath,
    fnc,
    spatial_profile,
    visit_tiles,
    max_error=0.5,
    overwrite=False,
    method="linear",
):
    # Export an upsampled version to match floodmap
    profile = {
        **gff.constants.S1_PROFILE_DEFAULTS,
        "MAX_Z_ERROR": max_error,
        **spatial_profile,
        "count": 1,
    }
    if overwrite and out_fpath.exists():
        return
    with rasterio.open(out_fpath, "w", **profile) as tif:
        for tile in tqdm.tqdm(np.array(visit_tiles.geometry), "Tiles", leave=False):
            data = fnc(tile, visit_tiles.crs, data_path)
            tile_4326 = gff.util.convert_crs(tile, visit_tiles.crs, "EPSG:4326")
            resampled = gff.util.resample_xr(
                data,
                tile_4326.bounds,
                (gff.constants.LOCAL_RESOLUTION,) * 2,
                method=method,
            ).band_data.values
            window = gff.util.shapely_bounds_to_rasterio_window(tile.bounds, tif.transform)
            tif.write(resampled, window=window)


def main(args):
    meta_paths = list((args.data_path / "rois").glob("*-meta.json"))
    for meta_path in tqdm.tqdm(meta_paths, "Files"):
        with open(meta_path) as f:
            meta = json.load(f)
        v_path = meta_path.parent / meta["visit_tiles"]
        visit_tiles = geopandas.read_file(v_path, engine="pyogrio", use_arrow=True)

        floodmap_path: Path = meta_path.parent / meta["floodmap"]
        floodmap_tif = rasterio.open(floodmap_path)
        spatial_profile = {
            "transform": floodmap_tif.transform,
            "crs": floodmap_tif.crs,
            "width": floodmap_tif.width,
            "height": floodmap_tif.height,
        }

        # Export S1 files
        if "s1" in args.export:
            s1_stem = gff.util.get_s1_stem_from_meta(meta)
            s1_profile = {
                **gff.constants.S1_PROFILE_DEFAULTS,
                **spatial_profile,
            }
            s1_in_fpath = args.data_path / "s1" / f"{s1_stem}.tif"
            if not s1_in_fpath.exists():
                if meta["type"] == "kurosiwo":
                    s1_in_fpath = args.data_path / "s1" / "kurosiwo-merge" / f"{s1_stem}.tif"
                if not s1_in_fpath.exists():
                    raise Exception("S1 couldn't be found")
            s1_in_tif = rasterio.open(s1_in_fpath)
            s1_out_fpath = meta_path.parent / f"{s1_stem}-s1.tif"
            with rasterio.open(s1_out_fpath, "w", **s1_profile) as s1_tif:
                s1_tif.update_tags(**s1_in_tif.tags())
                s1_tif.update_tags(1, **s1_in_tif.tags(1))
                s1_tif.update_tags(2, **s1_in_tif.tags(2))
                for i, tile_row in tqdm.tqdm(
                    visit_tiles.iterrows(), "S1 Tiles", leave=False, total=len(visit_tiles)
                ):
                    tile = tile_row.geometry
                    tile_in = gff.util.convert_crs(tile, visit_tiles.crs, s1_in_tif.crs)
                    s1_data = gff.util.get_tile(s1_in_tif, tile_in.bounds)
                    if np.isnan(s1_data).sum() != 0:
                        raise Exception("Should be no nan s1.")
                    window = gff.util.shapely_bounds_to_rasterio_window(
                        tile.bounds, s1_tif.transform
                    )
                    s1_tif.write(s1_data, window=window)

        if "dem" in args.export:
            dem_fpath = floodmap_path.with_name(floodmap_path.stem + "-dem-local.tif")
            fnc = gff.data_sources.get_dem
            export_resample_by_fnc(
                args.data_path, dem_fpath, fnc, spatial_profile, visit_tiles, max_error=0.5
            )

        if "hand" in args.export:
            hand_fpath = floodmap_path.with_name(floodmap_path.stem + "-hand.tif")
            fnc = gff.data_sources.get_hand
            export_resample_by_fnc(
                args.data_path, hand_fpath, fnc, spatial_profile, visit_tiles, max_error=0.5
            )

        classification_spatial_profile = csp = {
            **spatial_profile,
            "COMPRESS": "LERC",
            "MAX_Z_ERROR": 0,
            "INTERLEAVE": "BAND",
            "dtype": np.uint8,
            "nodata": 0,
        }
        if "worldcover" in args.export:
            wc_fpath = floodmap_path.with_name(floodmap_path.stem + "-worldcover.tif")
            fnc = gff.data_sources.get_world_cover
            export_resample_by_fnc(
                args.data_path, wc_fpath, fnc, csp, visit_tiles, method="nearest"
            )

        if "gswe" in args.export:
            gswe_fname = Path(meta["floodmap"]).with_name(floodmap_path.stem + "-gswe.tif")
            gswe_fpath = args.data_path / "gswe-rois" / gswe_fname
            flood_date = datetime.datetime.fromisoformat(meta["post_date"])
            gswe_profile = {**csp, "nodata": 255}

            def gswe_fnc(shp, crs, folder):
                return gff.data_sources.get_global_surface_water(shp, crs, folder, date=flood_date)

            export_resample_by_fnc(
                args.data_path, gswe_fpath, gswe_fnc, csp, visit_tiles, method="nearest"
            )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
