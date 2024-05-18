import argparse
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

    return parser.parse_args(argv)


def export_resample_by_fnc(
    data_path, out_fpath, fnc, spatial_profile, visit_tiles, max_error=0.5, overwrite=False
):
    # Export a upsampled to match floodmap
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
                method="linear",
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
        if not s1_out_fpath.exists():
            tiles_to_remove = []
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
                        tiles_to_remove.append(i)
                        continue
                    window = gff.util.shapely_bounds_to_rasterio_window(
                        tile.bounds, s1_tif.transform
                    )
                    s1_tif.write(s1_data, window=window)
            if len(tiles_to_remove) > 0:
                visit_tiles = visit_tiles.drop(tiles_to_remove)
                visit_tiles.to_file(v_path)

        dem_fpath = floodmap_path.with_name(floodmap_path.stem + "-dem-local.tif")
        fnc = gff.data_sources.get_dem
        export_resample_by_fnc(
            args.data_path, dem_fpath, fnc, spatial_profile, visit_tiles, max_error=0.5
        )

        hand_fpath = floodmap_path.with_name(floodmap_path.stem + "-hand.tif")
        fnc = gff.data_sources.get_hand
        export_resample_by_fnc(
            args.data_path, hand_fpath, fnc, spatial_profile, visit_tiles, max_error=0.5
        )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
