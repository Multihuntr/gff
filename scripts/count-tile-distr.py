import argparse
import json
from pathlib import Path
import sys

import geopandas


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("--only_kurosiwo", action="store_true")

    return parser.parse_args(argv)


THRESHOLD = 0.05


def main(args):
    n_tiles, n_anyw_t, n_fl_t = 0, 0, 0
    n_pixels, n_bg_px, n_pw_px, n_fl_px = 0, 0, 0, 0

    paths = list((args.data_path / "rois").glob("*-meta.json"))
    for path in paths:
        with open(path) as f:
            meta = json.load(f)

        if args.only_kurosiwo and meta["type"] != "kurosiwo":
            continue

        tiles = geopandas.read_file(
            path.parent / meta["visit_tiles"], engine="pyogrio", use_arrow=True
        )
        n_tiles += len(tiles)
        total_pixels = tiles.n_background + tiles.n_permanent_water + tiles.n_flooded
        flood_proportion = tiles.n_flooded / total_pixels
        water_proportion = (tiles.n_flooded + tiles.n_permanent_water) / total_pixels
        n_fl_t += (flood_proportion > THRESHOLD).values.sum()
        n_anyw_t += (water_proportion > THRESHOLD).values.sum()
        n_pixels += total_pixels.values.sum()
        n_bg_px += tiles.n_background.values.sum()
        n_pw_px += tiles.n_permanent_water.values.sum()
        n_fl_px += tiles.n_flooded.values.sum()

    print(
        f"   Bg (tiles) & Water (tiles) & Flood (tiles) &  Bg (pixels)  & P.Water (pixels) & Flood (pixels)"
    )
    print(
        f"---------------------------------------------------------------------------------------------------"
    )
    n_bg_t = n_tiles - n_anyw_t
    print(
        f" {n_bg_t:12d} & {n_anyw_t:13d} & {n_fl_t:13d} & {n_bg_px:13d} & {n_pw_px:16d} & {n_fl_px:14d}"
    )
    p_bg_t = n_bg_t / n_tiles
    p_anyw_t = n_anyw_t / n_tiles
    p_fl_t = n_fl_t / n_tiles
    p_bg_px = n_bg_px / n_pixels
    p_pw_px = n_pw_px / n_pixels
    p_fl_px = n_fl_px / n_pixels
    print(
        f" {p_bg_t:12.1%} & {p_anyw_t:13.1%} & {p_fl_t:13.1%} & {p_bg_px:13.1%} & {p_pw_px:16.1%} & {p_fl_px:14.1%}"
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
