import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import rasterio
import tqdm

import gff.generate.floodmaps
import gff.data_sources
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("floodmap_folder", type=Path)

    return parser.parse_args(argv)


def main(args):
    fpaths = list(args.floodmap_folder.glob("*-meta.json"))
    for fpath in tqdm.tqdm(fpaths, "files"):
        with open(fpath) as f:
            meta = json.load(f)

        floodmap_fpath = fpath.parent / meta["floodmap"]
        tiles_fpath = fpath.parent / meta["visit_tiles"]

        visit_tiles = geopandas.read_file(tiles_fpath, engine="pyogrio", use_arrow=True)
        geoms = list(visit_tiles.geometry)
        flooded_tiles = 0
        all_stats = []
        with rasterio.open(floodmap_fpath) as tif:
            for geom in tqdm.tqdm(geoms, "tiles", leave=False):
                tile = gff.util.get_tile(tif, geom.bounds, align=True)
                stats = gff.data_sources.ks_water_stats(tile)
                flooded_tiles += 1 if gff.generate.floodmaps.tile_flooded(stats) else 0
                all_stats.append(stats)

        gff.util.save_tiles(geoms, all_stats, tiles_fpath, visit_tiles.crs)
        meta["flooded_tiles"] = flooded_tiles
        with open(fpath, "w") as f:
            json.dump(meta, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
