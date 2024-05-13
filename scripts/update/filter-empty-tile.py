import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import rasterio
import tqdm

import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("floodmaps_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    for meta_path in tqdm.tqdm(list(args.floodmaps_path.glob("*-meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        v_p = meta_path.parent / meta["visit_tiles"]
        f_p = meta_path.parent / meta["floodmap"]
        geoms = geopandas.read_file(v_p, engine="pyogrio", use_arrow=True)
        geom_mask = []
        with rasterio.open(f_p) as tif:
            for i, geom_row in geoms.iterrows():
                data = gff.util.get_tile(tif, geom_row.geometry.bounds).astype(np.int64)
                keep = not (data.min().item() == 255 or data.max().item() == 255)
                geom_mask.append(keep)
        geoms_masked = geoms[geom_mask]
        if len(geoms) == 0:
            print(meta_path, "had 0 tiles. Skipping...")
            continue
        geoms_masked.to_file(v_p)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
