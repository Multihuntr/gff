raise NotImplementedError("This code is out of date. Use at your own risk.")

import argparse
import json
import sys
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import shapely

import gff.util as util


def parse_args(argv):
    parser = argparse.ArgumentParser("Deletes tiles outside the majority basin boundary")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--lvl", type=int, default=4)

    return parser.parse_args(argv)


def main(args):
    basin_dir = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_path = basin_dir / f"BasinATLAS_v10_lev{args.lvl:02d}.shp"
    basins_df = geopandas.read_file(basin_path, engine="pyogrio")

    for meta_path in (args.data_path / "floodmaps").glob("*-meta.json"):
        util.remove_tiles_outside(meta_path, basins_df)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
