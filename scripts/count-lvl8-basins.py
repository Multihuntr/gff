import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import shapely
import tqdm


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v10_shp"
    basins08_fname = f"BasinATLAS_v10_lev08.shp"
    basins_df = geopandas.read_file(basin_path / basins08_fname, use_arrow=True, engine="pyogrio")
    basin_geoms = np.array(basins_df.geometry.values)
    basin_mask = np.array([False,]* len(basin_geoms))

    paths = list((args.data_path / "rois").glob("*-meta.json"))
    for path in tqdm.tqdm(paths):
        with open(path) as f:
            meta = json.load(f)

        tiles = geopandas.read_file(
            path.parent / meta["visit_tiles"], engine="pyogrio", use_arrow=True
        )
        tiles_geom = shapely.unary_union(tiles.geometry)
        overlap = shapely.area(shapely.intersection(basin_geoms, tiles_geom))
        mask = overlap > (0.3 * shapely.area(basin_geoms))
        basin_mask[mask] = True

    print("Number of level 8 basins covered:", basin_mask.sum())


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
