import argparse
import json
from pathlib import Path
import sys
import warnings

import geopandas
import pandas as pd
import shapely
import tqdm


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Checks which rois are covered by the LISFLOOD condition used for GloFAS"
    )

    parser.add_argument("rois_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    rivers = []
    for raw_path in args.hydroatlas_path.glob("**/RiverATLAS_v10_*.shp"):
        with warnings.catch_warnings(action="ignore"):
            r = geopandas.read_file(
                raw_path, engine="pyogrio", use_arrow=True, where=f"UPLAND_SKM > 1000"
            )
        rivers.append(r)
    rivers_df = geopandas.GeoDataFrame(pd.concat(rivers, ignore_index=True), crs=rivers[0].crs)

    fpaths = list(args.rois_path.glob("*-meta.json"))
    non_touching_fpaths = []
    for fpath in tqdm.tqdm(fpaths, "files"):
        with open(fpath) as f:
            meta = json.load(f)

        tiles_fpath = fpath.parent / meta["visit_tiles"]
        tiles = geopandas.read_file(tiles_fpath, engine="pyogrio", use_arrow=True)
        tiles = tiles.to_crs("EPSG:4326")
        site_shp = shapely.unary_union(tiles.geometry)

        passes_through = rivers_df.geometry.intersects(site_shp)
        if passes_through.sum() == 0:
            non_touching_fpaths.append(fpath)
    print(non_touching_fpaths)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
