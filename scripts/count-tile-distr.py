import argparse
import json
from pathlib import Path
import sys

import geopandas


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    paths = list((args.data_path / "rois").glob("*-meta.json"))
    n_visit, n_flood = 0, 0
    for path in paths:
        with open(path) as f:
            meta = json.load(f)
        visit_tiles = geopandas.read_file(
            path.parent / meta["visit_tiles"], engine="pyogrio", use_arrow=True
        )
        flood_tiles = geopandas.read_file(
            path.parent / meta["flood_tiles"], engine="pyogrio", use_arrow=True
        )

        n_visit += len(visit_tiles)
        n_flood += len(flood_tiles)
    ratio = n_flood / n_visit
    print(n_visit, n_flood, f"{ratio:5.3f}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
