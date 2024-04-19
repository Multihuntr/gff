import argparse
import datetime
import json
from pathlib import Path
import sqlite3
import sys

import asf_search as asf
import geopandas

import gff.generate.search


def parse_args(argv):
    desc = "Search ASF sentinel-1 archives; build map basin->s1 images"
    parser = argparse.ArgumentParser(desc)

    parser.add_argument("gpkg_path", type=Path, help="File with (floods X basins) shps")
    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    shps = geopandas.read_file(args.gpkg_path, engine="pyogrio", where="BEGAN >= '2014-01-01'")
    index = sqlite3.connect(args.data_path / "s1" / "index.db")

    results = []
    for i, shp in shps.iterrows():
        results.append(gff.generate.search.get_search_results(index, shp))

    filtered_index = {}
    nab, ndb = 0, 0
    nad, ndd = 0, 0
    nac, ndc = 0, 0
    nat, ndt = 0, 0
    basin_prop = None  # 0.05
    n_required = 3
    for i, shp in shps.iterrows():
        _, (ab, db, ad, dd, ac, dc, at, dt) = gff.generate.search.filter_search_results(
            results, index, shp, n_required=n_required, basin_prop=basin_prop
        )
        nab += ab
        ndb += db
        nad += ad
        ndd += dd
        nac += ac
        ndc += dc
        nat += at
        ndt += dt

    print(
        f"Number of times at least {n_required} images that touch the (basin x flood) shapes are available"
    )
    print(f"Search results:           Ascending {nab:5d}  |  Descending {ndb:5d}")
    print(f"Grouped by day:           Ascending {nad:5d}  |  Descending {ndd:5d}")
    print(f"Filtered by {basin_prop:.0%} coverage: Ascending {nac:5d}  |  Descending {ndc:5d}")
    print(f"Timing not usable :       Ascending {nat:5d}  |  Descending {ndt:5d}")

    n_sites = len(
        set([k[5:] for k, v in filtered_index.items() if len(v["asc"]) > 0 or len(v["desc"]) > 0])
    )
    n_floods = len(
        set([k[:4] for k, v in filtered_index.items() if len(v["asc"]) > 0 or len(v["desc"]) > 0])
    )
    print(f"Number of unique sites:  {n_sites:5d}")
    print(f"Number of unique floods: {n_floods:5d}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
