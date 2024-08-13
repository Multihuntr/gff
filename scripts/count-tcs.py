import argparse
from pathlib import Path
import sys

import geopandas
import pandas

import gff.data_sources
import gff.generate.basins


def parse_args(argv):
    parser = argparse.ArgumentParser("Count stats using Tropical Cyclones")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("dfo_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("tcs_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)


def main(args):
    dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    tc_csv_path = args.tcs_path / "titleyetal2021_280storms.csv"
    tcs = pandas.read_csv(tc_csv_path, converters={"BASIN": str})

    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    basins08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    basins08_df = geopandas.read_file(
        basin_path / basins08_fname, use_arrow=True, engine="pyogrio"
    )
    gff.generate.basins.tcs_basins(dfo, basins08_df, tcs, args.tcs_path)

    print("Tropical cyclones directly used")
    tcs = pandas.read_csv(args.data_path / "tc_basins.csv")
    for i, tc_row in tcs.iterrows():
        check_str = f"{tc_row.FLOOD_ID}-{tc_row.HYBAS_ID}-*-meta.json"
        n = len(list((args.data_path / "release" / "rois").glob(check_str)))
        if n > 0:
            print(check_str)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
