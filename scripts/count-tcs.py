import argparse
from pathlib import Path
import sys

import pandas


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    tcs = pandas.read_csv(args.data_path / "tc_basins.csv")
    for i, tc_row in tcs.iterrows():
        check_str = f"{tc_row.FLOOD_ID}-{tc_row.HYBAS_ID}-*-meta.json"
        n = len(list((args.data_path / "rois").glob(check_str)))
        if n > 0:
            print(check_str)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
