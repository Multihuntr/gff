import argparse
import csv
from pathlib import Path
import random
import sys
import pandas as pd

import torch


# According to: https://github.com/kratzert/Caravan/issues/16
# There are basins which exist, but are "too small". They were supposed to be removed.
# But they were not.
# The following code could remove them from the fpaths, if there were in there.
# But, since we select by filenames, not the attributes, the mismatch doesn't affect
# this script.
"""
fpaths = _only_include_from_caravan_stats(args.data_path, folder, fpaths)
unmatched_i = _check_basins_exist(args.data_path, folder, fpaths)
unmatched.extend(unmatched_i)

if len(unmatched) > 0:
    print("There is an issue with the dataset. The following keys were in hydroatlas ")
    print("attributes but were not in the caravan attributes")
    print(unmatched)

def _only_include_from_caravan_stats(p, folder, fpaths):
    caravan_stats = pd.read_csv(p / "attributes" / folder / f"attributes_caravan_{folder}.csv")
    return [fpath for fpath in fpaths if fpath in caravan_stats["gauge_id"].values]

def _check_basins_exist(p, folder, fpaths):
    caravan_stats = pd.read_csv(p / "attributes" / folder / f"attributes_hydroatlas_{folder}.csv")
    exist = caravan_stats[~caravan_stats["gauge_id"].isin(fpaths)]
    return exist["gauge_id"]
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Scan Caravan netcdf directories and split into train/val/test"
    )

    parser.add_argument("data_path", type=Path, help="Root of extracted Caravan")
    parser.add_argument("--out_path", type=Path, default=Path("./split"), help="Output folder")
    parser.add_argument("--train_perc", type=int, default=90, help="Percent allocated to Train")
    parser.add_argument("--val_perc", type=int, default=5, help="Percent allocated to Validation")
    parser.add_argument("--test_perc", type=int, default=5, help="Percent allocated to Test")
    parser.add_argument("--seed", type=int, default=684)

    return parser.parse_args(argv)


def main(args):
    # Setup
    cdf_folders = [
        "camels",
        "camelsaus",
        "camelsbr",
        "camelscl",
        "camelsgb",
        "hysets",
        "lamah",
        "grdc",
    ]
    props = [args.train_perc / 100, args.val_perc / 100, args.test_perc / 100]
    ids = {"train": [], "val": [], "test": []}
    print(f"{'':13s} {'train':^6s} {'val':^6s} {'test':^6s}")
    unmatched = []

    # Sample such that it is evenly distributed across subdatasets
    gen = torch.Generator().manual_seed(args.seed)
    for folder in cdf_folders:
        fpaths = [p.stem for p in args.data_path.glob(f"**/{folder}/*.nc")]

        # According to: https://github.com/kratzert/Caravan/issues/26#issuecomment-1807990313
        # We should ignore these basins:
        if folder == "grdc":
            for basin_id in ["GRDC_5606274", "GRDC_5606174", "GRDC_5202086", "GRDC_5202088"]:
                fpaths.remove(basin_id)

        # Randomly split this dataset
        train_ids_i, val_ids_i, test_ids_i = torch.utils.data.random_split(fpaths, props, gen)
        ids["train"].extend(train_ids_i)
        ids["val"].extend(val_ids_i)
        ids["test"].extend(test_ids_i)
        print(f"{folder:13s} {len(train_ids_i):6d} {len(val_ids_i):6d} {len(test_ids_i):6d}")

    print("-" * (13 + 18 + 3))
    print(f"{'total':13s} {len(ids['train']):6d} {len(ids['val']):6d} {len(ids['test']):6d}")

    # Write out all the splits to txt files; one id per line.
    for k in ["train", "val", "test"]:
        with (args.out_path / f"{k}.txt").open("w") as f:
            f.write("\n".join(ids[k]))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
