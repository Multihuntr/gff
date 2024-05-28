import argparse
import datetime
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    diffs = []
    paths = list((args.data_path / "rois").glob("*-meta.json"))
    for meta_path in paths:
        with open(meta_path) as f:
            meta = json.load(f)

        pre1_date = datetime.datetime.fromisoformat(meta[f"pre1_date"])
        post_date = datetime.datetime.fromisoformat(meta[f"post_date"])
        diffs.append((post_date - pre1_date).days)

    diffs_np = np.array(diffs)
    diffs_np[diffs_np > 59] = 59

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=300)
    ax.hist(diffs_np, bins=60, range=(0, 60))
    ax.set_xlabel("Days")

    fig.tight_layout()
    fig.savefig("vis/lead-time.png")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
