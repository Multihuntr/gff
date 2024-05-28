import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


def parse_args(argv):
    parser = argparse.ArgumentParser("Reads results from folders")

    parser.add_argument("paperruns_path", type=Path)

    return parser.parse_args(argv)


HEADERS = ["F1", "F1 BG", "F1 Water", "F1_{KS}", "F1_{KS} BG", "F1_{KS} Water"]


def extract_f1s(fpath):
    with open(fpath / "eval_results.yml") as f:
        results = yaml.safe_load(f)
    f1_bg, f1_water = results["f1"]["overall"]
    f1_mean = (f1_bg + f1_water) / 2

    with open(fpath / "eval_results_ks.yml") as f:
        results_ks = yaml.safe_load(f)
    f1_bg_ks, f1_water_ks = results_ks["f1"]["overall"]
    f1_mean_ks = (f1_bg_ks + f1_water_ks) / 2

    return f1_mean, f1_bg, f1_water, f1_mean_ks, f1_bg_ks, f1_water_ks


def main(args):
    names = ["utae", "recunet", "metnet", "3dunet"]
    n_partitions = 5
    vals = {}
    for name in names:
        vals[name] = []
        for i in range(n_partitions):
            fpath = args.paperruns_path / f"{name}_{i}"
            vals[name].append(extract_f1s(fpath))
    metrics: np.ndarray = np.zeros((len(names), 2, len(HEADERS)))
    for i, name in enumerate(names):
        a = np.array(vals[name])
        # shaped [n_partitions, len(HEADERS)]
        mean = a.mean(axis=(0))
        std = a.std(axis=(0))
        metrics[i] = (mean, std)
    metrics = np.transpose(metrics, axes=(0, 2, 1))

    # Magic string formatting :D
    preheader = r"& \multicolumn{3}{c}{Everything} & \multicolumn{3}{c}{Kuro Siwo Labels} \\"
    # For each column header, create a textbf statement that is a specific width (incl. latex bf)
    header = f"{'model':>8s} & " + " & ".join([f"{f'{col}':^18s}" for col in HEADERS])
    # (This is not necessary for latex, it just makes the source easier to read)
    vals_fmt = [f"   {{{i}:5.2f}} ({{{i+1}:4.2f}})   " for i in range(1, len(HEADERS) * 2 + 1, 2)]
    row_fmt_str = r"{0:>8s} & " + " & ".join(vals_fmt)

    # Print table
    print(preheader)
    print(header, end="\\\\\n")
    print("\\midrule")
    for i, name in enumerate(names):
        print(row_fmt_str.format(name, *(metrics[i].flatten())), end="\\\\\n")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
