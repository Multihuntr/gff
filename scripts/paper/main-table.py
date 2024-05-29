import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


def parse_args(argv):
    parser = argparse.ArgumentParser("Reads results from folders")

    parser.add_argument("paperruns_path", type=Path)
    parser.add_argument("--table", type=str, default="main")

    return parser.parse_args(argv)


HEADERS = {
    "main": ["F1", "F1-B", "F1-W", "F1_{KS}", "F1-B_{KS}", "F1-W_{KS}"],
    "coastal": ["F1_c", "F1-B_c", "F1-W_c", "F1_i", "F1-B_i", "F1-W_i"],
}


def extract_f1s(fpath, table="main"):
    if table == "main":
        with open(fpath / "eval_results.yml") as f:
            results = yaml.safe_load(f)
        f1_bg, f1_water = results["f1"]["overall"]
        f1_mean = (f1_bg + f1_water) / 2

        with open(fpath / "eval_results_ks.yml") as f:
            results_ks = yaml.safe_load(f)
        f1_bg_ks, f1_water_ks = results_ks["f1"]["overall"]
        f1_mean_ks = (f1_bg_ks + f1_water_ks) / 2

        return f1_mean, f1_bg, f1_water, f1_mean_ks, f1_bg_ks, f1_water_ks
    elif table == "coastal":
        with open(fpath / "eval_results.yml") as f:
            results = yaml.safe_load(f)
        f1c_bg, f1c_water = results["f1"]["coast"]
        f1c_mean = (f1c_bg + f1c_water) / 2

        f1i_bg, f1i_water = results["f1"]["inland"]
        f1i_mean = (f1i_bg + f1i_water) / 2

        return f1c_mean, f1c_bg, f1c_water, f1i_mean, f1i_bg, f1i_water


def main(args):
    names = ["utae", "recunet", "metnet", "3dunet"]
    n_partitions = 5
    vals = {}
    for name in names:
        vals[name] = []
        for i in range(n_partitions):
            fpath = args.paperruns_path / f"{name}_{i}"
            vals[name].append(extract_f1s(fpath, args.table))
    headers = HEADERS[args.table]
    metrics: np.ndarray = np.zeros((len(names), 2, len(headers)))
    for i, name in enumerate(names):
        a = np.array(vals[name])
        # shaped [n_partitions, len(headers)]
        mean = a.mean(axis=(0))
        std = a.std(axis=(0))
        metrics[i] = (mean, std)
    metrics = np.transpose(metrics, axes=(0, 2, 1))

    header = f"{'model':>8s} & " + " & ".join([f"{f'{col}':^14s}" for col in headers])
    vals_fmt = [f" {{{i}:5.2f}} ({{{i+1}:4.2f}}) " for i in range(1, len(headers) * 2 + 1, 2)]
    row_fmt_str = r"{0:>8s} & " + " & ".join(vals_fmt)

    # Print table
    print(header, end="\\\\\n")
    print("\\midrule")
    for i, name in enumerate(names):
        print(row_fmt_str.format(name, *(metrics[i].flatten())), end="\\\\\n")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
