import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


def parse_args(argv):
    parser = argparse.ArgumentParser("Aggregates results from run folders")

    parser.add_argument("run_fpaths", nargs="+", type=Path)

    return parser.parse_args(argv)


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
    elif table == "ablation":
        with open(fpath / "eval_results.yml") as f:
            results = yaml.safe_load(f)
        f1_bg, f1_water = results["f1"]["overall"]
        f1_mean = (f1_bg + f1_water) / 2
        return f1_mean, f1_bg, f1_water


def main(args):
    metrics = []
    for fpath in args.run_fpaths:
        metrics.append(extract_f1s(fpath))
    metrics = np.array(metrics).T
    metric_names = ["F1", "F1-BG", "F1-W", "F1-KS", "F1-KS-BG", "F1-KS-W"]
    for name, metric in zip(metric_names, metrics):
        print(f"{name:10s}  :  {metric.mean():5.3f} +- {metric.std():5.3f}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
