import argparse
import re
import sys
from pathlib import Path

from matplotlib import pyplot as plt
import pandas
import yaml

import gff.constants
import gff.dataloaders
import gff.evaluation
import gff.models.creation
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Evaluate floodmaps stored in the same format as the labels")

    parser.add_argument("model_folder", type=Path)
    parser.add_argument(
        "--overwrite",
        "-o",
        type=gff.util.pair,
        nargs="*",
        default=[],
        help="Overwrite config setting",
    )

    return parser.parse_args(argv)


def fname_is_ks(fname: str):
    return re.match(r"\d{3}-\d{1,2}-", fname) is not None


def evaluate_and_save(model_folder, fnames, out_path, targ_path, n_cls, suffix=""):
    eval_results, test_cm = gff.evaluation.evaluate_floodmaps(fnames, out_path, targ_path, n_cls)
    with open(model_folder / f"eval_results{suffix}.yml", "w") as f:
        yaml.safe_dump(eval_results, f)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    test_cm.plot(ax=ax, labels=gff.constants.KUROSIWO_CLASS_NAMES[:n_cls])
    ax.set_title(f"Test{suffix}")
    fig.tight_layout()
    fig.savefig(model_folder / f"test_cm{suffix}.png")
    fig.savefig(model_folder / f"test_cm{suffix}.eps")
    plt.close(fig)


def main(args):
    # Load config file, and overwrite anything from cmdline arguments
    with open(args.model_folder / "config.yml") as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    # Determine fnames
    data_path = Path(C["data_folder"]).expanduser()
    fold_names_fpath = data_path / "partitions" / f"floodmap_partition_{C['fold']}.txt"
    fnames = pandas.read_csv(fold_names_fpath, header=None)[0].values.tolist()
    ks_fnames = [fname for fname in fnames if fname_is_ks(fname)]

    # Evaluate the model
    targ_path = data_path / "rois"
    out_path = args.model_folder / "inference"
    n_cls = C["n_classes"]
    evaluate_and_save(args.model_folder, fnames, out_path, targ_path, n_cls)
    evaluate_and_save(args.model_folder, ks_fnames, out_path, targ_path, n_cls, "_ks")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
