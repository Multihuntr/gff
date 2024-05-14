import argparse
from pathlib import Path
import re
import sys

import pandas
import torch
import yaml

import gff.models.creation
import gff.evaluation


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)
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


def main(args):
    with open(args.model_folder / "config.yml") as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    # Load model
    model = gff.models.creation.create(C)
    model.to(C["device"])
    checkpoint_files = list(args.model_folder.glob("checkpoint_*.th"))
    checkpoint_files.sort()
    checkpoint = torch.load(checkpoint_files[-1])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Pick Kuro Siwo test rois
    fold_names_fpath = args.data_path / "partitions" / f"floodmap_partition_{C['fold']}.txt"
    fnames = pandas.read_csv(fold_names_fpath, header=None)[0].values.tolist()
    fnames = [fname for fname in fnames if fname_is_ks(fname)]

    # Evaluate
    eval_results = gff.evaluation.evaluate_model_from_fnames(C, args.data_path, model, fnames)
    with open(args.model_folder / "eval_results_ks.yml", "w") as f:
        yaml.safe_dump(eval_results, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
