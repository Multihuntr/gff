import argparse
import re
import sys
from pathlib import Path

import pandas
import torch
import yaml

import gff.dataloaders
import gff.evaluation
import gff.models.creation
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Train a model")

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
    # Load config file, and overwrite anything from cmdline arguments
    with open(args.model_folder / "config.yml") as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    # Load model weights
    model = gff.models.creation.create(C)
    model.to(C["device"])
    checkpoint_files = list(args.model_folder.glob("checkpoint_*.th"))
    checkpoint_files.sort()
    checkpoint = torch.load(checkpoint_files[-1])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Determine fnames
    data_path = Path(C["data_folder"]).expanduser()
    fold_names_fpath = data_path / "partitions" / f"floodmap_partition_{C['fold']}.txt"
    fnames = pandas.read_csv(fold_names_fpath, header=None)[0].values.tolist()
    ks_fnames = [fname for fname in fnames if fname_is_ks(fname)]

    # Test dataloader
    test_dl = gff.dataloaders.create_test(C)

    # Evaluate the model
    out_path = gff.evaluation.model_inference(args.model_folder, model, test_dl)
    targ_path = data_path / "rois"
    n_cls = C["n_classes"]
    eval_results = gff.evaluation.evaluate_floodmaps(fnames, out_path, targ_path, n_cls)
    with open(args.model_folder / "eval_results.yml", "w") as f:
        yaml.safe_dump(eval_results, f)
    ks_eval_results = gff.evaluation.evaluate_floodmaps(ks_fnames, out_path, targ_path, n_cls)
    with open(args.model_folder / "eval_results_ks.yml", "w") as f:
        yaml.safe_dump(ks_eval_results, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
