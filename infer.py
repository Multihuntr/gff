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
    parser = argparse.ArgumentParser("Run a trained model over the test set.")

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

    # Run the model over the test set.
    test_dl = gff.dataloaders.create_test(C)
    gff.evaluation.model_inference(args.model_folder, model, test_dl)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
