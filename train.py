import argparse
import datetime
import sys
from pathlib import Path

import torch
import yaml

import gff.dataloaders
import gff.evaluation
import gff.models.creation
import gff.training
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Train a model")

    parser.add_argument("config_path", type=Path)
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
    with open(args.config_path) as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    # Set seed for reproducibility
    gff.util.seed_packages(C["seed"])

    # Instantiate the model
    model = gff.models.creation.create(C)
    model.to(C["device"])

    # Prepare data loaders
    g = torch.Generator()
    g.manual_seed(C["seed"])
    train_dl, val_dl, test_dl = gff.dataloaders.create(C, g)

    # Train the model
    model_folder = Path("runs") / f'{C["name"]}_{datetime.datetime.now().isoformat()}'
    model_folder.mkdir(parents=True, exist_ok=True)
    with open(model_folder / "config.yml", "w") as f:
        yaml.safe_dump(C, f)
    gff.training.training_loop(C, model_folder, model, (train_dl, val_dl))

    # Evaluate the model
    eval_results = gff.evaluation.evaluate_model(model, test_dl)
    with open(model_folder / "eval_results.yml", "w") as f:
        yaml.safe_dump(eval_results, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
