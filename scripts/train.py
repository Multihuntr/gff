import argparse
import sys
from pathlib import Path

import yaml

import gff.models as models
import gff.training as training
import gff.dataloaders as dataloaders
import gff.evaluation as evaluation


def pair(str):
    k, v = str.split("|")
    if v.isdigit():
        v = int(v)
    else:
        try:
            v = float(v)
        except ValueError:
            pass
    return k, v


def parse_args(argv):
    parser = argparse.ArgumentParser("Train a model")

    parser.add_argument("config_path", type=Path)
    parser.add_argument("--overwrite", "-o", type=pair, nargs="*", help="Overwrite config setting")

    return parser.parse_args(argv)


def main(args):
    with open(args.config_path) as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    # Set up experiment folder

    model = models.create(C)

    dataloaders = dataloaders.create(C)

    training.main_loop(C, model, dataloaders)

    evaluation.evaluate(C, model)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
