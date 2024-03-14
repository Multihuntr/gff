import argparse
from pathlib import Path
import sys

import shapely
import torch

import dataset_generation


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Test dataset main dataset generation function on known images"
    )

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("geom", type=shapely.from_wkt, default=None)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("img_paths", nargs="+", type=Path)

    return parser.parse_args(argv)


def main(args):
    torch.set_grad_enabled(False)
    flood_model = torch.hub.load("Multihuntr/KuroSiwo", "vit_decoder", pretrained=True).cuda()
    rivers = dataset_generation.load_rivers(args.hydroatlas_path)
    dataset_generation.progressively_grow_floodmaps(
        args.img_paths,
        rivers,
        args.geom,
        args.out_path,
        flood_model,
        include_s1=True,
        print_freq=100,
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
