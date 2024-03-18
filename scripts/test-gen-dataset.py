import argparse
import json
from pathlib import Path
import sys

import geopandas
import shapely
import torch

import dataset_generation


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Test dataset main dataset generation function on known images"
    )

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("gpkg_path", type=Path, help="Basin x Floods")
    parser.add_argument("--s1_img_paths", nargs="+", type=Path)
    parser.add_argument("--geom", type=shapely.from_wkt, default=None)
    parser.add_argument("--out_path", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Otherwise just tests grow floodmaps on predefined test images",
    )
    parser.add_argument("--flood_id", type=int, help="if full, choose a flood")
    parser.add_argument("--hybas_id", type=int, help="if full, choose a basin")
    parser.add_argument("--img_path", type=Path, help="location to save floodmaps")
    parser.add_argument("--s1_index_path", type=Path, help="index of s1 search results")

    return parser.parse_args(argv)


def main(args):
    torch.set_grad_enabled(False)
    flood_model = torch.hub.load("Multihuntr/KuroSiwo", "vit_decoder", pretrained=True).cuda()
    rivers = dataset_generation.load_rivers(args.hydroatlas_path)

    if not args.full:
        dataset_generation.progressively_grow_floodmaps(
            args.s1_img_paths,
            rivers,
            args.geom,
            args.out_path,
            flood_model,
            include_s1=True,
            print_freq=100,
        )
    else:
        # Get search results
        with open(args.s1_index_path) as f:
            s1_index = json.load(f)
        search_results = s1_index[f"{args.flood_id}-{args.hybas_id}"]

        # Get basin shape
        basin_floods = geopandas.read_file(args.gpkg_path, engine="pyogrio")
        is_flood = basin_floods["ID"] == args.flood_id
        is_hybas = basin_floods["HYBAS_ID"] == args.hybas_id
        basin_shp = basin_floods.loc[is_flood & is_hybas].iloc[0]

        # Try to generate some floodmaps!
        if len(search_results["asc"]) > 0:
            dataset_generation.create_flood_maps(
                args.img_path, search_results["asc"], basin_shp, rivers, flood_model
            )
        elif len(search_results["desc"]) > 0:
            dataset_generation.create_flood_maps(
                args.img_path, search_results["desc"], basin_shp, rivers, flood_model
            )
        else:
            print("Not enough Sentinel-1 images at this location.")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
