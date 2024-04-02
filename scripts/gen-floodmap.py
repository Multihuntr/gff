import argparse
import json
import sys
from pathlib import Path
import warnings

import geopandas
import numpy as np
import torch

import gff.dataset_generation as dataset_generation


def parse_args(argv):
    parser = argparse.ArgumentParser("Run model on S1 images to generate flood maps")

    parser.add_argument("data_folder", type=Path, help="GFF Dataset root")
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("flood_id", type=int)
    parser.add_argument("hybas_id", type=int)
    parser.add_argument("--export_s1", action="store_true", help="export S1 alongside floodmaps")
    # parser.add_argument("hub_url", type=str, help="torch.hub url of flood mapping model")
    parser.add_argument(
        "--model", type=str, default="vit", choices=["vit", "snunet", "vit+snunet"]
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if floodmap already exists"
    )
    parser.add_argument("--floodmap_folder", type=str, default=None)

    return parser.parse_args(argv)


def main(args):
    """
    Final folder structure:
        basin_floods.gpkg
        index.json
        s1/
            [FLOOD_ID]-[HYBAS_ID]-[YYYYMMDD].tif    # contains both vv and vh
        floodmaps/
            [FLOOD_ID]-[HYBAS_ID]-[YYYY-MM-DD]-[YYYY-MM-DD]-[YYYY-MM-DD].tif
            [FLOOD_ID]-[HYBAS_ID]-[YYYY-MM-DD]-[YYYY-MM-DD]-[YYYY-MM-DD]-meta.json
    """
    key = f"{args.flood_id}-{args.hybas_id}"
    if not (args.overwrite):
        if dataset_generation.check_floodmap_exists(args.data_folder, key, args.floodmap_folder):
            print(f"Floodmap for {key} already exists, not overwriting.")
            return
    print(f"Creating floodmaps for {key}.")

    torch.set_grad_enabled(False)
    if args.model == "vit":
        run_flood_model = dataset_generation.vit_decoder_runner()
    elif args.model == "snunet":
        run_flood_model = dataset_generation.snunet_runner("EPSG:4326", args.data_folder)
    elif args.model == "vit+snunet":
        run_flood_model = dataset_generation.average_vit_snunet_runner(
            "EPSG:4326", args.data_folder
        )
    rivers_df = dataset_generation.load_rivers(args.hydroatlas_path)

    # Get search results
    with open(args.data_folder / "index.json") as f:
        s1_index = json.load(f)
    search_results = s1_index[key]

    # Get basin x flood row
    gpkg_path = args.data_folder / "basin_floods.gpkg"
    cond = "BEGAN >= '2014-01-01'"
    with warnings.catch_warnings(action="ignore"):
        basin_floods = geopandas.read_file(gpkg_path, engine="pyogrio", where=cond)
    is_flood = basin_floods["ID"] == args.flood_id
    is_hybas = basin_floods["HYBAS_ID"] == args.hybas_id
    basin_row = basin_floods.loc[is_flood & is_hybas].iloc[0]

    # Generate some floodmaps!
    if len(search_results["asc"]) > 0:
        approach_results = search_results["asc"]
    elif len(search_results["desc"]) > 0:
        approach_results = search_results["desc"]
    else:
        print("Not enough Sentinel-1 images at this basin/flood.")
        return
    meta = dataset_generation.create_flood_maps(
        args.data_folder,
        approach_results,
        basin_row,
        rivers_df,
        run_flood_model,
        export_s1=args.export_s1,
        floodmap_folder=args.floodmap_folder,
    )

    print(f"Saved {meta['FLOOD']}-{meta['HYBAS_ID']} to: {meta['floodmap']}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
