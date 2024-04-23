import argparse
import json
import sqlite3
import sys
from pathlib import Path
import warnings

import geopandas
import numpy as np
import torch

import gff.data_sources
import gff.generate.basins
import gff.generate.floodmaps
import gff.generate.search


def parse_args(argv):
    parser = argparse.ArgumentParser("Run model on S1 images to generate flood maps")

    parser.add_argument("data_folder", type=Path, help="GFF Dataset root")
    parser.add_argument("dfo_path", type=Path)
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
        if gff.generate.floodmaps.check_floodmap_exists(
            args.data_folder, key, args.floodmap_folder
        ):
            print(f"Floodmap for {key} already exists, not overwriting.")
            return
    print(f"Creating floodmaps for {key}.")

    torch.set_grad_enabled(False)
    run_flood_model = gff.generate.floodmaps.model_runner(args.model, args.data_folder)
    rivers_df = gff.generate.floodmaps.load_rivers(args.hydroatlas_path)

    # Get basin x flood row
    basin_path = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basins08_fname = f"BasinATLAS_v10_lev08.shp"
    basins08_df = geopandas.read_file(basin_path / basins08_fname, engine="pyogrio")
    dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    basin_floods = gff.generate.basins.coastal_basin_x_floods(basins08_df, dfo)
    is_flood = basin_floods["ID"] == args.flood_id
    is_hybas = basin_floods["HYBAS_ID"] == args.hybas_id
    basin_row = basin_floods.loc[is_flood & is_hybas].iloc[0]

    # Get search results
    p = args.data_folder / f"s1" / f"index.db"
    s1_index = sqlite3.connect(p, timeout=10)
    search_results = gff.generate.search.get_search_results(s1_index, basin_row)
    search_results, _ = gff.generate.search.filter_search_results(search_results, basin_row)

    # Generate some floodmaps!
    if len(search_results["asc"]) > 0:
        approach_results = search_results["asc"]
    elif len(search_results["desc"]) > 0:
        approach_results = search_results["desc"]
    else:
        print("Not enough Sentinel-1 images at this basin/flood.")
        return
    metas = gff.generate.floodmaps.create_flood_maps(
        args.data_folder,
        approach_results,
        basin_row,
        rivers_df,
        run_flood_model,
        export_s1=args.export_s1,
        floodmap_folder=args.floodmap_folder,
    )

    for meta in metas:
        print(f"Saved {meta['FLOOD']}-{meta['HYBAS_ID']} to: {meta['floodmap']}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
