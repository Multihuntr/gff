import argparse
import json
import sys
from pathlib import Path
import warnings

import geopandas
import torch

import dataset_generation


def parse_args(argv):
    parser = argparse.ArgumentParser("Run model on S1 images to generate flood maps")

    parser.add_argument("data_folder", type=Path, help="GFF Dataset root")
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("flood_id", type=int)
    parser.add_argument("hybas_id", type=int)
    parser.add_argument("--export_s1", action="store_true", help="export S1 alongside floodmaps")
    # parser.add_argument("hub_url", type=str, help="torch.hub url of flood mapping model")
    # parser.add_argument("model_name", type=str, help="torch.hub model name at url")

    return parser.parse_args(argv)


def main(args):
    """
    Final folder structure:
        basin_floods.gpkg
        index.json
        meta.json
        s1/
            [FLOOD_ID]-[HYBAS_ID]-[YYYYMMDD].tif    # contains both vv and vh
        floodmaps/
            [FLOOD_ID]-[HYBAS_ID].tif
        shps.gpkg

    meta.json structured like:
    [
        {
            'HYBAS_ID': <HYBAS_ID>,
            'FLOOD': <FLOOD_ID>,
            'BEGAN': <date isoformat>,
            'ENDED': <date isoformat>,
            'pre1': 's1/[FLOOD_ID]-[HYBAS_ID]-[YYYMMDD].tif',
            'pre2': 's1/[FLOOD_ID]-[HYBAS_ID]-[YYYMMDD].tif',
            'post': 's1/[FLOOD_ID]-[HYBAS_ID]-[YYYMMDD].tif',
            'floodmap': 'floodmaps/[FLOOD_ID]-[HYBAS_ID].tif',
            'tiles': [<idx>, <idx>, <idx>, ...] # indices into shps.gpkg
        }, ...
    ]
    """
    torch.set_grad_enabled(False)
    run_flood_model = dataset_generation.vit_decoder_runner()
    rivers_df = dataset_generation.load_rivers(args.hydroatlas_path)

    # Get search results
    with open(args.data_folder / "index.json") as f:
        s1_index = json.load(f)
    search_results = s1_index[f"{args.flood_id}-{args.hybas_id}"]

    # Get basin row
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
    )

    # Save the meta details to the meta index file
    # NOT MULTIPROCESSING SAFE!
    metas_fpath = args.data_folder / "meta.json"
    if metas_fpath.exists():
        with open(args.data_folder / "meta.json") as f:
            metas = json.load(f)
    else:
        metas = []
    metas.append(meta)

    with open(args.data_folder / "metas.json", "w") as f:
        json.dump(metas, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
