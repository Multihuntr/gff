import argparse
import json
import re
import sys
from pathlib import Path

import geopandas
import numpy as np
import shapely
import pandas
import tqdm
import torch
import yaml

import gff.constants
import gff.dataloaders
import gff.evaluation
import gff.models.creation
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Evaluate floodmaps stored in the same format as the labels")

    parser.add_argument("model_folder", type=Path)
    parser.add_argument(
        "worldcover_path",
        type=Path,
        help="Folder containing exported WorldCover files, one per ROI",
    )
    parser.add_argument("hydroatlas_path", type=Path)
    # parser.add_argument(
    #     "--blockout_ks_pw",
    #     action="store_true",
    #     help="When evaluating Kuro Siwo Labels, don't consider permanent water",
    # )
    # parser.add_argument(
    #     "--blockout_worldcover_water",
    #     action="store_true",
    #     help="For main evaluation, don't consider pixels in WorldCover water class",
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="evaluating on GPU device is much faster with torchmetrics",
    )
    parser.add_argument("--coast_buffer", type=float, default=0.1)
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


def get_coast_masks(
    folder: Path,
    fnames: list[str],
    basins_df: geopandas.GeoDataFrame,
    world_coast: shapely.Geometry,
    coast_buffer: float = 0.3,
):
    coast_masks = {}
    for fname in tqdm.tqdm(fnames, "Getting coast masks"):
        if not (folder / fname).exists():
            continue
        with open(folder / fname) as f:
            meta = json.load(f)
        tile_fpath = folder / meta["visit_tiles"]
        tiles = geopandas.read_file(tile_fpath, engine="pyogrio", use_arrow=True)
        if tiles.crs != "EPSG:4326":
            tiles.to_crs("EPSG:4326")

        basin_idx = basins_df.HYBAS_ID.values.tolist().index(meta["HYBAS_ID_4"])
        basin_geoms = np.array(basins_df.geometry.values)

        basin_geom = basin_geoms[basin_idx]
        basin_buffer = shapely.buffer(basin_geom, coast_buffer)
        # For speed: only intersect tiles with a smaller geometry
        basin_coast = shapely.intersection(basin_buffer, world_coast)

        # Intersect with tile
        tile_geoms = np.array(tiles.geometry.values)[:, None]
        mask = shapely.intersects(tile_geoms, basin_coast)
        coast_masks[fname] = mask
    return coast_masks


def main(args):
    inf_fpath = args.model_folder / "inference"
    if not inf_fpath.exists():
        print(f"Evaluation requires inferences in {inf_fpath}.")
        sys.exit(0)

    # Load config file, and overwrite anything from cmdline arguments
    with open(args.model_folder / "config.yml") as f:
        C = yaml.safe_load(f)
    for k, v in args.overwrite:
        C[k] = v

    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v10_shp"
    basins04_fname = f"BasinATLAS_v10_lev04.shp"
    basins_df = geopandas.read_file(basin_path / basins04_fname, use_arrow=True, engine="pyogrio")
    coast_fpath = args.hydroatlas_path / f"world_coast_{args.coast_buffer}.gpkg"
    if coast_fpath.exists():
        world_coast = geopandas.read_file(coast_fpath, use_arrow=True, engine="pyogrio")
        world_coast = world_coast.geometry
    else:
        print("Combining geometries to find world coastline (approx 20 minutes)...", end="")
        coast_geom = shapely.unary_union(np.array(basins_df.geometry.values)).simplify(0.01)
        # Create coast shape - two shapes: original, negative buffer, difference to get buffer zone
        # Note: "coast" includes "ocean", else ocean tiles wouldn't be counted.
        neg_buffer = shapely.buffer(coast_geom, -args.coast_buffer)
        pos_buffer = shapely.buffer(coast_geom, args.coast_buffer)
        world_coast = shapely.difference(pos_buffer, neg_buffer)
        print("done!")
        data = {"geometry": [world_coast]}
        gdf = geopandas.GeoDataFrame(data, geometry="geometry", crs=basins_df.crs)
        gdf.to_file(coast_fpath)

    # Determine fnames
    data_path = Path(C["data_folder"]).expanduser()
    fold_names_fpath = data_path / "partitions" / f"floodmap_partition_{C['fold']}.txt"
    fnames = pandas.read_csv(fold_names_fpath, header=None)[0].values.tolist()
    targ_path = data_path / "rois"
    out_path = args.model_folder / "inference"
    n_cls = C["n_classes"]

    # Base results
    coast_masks_fname = Path("/tmp") / f'coast_masks_{C["fold"]}_{args.coast_buffer}.npy'
    if not coast_masks_fname.exists():
        coast_masks = get_coast_masks(targ_path, fnames, basins_df, world_coast, args.coast_buffer)
        torch.save(coast_masks, coast_masks_fname)
    coast_masks = torch.load(coast_masks_fname)

    blockout = gff.evaluation.processing_blockout_fnc(args.worldcover_path, "worldcover-water")
    eval_results, test_cm = gff.evaluation.evaluate_floodmaps(
        fnames,
        out_path,
        targ_path,
        n_cls,
        coast_masks,
        extra_processing=blockout,
        device=args.device,
    )
    gff.evaluation.save_results(args.model_folder / "eval_results.yml", eval_results)
    gff.evaluation.save_cm(test_cm, n_cls, "Test", args.model_folder / "test_cm.png")

    # On KuroSiwo labels
    blockout = gff.evaluation.processing_blockout_fnc(None, "kurosiwo-pw")
    ks_fnames = [fname for fname in fnames if fname_is_ks(fname)]
    eval_results, test_cm = gff.evaluation.evaluate_floodmaps(
        ks_fnames, out_path, targ_path, n_cls, extra_processing=blockout, device=args.device
    )
    gff.evaluation.save_results(args.model_folder / "eval_results_ks.yml", eval_results)
    gff.evaluation.save_cm(test_cm, n_cls, "Test", args.model_folder / "test_cm_ks.png")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
