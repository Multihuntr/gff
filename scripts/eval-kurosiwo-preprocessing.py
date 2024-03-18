import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import shapely
import shapely.coordinates

import torch
import torch.nn as nn
import tqdm

import constants
import data_sources
import util


def load_tile(p):
    with rasterio.open(p) as tif:
        return tif.read()


def onehot(arr: np.ndarray, n: int = None):
    if len(arr.shape) == 3:
        if n is None:
            # If n is not known; interpret as arbitrary predicted map
            n = arr.shape[0]
            idx = arr.argmax(axis=0, keepdims=True)
        else:
            # if n is known; interpret as already a class index
            idx = arr
        ran = np.arange(n).reshape((-1, 1, 1))
        return (idx == ran).astype(np.uint8)
    elif len(arr.shape) == 4:
        if n is None:
            n = arr.shape[1]
            idx = arr.argmax(axis=1, keepdims=True)
        else:
            idx = arr
        ran = np.arange(n).reshape((1, -1, 1, 1))
        return (idx == ran).astype(np.uint8)


def parse_args(argv):
    parser = argparse.ArgumentParser("Evaluate my preprocessing script")

    parser.add_argument("kurosiwo_path", type=Path)
    parser.add_argument("s1_image_paths", type=Path, nargs="+")

    return parser.parse_args(argv)


def correct_scene(info, img_paths):
    # fname like: [FLOOD]-[HYBAS_ID]-[YYYY-MM-DD].tif
    img_date_strs = ["-".join(p.stem.split("-")[-3:]) for p in img_paths]
    dates_correct = (
        info["sources"]["MS1"]["source_date"] in img_date_strs
        and info["sources"]["SL1"]["source_date"] in img_date_strs
        and (len(img_date_strs) == 2 or info["sources"]["SL2"]["source_date"] in img_date_strs)
    )
    shp = shapely.from_wkt(info["geom"])
    footprint = util.image_footprint(img_paths[0])
    space_correct = shapely.intersection(shp, footprint).area == shp.area
    return dates_correct and space_correct


def run_kurosiwo_preprocessed_flood_vit(tile_folder: Path, model: nn.Module):
    files = list(tile_folder.iterdir())

    post_vv = load_tile(next((fname for fname in files if "MS1_IVV" in fname.stem)))
    post_vh = load_tile(next((fname for fname in files if "MS1_IVH" in fname.stem)))
    pre1_vv = load_tile(next((fname for fname in files if "SL1_IVV" in fname.stem)))
    pre1_vh = load_tile(next((fname for fname in files if "SL1_IVH" in fname.stem)))
    pre2_vv = load_tile(next((fname for fname in files if "SL2_IVV" in fname.stem)))
    pre2_vh = load_tile(next((fname for fname in files if "SL2_IVH" in fname.stem)))

    post = torch.tensor(np.concatenate((post_vv, post_vh))[None]).cuda()
    pre1 = torch.tensor(np.concatenate((pre1_vv, pre1_vh))[None]).cuda()
    pre2 = torch.tensor(np.concatenate((pre2_vv, pre2_vh))[None]).cuda()

    out = model([pre2, pre1, post])[0].cpu().numpy()
    return out


def run_kurosiwo_preprocessed_snunet(tile_folder: Path, model: nn.Module):
    files = list(tile_folder.iterdir())

    post_vv = load_tile(next((fname for fname in files if "MS1_IVV" in fname.stem)))
    post_vh = load_tile(next((fname for fname in files if "MS1_IVH" in fname.stem)))
    pre1_vv = load_tile(next((fname for fname in files if "SL1_IVV" in fname.stem)))
    pre1_vh = load_tile(next((fname for fname in files if "SL1_IVH" in fname.stem)))
    dem = load_tile(next((fname for fname in files if "MK0_DEM" in fname.stem)))

    post = torch.tensor(np.concatenate((post_vv, post_vh))[None]).cuda()
    pre1 = torch.tensor(np.concatenate((pre1_vv, pre1_vh))[None]).cuda()

    out = model([pre1, post], dem=dem)[0].cpu().numpy()
    return out


def compare(pred: np.ndarray, targ: np.ndarray, flood_class=constants.KUROSIWO_FLOOD_CLASS):
    if len(pred.shape) == 4:  # B C H W
        t = targ[flood_class][None]
        intersection = np.logical_and(pred[:, flood_class], t).sum(axis=(1, 2))
        union = np.logical_or(pred[:, flood_class], t).sum(axis=(1, 2))
    elif len(pred.shape) == 3:  # C H W
        intersection = np.logical_and(pred[flood_class], targ[flood_class]).sum()
        union = np.logical_or(pred[flood_class], targ[flood_class]).sum()
    return intersection, union


def main(args):
    torch.set_grad_enabled(False)
    flood_vit = torch.hub.load("Multihuntr/KuroSiwo", "vit_decoder", pretrained=True).cuda()
    snunet = torch.hub.load("Multihuntr/KuroSiwo", "snunet", pretrained=True).cuda()

    base_intersection, base_union = 0, 0
    mine_intersection, mine_union = 0, 0
    with rasterio.open(args.s1_image_paths[0]) as s1_tif:
        profile = {
            **s1_tif.profile,
            "compress": "deflate",
            "predictor": "2",
            "dtype": "uint8",
            "nodata": 255,
        }
    ksw_tif = rasterio.open(f"ksw.tif", "w", **{**profile, "count": 1, "dtype": "float32"})
    our_tif = rasterio.open(f"our.tif", "w", **{**profile, "count": 1, "dtype": "float32"})
    ksw_logit_tif = rasterio.open(
        f"ksw_logit.tif", "w", **{**profile, "count": 3, "dtype": "float32"}
    )
    our_logit_tif = rasterio.open(
        f"our_logit.tif", "w", **{**profile, "count": 3, "dtype": "float32"}
    )
    for targets_path in tqdm.tqdm(list(args.kurosiwo_path.glob("**/MK0_MLU_*.tif"))):
        with (targets_path.parent / "info.json").open() as f:
            info = json.load(f)

        # Only try to evaluate tiles within the flood event
        if not correct_scene(info, args.s1_image_paths):
            continue

        # Evaluate
        with rasterio.open(targets_path) as tif:
            targets = tif.read()
            target_onehot = onehot(targets, 3)
        shp = shapely.from_wkt(info["geom"])
        base_out_vit = run_kurosiwo_preprocessed_flood_vit(targets_path.parent, flood_vit)
        _, mine_out_vit = data_sources.run_flood_vit_once(args.s1_image_paths, shp, flood_vit)
        # base_out_snu = run_kurosiwo_preprocessed_snunet(targets_path.parent, snunet)
        # _, _, mine_out_snu = data_sources.run_snunet_once(args.s1_image_paths[1:], shp, snunet)
        base_out = base_out_vit  # + base_out_snu
        mine_out = mine_out_vit  # + mine_out_snu
        base_i, base_u = compare(onehot(base_out), target_onehot)
        mine_i, mine_u = compare(onehot(mine_out), target_onehot)

        # Accumulate
        base_intersection += base_i
        base_union += base_u
        mine_intersection += mine_i
        mine_union += mine_u

        # Export for visualisation
        window = util.shapely_bounds_to_rasterio_window(shp.bounds, ksw_tif.transform)
        ksw_logit_tif.write(base_out, window=window)
        our_logit_tif.write(mine_out, window=window)
        ksw_tif.write(base_out.argmax(axis=0)[None], window=window)
        our_tif.write(mine_out.argmax(axis=0)[None], window=window)
    ksw_logit_tif.close()
    our_logit_tif.close()
    ksw_tif.close()
    our_tif.close()

    base_iou = base_intersection / base_union
    mine_iou = mine_intersection / mine_union
    print(f"Kurosiwo: {base_intersection:10d}  {base_union:10d} | {base_iou:5.3f}")
    print(f"Mine:     {mine_intersection:10d}  {mine_union:10d} | {mine_iou:5.3f}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
