import argparse
import json
from pathlib import Path
import sys

import rasterio
import shapely
import numpy as np
import tqdm

import gff.constants as constants
import gff.util as util


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("folder1", type=Path)
    parser.add_argument("folder2", type=Path)

    return parser.parse_args(argv)


def check_whole_metas(folder1, folder2, m1, m2):
    metas1_not_in2 = m1.difference(m2)
    if len(metas1_not_in2) > 0:
        print(f"These files are in {folder1}, but not in {folder2}")
        for p in metas1_not_in2:
            print(f"  {str(p)}")
        return True
    return False


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


def main(args):
    metas1 = set([Path(p.name) for p in list(args.folder1.glob("*-meta.json"))])
    metas2 = set([Path(p.name) for p in list(args.folder2.glob("*-meta.json"))])

    any_missing1 = check_whole_metas(args.folder1, args.folder2, metas1, metas2)
    any_missing2 = check_whole_metas(args.folder2, args.folder1, metas2, metas1)
    if not any_missing1 and not any_missing2:
        print("Identical sites chosen.")

    cn = constants.KUROSIWO_CLASS_NAMES

    intersection, union = np.zeros(3), np.zeros(3)
    counts1, counts2 = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    tiles_intersection, tiles_union = 0, 0
    for meta_fname in metas1.intersection(metas2):
        with open(args.folder1 / meta_fname) as f:
            meta1 = json.load(f)
        with open(args.folder2 / meta_fname) as f:
            meta2 = json.load(f)
        with rasterio.open(args.folder1 / Path(meta1["floodmap"]).name) as tif1:
            with rasterio.open(args.folder2 / Path(meta2["floodmap"]).name) as tif2:

                if not np.allclose(np.array(tif1.transform), np.array(tif2.transform)):
                    print(f" {meta_fname.stem}: Different transform")

                tiles1 = np.array([shapely.Polygon(t) for t in meta1["visit_tiles"]])
                tiles2 = np.array([shapely.Polygon(t) for t in meta2["visit_tiles"]])
                tiles1_px = util.convert_affine(tiles1.copy(), ~tif1.transform)
                tiles2_px = util.convert_affine(tiles2.copy(), ~tif1.transform)
                tiles1_px_int = set(util.convert_shp_inplace(tiles1_px, lambda c: np.round(c)))
                tiles2_px_int = set(util.convert_shp_inplace(tiles2_px, lambda c: np.round(c)))
                tiles_union += len(tiles1_px_int.union(tiles2_px_int))
                tiles_intersection += len(tiles1_px_int.intersection(tiles2_px_int))

                intersection_i, union_i = np.zeros(3), np.zeros(3)
                counts1_i, counts2_i = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)

                for tile in tiles1:
                    # Read tile
                    window1 = util.shapely_bounds_to_rasterio_window(tile.bounds, tif1.transform)
                    window2 = util.shapely_bounds_to_rasterio_window(tile.bounds, tif2.transform)
                    data1 = tif1.read(window=window1)
                    if data1.min() == 255:
                        continue
                    data2 = tif2.read(window=window2)
                    if data2.min() == 255:
                        continue
                    data1_oh = onehot(data1, 3)
                    data2_oh = onehot(data2, 3)

                    # Accumulate stats
                    intersection_i += np.logical_and(data1_oh, data2_oh).sum(axis=(1, 2))
                    union_i += np.logical_or(data1_oh, data2_oh).sum(axis=(1, 2))
                    counts1_i += [(data1 == j).sum() for j in range(3)]
                    counts2_i += [(data2 == j).sum() for j in range(3)]

                px_iou = intersection_i / union_i
                print(f"{meta_fname.stem}", " ".join([f"      {iou:6.4f}" for iou in px_iou]))

                intersection += intersection_i
                union += union_i
                counts1 += counts1_i
                counts2 += counts2_i

    print(f"Tilewise IoU  : {tiles_intersection/tiles_union:6.4f}")

    px_iou = intersection / union
    print(f"               ", " ".join([f"{cn:^12s}" for cn in constants.KUROSIWO_CLASS_NAMES]))
    print(f"Pixelwise IoU :", " ".join([f"      {iou:6.4f}" for iou in px_iou]))
    print(f"Pixels before :", " ".join([f"{c:12d}" for c in counts1]))
    print(f"Pixels after  :", " ".join([f"{c:12d}" for c in counts2]))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
