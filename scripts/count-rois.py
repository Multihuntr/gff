import argparse
import json
import sys
from pathlib import Path

import pandas
import geopandas

import gff.constants


def parse_args(argv):
    parser = argparse.ArgumentParser("Counts ROIs in main dataset.")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("--include_extras", action="store_true")

    return parser.parse_args(argv)


def main(args):
    roi_folder = args.data_path / "rois"
    print(f'{" ":10s}|{"ROIs":^25s}|{"Tiles":^25s}')
    print(
        f'{"Partition":^10s}|{"Generated":^12s}|{"Kuro Siwo":^12s}|{"Generated":^12s}|{"Kuro Siwo":^12s}'
    )
    print("-" * 62)
    for i in range(gff.constants.N_PARTITIONS):
        fpath = args.data_path / "partitions" / f"floodmap_partition_{i}.txt"
        fnames = pandas.read_csv(fpath, header=None)[0].values.tolist()
        n_gen_rois, n_ks_rois, n_gen_tiles, n_ks_tiles = 0, 0, 0, 0
        for fname in fnames:
            folder = roi_folder
            fpath = folder / fname
            if not fpath.exists():
                if args.include_extras:
                    folder = args.data_path / "extras" / "vit-snunet-extra"
                    fpath = folder / fname
                else:
                    continue

            with open(fpath) as f:
                meta = json.load(f)

            tile_path = folder / meta["visit_tiles"]
            tiles = geopandas.read_file(tile_path, use_arrow=True, engine="pyogrio")

            if meta["type"] == "generated":
                n_gen_rois += 1
                n_gen_tiles += len(tiles)
            elif meta["type"] == "kurosiwo":
                n_ks_rois += 1
                n_ks_tiles += len(tiles)

        print(f"{i:10d}|{n_gen_rois:12d}|{n_ks_rois:12d}|{n_gen_tiles:12d}|{n_ks_tiles:12d}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
