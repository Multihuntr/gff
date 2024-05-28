import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import shapely
import tqdm

import gff.generate.floodmaps


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Runs postprocessing on the raw files and pastes over the ROI floodmaps"
    )

    parser.add_argument("data_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("rois_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v10_shp"
    basins04_fname = f"BasinATLAS_v10_lev04.shp"
    basins_df = geopandas.read_file(basin_path / basins04_fname, use_arrow=True, engine="pyogrio")
    basins_geom = shapely.unary_union(np.array(basins_df.geometry.values))

    fpaths = list(args.rois_path.glob("*-meta.json"))
    for fpath in tqdm.tqdm(fpaths):
        with open(fpath) as f:
            meta = json.load(f)

        floodmap_path = fpath.parent / meta["floodmap"]
        raw_floodmap_path = floodmap_path.with_name(floodmap_path.stem + "-raw.tif")
        if meta["type"] == "generated":

            gff.generate.floodmaps.postprocess_classes(raw_floodmap_path, floodmap_path)

            gff.generate.floodmaps.remove_tiles_outside(fpath, basins_df)

            gff.generate.floodmaps.postprocess_world_cover(
                args.data_path, fpath, basins_df, basins_geom
            )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
