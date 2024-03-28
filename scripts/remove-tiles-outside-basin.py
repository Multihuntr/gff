import argparse
import json
import sys
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import shapely

import gff.util as util


def parse_args(argv):
    parser = argparse.ArgumentParser("Deletes tiles outside the majority basin boundary")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--lvl", type=int, default=4)

    return parser.parse_args(argv)


def main(args):
    basin_dir = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_path = basin_dir / f"BasinATLAS_v10_lev{args.lvl:02d}.shp"
    basins_df = geopandas.read_file(basin_path, engine="pyogrio")

    for meta_path in (args.data_path / "floodmaps").glob("*-meta.json"):
        with meta_path.open() as f:
            meta = json.load(f)

        visit_tiles = np.array(meta["visit_tiles"])
        flood_tiles = np.array(meta["flood_tiles"])
        _, visit_mask = util.tile_mask_for_basin(visit_tiles, basins_df)
        _, flood_mask = util.tile_mask_for_basin(flood_tiles, basins_df)

        # Write nodata to tiles outside majority basin
        with rasterio.open(args.data_path / meta["floodmap"], "r+") as tif:
            for tile in visit_tiles[visit_mask]:
                tile_geom = shapely.Polygon(tile)
                window = util.shapely_bounds_to_rasterio_window(tile_geom.bounds, tif.transform)
                (yhi, ylo), (xhi, xlo) = window
                tif.write(
                    np.full((tif.count, abs(yhi - ylo), abs(xhi - xlo)), tif.nodata), window=window
                )

        meta["visit_tiles"] = visit_tiles[~visit_mask]
        meta["flood_tiles"] = flood_tiles[~flood_mask]


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
