import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import shapely


def parse_args(argv):
    parser = argparse.ArgumentParser("Convert meta to new style")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def main(args):
    kwargs = {"columns": ["geometry"], "geometry": "geometry", "crs": "EPSG:4326"}
    count = 0
    for path in args.data_path.glob("**/*-meta.json"):
        count += 1
        with open(path) as f:
            meta = json.load(f)

        if isinstance(meta["visit_tiles"], list):
            tif_filename = Path(meta["floodmap"])
            tif_filename = Path(tif_filename.name)
            print(f"Writing to {path}")
            visit_filename = tif_filename.with_name(tif_filename.stem + "-visit.gpkg")
            flood_filename = tif_filename.with_name(tif_filename.stem + "-flood.gpkg")

            if len(meta["visit_tiles"]) > 0:
                visit_df = geopandas.GeoDataFrame(shapely.polygons(meta["visit_tiles"]), **kwargs)
            else:
                visit_df = geopandas.GeoDataFrame([], **kwargs)
            if len(meta["flood_tiles"]) > 0:
                flood_df = geopandas.GeoDataFrame(shapely.polygons(meta["flood_tiles"]), **kwargs)
            else:
                flood_df = geopandas.GeoDataFrame([], **kwargs)
            visit_df.to_file(path.parent / visit_filename)
            flood_df.to_file(path.parent / flood_filename)

            meta["visit_tiles"] = str(visit_filename)
            meta["flood_tiles"] = str(flood_filename)
            meta["floodmap"] = str(tif_filename)
            with open(path, "w") as f:
                json.dump(meta, f)
    print(f"Checked {count} files")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
