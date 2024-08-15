import argparse
import sys
from pathlib import Path

import geopandas
import json

import pandas
import shapely

import gff.data_sources


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("floodmaps_path", type=Path)
    parser.add_argument("partitions_path", type=Path)
    parser.add_argument("out_fpath", type=Path)
    parser.add_argument("dfo_path", type=Path)

    return parser.parse_args(argv)


THRESHOLD = 0.05


def main(args):
    dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    fpaths = list(args.floodmaps_path.glob("*-meta.json"))
    partition_lists = [
        pandas.read_csv(args.partitions_path / f"floodmap_partition_{i}.txt", header=None)[
            0
        ].values.tolist()
        for i in range(5)
    ]

    data = []
    for fpath in fpaths:
        with open(fpath) as f:
            meta = json.load(f)

        hybas_key = "HYBAS_ID" if "HYBAS_ID" in meta else "HYBAS_ID_4"
        continent = int(str(meta[hybas_key])[0])

        for i, partition_list in enumerate(partition_lists):
            if fpath.name in partition_list:
                partition = i
                break

        key = meta["key"]
        source = meta["type"]
        if meta["type"] == "generated":
            flood_id = int(key.split("-")[0])  # Yes, I know this is hacky... but it works!
            cause = dfo[dfo.ID == flood_id].MAINCAUSE.values.item()
        else:
            cause = None

        tile_fpath = fpath.parent / meta["visit_tiles"]
        tiles = geopandas.read_file(tile_fpath, engine="pyogrio", use_arrow=True)
        tiles = tiles.to_crs("EPSG:4326")
        centroid = shapely.unary_union(tiles.geometry).centroid
        n_tiles = len(tiles)
        total_pixels = tiles.n_background + tiles.n_permanent_water + tiles.n_flooded
        flood_proportion = tiles.n_flooded / total_pixels
        water_proportion = (tiles.n_flooded + tiles.n_permanent_water) / total_pixels
        n_fl_t = (flood_proportion > THRESHOLD).values.sum()
        n_anyw_t = (water_proportion > THRESHOLD).values.sum()
        n_pixels = total_pixels.values.sum()
        n_bg_px = tiles.n_background.values.sum()
        n_pw_px = tiles.n_permanent_water.values.sum()
        n_fl_px = tiles.n_flooded.values.sum()
        is_flooded = n_fl_t > 50
        data.append(
            (
                centroid,
                key,
                source,
                continent,
                partition,
                is_flooded,
                n_tiles,
                n_fl_t,
                n_anyw_t,
                n_pixels,
                n_bg_px,
                n_pw_px,
                n_fl_px,
                cause,
            )
        )

    df = geopandas.GeoDataFrame(
        data,
        columns=[
            "geometry",
            "key",
            "source",
            "continent",
            "partition",
            "is-flooded",
            "tiles",
            "flooded-tiles",
            "any-water-tiles",
            "pixels",
            "background-pixels",
            "permanent-water-pixels",
            "flooded-pixels",
            "cause",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    df.to_file(args.out_fpath)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
