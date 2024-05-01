import argparse
import datetime
import json
from pathlib import Path
import sys

import geopandas
import shapely
import tqdm

import gff.constants
import gff.generate.basins


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("floodmap_name", type=str, default="vit")
    parser.add_argument("--hydroatlas_ver", type=int, default=10)
    parser.add_argument("--sites", type=int, default=200)

    return parser.parse_args(argv)


def main(args):
    # NOTE: IMPORTANT. This code is mostly copied from places in gen-whole-dataset.py
    # Any modifications should be in both places. I know, I know, DRY.
    # But do you realise how much work that is for this?

    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    basins08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    basins08_df = geopandas.read_file(
        basin_path / basins08_fname, use_arrow=True, engine="pyogrio"
    )

    flood_cache = args.data_path / "flood_distribution.json"
    # Hackily read the flood distribution
    flood_distr = gff.generate.basins.flood_distribution(None, None, cache_path=flood_cache)
    exp_cont_sites, exp_basin_distr = gff.generate.basins.mk_expected_distribution(
        flood_distr, n_sites=args.sites
    )

    cur_basin_flood_distr = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    cur_basin_noflood_distr = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    current_distr_tiles = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    paths = list((args.data_path / "floodmaps" / args.floodmap_name).glob("*-meta.json"))
    for meta_path in tqdm.tqdm(paths):
        with open(meta_path) as f:
            meta = json.load(f)
        hybas_key = "HYBAS_ID" if "HYBAS_ID" in meta else "HYBAS_ID_4"
        continent = int(str(meta[hybas_key])[0])

        geoms = geopandas.read_file(meta_path.parent / meta["visit_tiles"], engine="pyogrio")
        basin_counts = gff.generate.basins.zone_counts_shp(
            basins08_df, shapely.ops.unary_union(geoms)
        )
        for clim_zone, count in basin_counts.iterrows():
            if meta["flooding"]:
                cur_basin_flood_distr[continent][clim_zone] += count.item()
            else:
                cur_basin_noflood_distr[continent][clim_zone] += count.item()

    distr = {
        "floods": flood_distr,
        "expected": exp_basin_distr,
        "created": cur_basin_flood_distr,
        "negatives": cur_basin_noflood_distr,
        "created_tiles": current_distr_tiles,
    }
    with open(args.data_path / "measured_distribution.json", "w") as f:
        json.dump(distr, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
