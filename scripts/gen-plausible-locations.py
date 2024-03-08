import argparse
import csv
import sys
import os
from pathlib import Path

import geopandas
import tqdm
import pandas as pd

INCLUDE_TYPES = [
    "Cyclone/storm",
    "Heavy Rain",
    "Heavy Rain AND Cyclone/storm",
    "Heavy Rain AND Tides/Surge",
    "Tides/Surge",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Process HydroATLAS geometries and flood database to find plausible compound flood events"
    )

    parser.add_argument("hydroatlas_path", type=Path, help="Folder with Basin and River ATLAS")
    parser.add_argument("flood_database_path", type=Path, help="Dartmouth flood database .shp")
    parser.add_argument("classifications_csv", type=Path, help="CSV of classification types")
    parser.add_argument("out_path", type=Path, help=".gpkg file path for output")
    parser.add_argument("--basin_level", type=int, default=8)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)
    parser.add_argument("--min_river_vol", type=int, default=20000, help="Basin condition")
    parser.add_argument("--min_population", type=int, default=1000, help="Basin condition")
    parser.add_argument("--min_dead", type=int, default=20, help="Flood condition")
    parser.add_argument("--min_displaced", type=int, default=10000, help="Flood condition")

    return parser.parse_args(argv)


def static_filter_basins_df(
    basins_df: geopandas.GeoDataFrame, min_river_vol: int, min_population: int
):
    """
    Filter basins using (relatively) static properties of the river the basin is in.

    Args:
        basins_df (geopandas.GeoDataFrame): Collection of basin geometries
        min_river_vol (int): Remove basins with total upstream volume less than this ('000 m3)
        min_population (int): Remove basins with total upstream population less than this

    Returns:
        geopandas.GeoDataFrame: Filtered dataframe
    """
    # Filter by basins that drain into the ocean
    query = (basins_df["NEXT_DOWN"] == 0) & (basins_df["ENDO"] == 0)

    # Filter by flow
    query &= basins_df["riv_tc_usu"] >= min_river_vol

    # Filter by population
    query &= basins_df["pop_ct_usu"] >= min_population

    # Apply filter
    return basins_df[query]


def filter_floods_df(
    floods_df: geopandas.GeoDataFrame,
    classification_types: pd.DataFrame,
    min_dead: int,
    min_displaced: int,
):
    """
    Filter flood events caused by known unimportant types,
    as well as those with a lower impact.

    Args:
        floods_df (geopandas.GeoDataFrame): Dataframe of flood events
        classification_types (pd.DataFrame): Manually grouped types of MAINCAUSE in floods_df
        min_dead (int): Include flood events with at least this many dead
        min_displaced (int): Include flood events with at least this many displaced

    Returns:
        geopandas.GeoDataFrame: Filtered dataframe
    """
    # Filter by enough people affected
    query = (floods_df["DEAD"] >= min_dead) | (floods_df["DISPLACED"] >= min_displaced)

    # Filter by known flood types we care about
    ct = classification_types
    care_groups = ct[ct["group"].isin(INCLUDE_TYPES)]
    care_names = ";".join(care_groups["all_names"]).split(";")
    query &= floods_df["MAINCAUSE"].isin(care_names)

    return floods_df[query]


def find_overlaps(basins_df: geopandas.GeoDataFrame, floods_df: geopandas.GeoDataFrame):
    """
    Overlap the geometries of the floods with the basins to find co-occurances of each.

    Args:
        basins_df (geopandas.GeoDataFrame): Basins (attributes and geometries)
        floods_df (geopandas.GeoDataFrame): Flood events

    Returns:
        geopandas.GeoDataFrame: Inner join of Basins and floods which overlap
    """
    # Some flood polygons self intersect (e.g. bowtie);
    # applying buffer(0) makes it valid by creating a single outer boundary
    floods_df = floods_df.to_crs(basins_df.crs)
    floods_df.geometry = floods_df.geometry.buffer(0)

    # Join by overlap area > 0 - keeping the basin geometry
    joined = basins_df.sjoin(floods_df, how="inner", rsuffix="flood")
    return joined


def main(args):
    print("Loading basins...")
    basin_path = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev{args.basin_level:02d}.shp"
    basins_df = geopandas.read_file(basin_path / basin_fname, engine="pyogrio")
    print(" - filtering basins...")
    basins_df = static_filter_basins_df(basins_df, args.min_river_vol, args.min_population)

    print("Loading floods...")
    floods_df = geopandas.read_file(args.flood_database_path, engine="pyogrio")
    print(" - filtering floods...")
    classification_types = pd.read_csv(args.classifications_csv, quotechar="'")
    floods_df = filter_floods_df(
        floods_df, classification_types, args.min_dead, args.min_displaced
    )

    print("Cross-checking floods with basins...")
    basin_floods = find_overlaps(basins_df, floods_df)

    print("Write results to disk...")
    basin_floods.to_file(args.out_path)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
