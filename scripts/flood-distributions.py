import argparse
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas
import geopandas
import shapely


def parse_args(argv):
    parser = argparse.ArgumentParser("Calculate flood distributions")

    parser.add_argument("world_borders_path", type=Path)
    parser.add_argument("inform_risk_path", type=Path)
    parser.add_argument("flood_database_path", type=Path, help="Dartmouth flood database .shp")

    return parser.parse_args(argv)


def floods_per_continent_country(
    world_borders: geopandas.GeoDataFrame, floods_df: geopandas.GeoDataFrame
):
    # By DFO only
    floods_df_clean = floods_df.copy(deep=True)
    floods_df_clean["COUNTRY"] = floods_df_clean["COUNTRY"].str.strip()
    main = floods_df_clean.groupby("COUNTRY").count()
    other = floods_df_clean.groupby("OTHERCOUNT").count()
    dfo_counts = main[["ID"]].add(other[["ID"]], fill_value=0).astype(int)

    # By shapefile intersection with continent borders
    continent_borders = world_borders.groupby("continent").agg({"geometry": shapely.union_all})
    continent_borders = continent_borders.set_geometry("geometry")
    continent_intersects = shapely.intersects(
        np.array(continent_borders.geometry.values)[:, None],
        np.array(floods_df.geometry.values)[None],
    )
    continent_counts = pandas.DataFrame(
        list(zip(continent_borders.index, continent_intersects.sum(axis=1))),
        columns=["continent", "count"],
    )

    # By shapefile intersection with country borders
    country_borders = world_borders[world_borders.iso3.values != None].copy(deep=True)
    country_borders["iso3"] = country_borders["iso3"].astype(str)
    country_intersects = shapely.intersects(
        np.array(country_borders.geometry.values)[:, None],
        np.array(floods_df.geometry.values)[None],
    )
    country_counts = pandas.DataFrame(
        list(zip(country_borders.iso3.values, country_intersects.sum(axis=1))),
        columns=["iso3", "count"],
    )
    return dfo_counts, continent_counts, country_counts


def plot_floods_by_inform_gdp_proxy(inform_risk, country_counts):
    combined = country_counts.merge(inform_risk, on="iso3", how="inner")
    x = combined["infrastructure"]
    y = combined["count"]

    fig, ax = plt.subplots(1, 1, figsize=(19, 8))
    ax.scatter(x, y)
    ax.set_xlabel("INFORM Infrastructure index")
    ax.set_ylabel("Flood counts")
    fig.savefig("scraps/vis/inform.png")


def main(args):
    world_borders = geopandas.read_file(args.world_borders_path)
    inform_risk = pandas.read_csv(args.inform_risk_path)
    floods_df = geopandas.read_file(args.flood_database_path, engine="pyogrio")

    dfo_counts, continent_counts, country_counts = floods_per_continent_country(
        world_borders, floods_df
    )
    print(continent_counts)

    plot_floods_by_inform_gdp_proxy(inform_risk, country_counts)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
