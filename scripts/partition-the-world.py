import argparse
import json
from pathlib import Path
import random
import sys

import geopandas
import numpy as np
import pandas
import shapely

import gff.util as util


def parse_args(argv):
    parser = argparse.ArgumentParser("Split basins into partitions")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("caravan_path", type=Path)
    parser.add_argument("out_folder", type=Path)
    parser.add_argument("ks_agg_labels_path", type=Path)
    parser.add_argument("floodmaps_path", type=Path)
    parser.add_argument("--n_partitions", type=int, default=5)
    parser.add_argument("--lvl", type=int, default=4)
    parser.add_argument("--trials", type=int, default=20)

    return parser.parse_args(argv)


def allocate_caravan(
    caravan_path: Path, partitions: list[geopandas.GeoDataFrame]
) -> list[pandas.DataFrame]:
    # Get list of caravan gauges, with geometry
    dfs = [
        pandas.read_csv(x)
        for x in (caravan_path / "attributes").glob("**/attributes_caravan_*.csv")
    ]
    caravan_df = pandas.concat(dfs)
    caravan_df["geometry"] = shapely.points(caravan_df["gauge_lon"], y=caravan_df["gauge_lat"])
    caravan_gdf = geopandas.GeoDataFrame(caravan_df, geometry="geometry")

    # Intersect the caravan gauges with each partition individually
    caravan_partitions: list[geopandas.GeoDataFrame] = []
    caravan_geoms = np.array(caravan_gdf.geometry.values)[:, None]
    for partition in partitions:
        partition_geom = np.array(partition.geometry.values)[None, :]
        index = shapely.within(caravan_geoms, partition_geom)
        caravan_partition = caravan_gdf[index.any(axis=1)]
        caravan_partitions.append(caravan_partition[["gauge_id"]])

    # Sanity-check the results
    covered = pandas.concat(caravan_partitions)
    shared = caravan_df.merge(covered, how="left")
    assert len(shared) == len(caravan_df), "Some gauges not assigned to a partition"

    return caravan_partitions


def allocate_by_points(points: np.ndarray, partitions: list[shapely.GeometryCollection]):
    return [shapely.within(points, partition) for partition in partitions]


def calc_distribution_score(*counts: list[np.ndarray]):
    scores = []
    for c in counts:
        scale = np.asarray(c).sum()
        score = np.std(c / scale)
        scores.append(score)
    return np.mean(scores)


def center_of(p: Path):
    footprint = util.image_footprint(p)
    footprint = util.convert_crs(footprint, "EPSG:3857", "EPSG:4326")
    return shapely.centroid(footprint)


def get_tiles(p: Path):
    with p.open() as f:
        meta = json.load(f)
    return [shapely.Polygon(t) for t in meta["visit_tiles"]]


def count_within(points, partition):
    within = np.any(shapely.within(points[:, None], partition[None, :]), axis=1)
    return within.sum()


def count_groups(partitions, column, index):
    partition_groups = [util.count_group(p, column, index) for p in partitions]
    return np.array(partition_groups).T


N_CONTINENTS = 9
N_CLIMATE_ZONES = 18
CONTINENT_INDEX = list(range(1, N_CONTINENTS + 1))
CLIMATE_ZONE_INDEX = list(range(1, N_CLIMATE_ZONES + 1))


def main(args):
    basin_dir = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_path = basin_dir / f"BasinATLAS_v10_lev{args.lvl:02d}.shp"
    basin_df = geopandas.read_file(basin_path, engine="pyogrio")
    # Add column for grouping and counting in partitions
    basin_df["continent"] = basin_df.HYBAS_ID.astype(str).str[0].astype(int)

    ks_points = np.array([center_of(path) for path in args.ks_agg_labels_path.glob("*.tif")])
    tile_points = [
        shapely.centroid(shapely.union_all(get_tiles(path)))
        for path in args.floodmaps_path.glob("*-meta.json")
    ]
    tile_points = np.array(tile_points)

    # Allocate randomly, then check distribution, and pick the most well distributed
    best_score = 100
    best = None
    for x in range(args.trials):
        # Create partitions by shuffling the index
        idx = basin_df.index.to_list()
        random.shuffle(idx)
        partitions = [basin_df.iloc[idx[i :: args.n_partitions]] for i in range(args.n_partitions)]
        partition_shps = [np.array(partition.geometry.values) for partition in partitions]

        # Find which caravan gauges are assigned to each partition
        caravan_partitions = allocate_caravan(args.caravan_path, partitions)

        # Count how many of each thing of interest
        counts = {
            "continent": count_groups(partitions, "continent", CONTINENT_INDEX),
            "clim_zone": count_groups(partitions, "clz_cl_smj", CLIMATE_ZONE_INDEX),
            "coast": count_groups(partitions, "COAST", [0, 1]),
            "caravan": np.array([len(p) for p in caravan_partitions]),
            "ks": [count_within(ks_points, partition) for partition in partition_shps],
            "gff": [count_within(tile_points, partition) for partition in partition_shps],
        }

        # Determine how "good" the splits are with a heuristic
        score = (
            calc_distribution_score(*counts["continent"])
            + calc_distribution_score(*counts["clim_zone"])
            + calc_distribution_score(counts["caravan"])
            + calc_distribution_score(counts["ks"])
            + calc_distribution_score(counts["gff"])
        )
        if score < best_score:
            best = [partitions, caravan_partitions, counts]
            best_score = score
        print(
            f"Last: {score:7.5f}, Best: {best_score:7.5f}.   ({x+1:5d}/{args.trials:5d} complete)"
        )

    (partitions, caravan_partitions, counts) = best

    # Store to disk
    partition_folder: Path = args.out_folder / "partitions"
    partition_folder.mkdir(exist_ok=True)
    for i, partition in enumerate(partitions):
        partition.to_file(partition_folder / f"partition_{i}.gpkg", engine="pyogrio")
    for i, caravan_partition in enumerate(caravan_partitions):
        caravan_partition.to_csv(partition_folder / f"caravan_partition_{i}.txt", index=False)

    row_template = "{0:10s}: " + "|".join([f" {{{i+1}:7d}} " for i in range(args.n_partitions)])
    print("Continents")
    for i in range(N_CONTINENTS):
        print(row_template.format(str(i), *counts["continent"][i]))
    print("Climate Zones")
    for i in range(N_CLIMATE_ZONES):
        print(row_template.format(str(i), *counts["clim_zone"][i]))
    print("Coast vs Non-Coast")
    for i in range(2):
        print(row_template.format(str(i), *counts["coast"][i]))

    print("Caravan gauges")
    print(row_template.format("# gauges", *counts["caravan"]))

    print("Kurosiwo targets")
    print(row_template.format("# roi", *counts["ks"]))

    print("Generated floodmaps")
    print(row_template.format("# roi x T", *counts["gff"]))

    print("Distribution score: ", best_score)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
