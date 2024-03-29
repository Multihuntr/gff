import argparse
from pathlib import Path
import random
import sys

import geopandas
import numpy as np
import pandas
import shapely


def parse_args(argv):
    parser = argparse.ArgumentParser("Split basins into partitions")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("basin_level", type=int)
    parser.add_argument("caravan_path", type=Path)
    parser.add_argument("out_folder", type=Path)
    parser.add_argument("--n_partitions", type=int, default=6)
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


def calc_distribution_score(*counts: list[np.ndarray]):
    scores = []
    for c in counts:
        scale = c.sum()
        score = np.std(c / scale)
        scores.append(score)
    return np.mean(scores)


N_CONTINENTS = 9
N_CLIMATE_ZONES = 18


def main(args):
    basin_dir = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_path = basin_dir / f"BasinATLAS_v10_lev{args.lvl:02d}.shp"
    basin_df = geopandas.read_file(basin_path, engine="pyogrio")

    best_score = 100
    best = None
    for x in range(args.trials):
        # Allocate randomly: continent, climate zone, coast/non, ???
        partitions: list[list[geopandas.GeoSeries]] = [[] for i in range(args.n_partitions)]
        continent_counts = np.zeros((N_CONTINENTS, args.n_partitions), dtype=np.int32)
        clim_zone_counts = np.zeros((N_CLIMATE_ZONES, args.n_partitions), dtype=np.int32)
        coast_counts = np.zeros((2, args.n_partitions), dtype=np.int32)
        idx = basin_df.index.to_list()
        random.shuffle(idx)
        for i in idx:
            row = basin_df.iloc[i]
            p = np.argmin([len(partition) for partition in partitions])
            partitions[p].append(row)
            c = int(str(row.HYBAS_ID)[:1]) - 1
            z = int(row.clz_cl_smj) - 1
            co = int(bool(row.COAST))
            continent_counts[c, p] += 1
            clim_zone_counts[z, p] += 1
            coast_counts[co, p] += 1

        # Compile rows to DF and save to disk
        partition_folder: Path = args.out_folder / "partitions"
        partition_folder.mkdir(exist_ok=True)
        partition_gdfs = []
        for i, partition in enumerate(partitions):
            partition_df = pandas.DataFrame(partition)
            partition_gdf = geopandas.GeoDataFrame(
                partition_df, geometry="geometry", crs="EPSG:4326"
            )
            partition_gdfs.append(partition_gdf)

        # Discover how caravan fits into these partitions
        caravan_partitions = allocate_caravan(args.caravan_path, partition_gdfs)
        caravan_counts = np.array([len(p) for p in caravan_partitions])

        # Determine how "good" the splits are with a heuristic
        score = (
            calc_distribution_score(*continent_counts)
            + calc_distribution_score(*clim_zone_counts)
            + calc_distribution_score(caravan_counts)
        ) / 3
        if score < best_score:
            best = [
                partitions,
                caravan_partitions,
                continent_counts,
                clim_zone_counts,
                coast_counts,
                caravan_counts,
            ]
            best_score = score
        print(
            f"Last: {score:7.5f}, Best: {best_score:7.5f}.   ({x+1:5d}/{args.trials:5d} complete)"
        )

    (
        partitions,
        caravan_partitions,
        continent_counts,
        clim_zone_counts,
        coast_counts,
        caravan_counts,
    ) = best

    # Store to disk
    for i, partition in enumerate(partitions):
        partition_gdf.to_file(partition_folder / f"partition_{i}.gpkg", engine="pyogrio")
    for i, caravan_partition in enumerate(caravan_partitions):
        caravan_partition.to_csv(partition_folder / f"caravan_partition_{i}.txt", index=False)

    row_template = "{0:10s}: " + "|".join([f" {{{i+1}:7d}} " for i in range(args.n_partitions)])
    print("Continents")
    for i in range(N_CONTINENTS):
        print(row_template.format(str(i), *continent_counts[i]))
    print("Climate Zones")
    for i in range(N_CLIMATE_ZONES):
        print(row_template.format(str(i), *clim_zone_counts[i]))
    print("Coast vs Non-Coast")
    for i in range(2):
        print(row_template.format(str(i), *coast_counts[i]))

    print("Caravan gauges")
    print(row_template.format("# gauges", *caravan_counts))

    print("Distribution score: ", best_score)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
