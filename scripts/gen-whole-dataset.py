import argparse
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import pandas
import shapely

import gff.generate.basins
import gff.util

def parse_args(argv):
    parser = argparse.ArgumentParser("Generate the whole dataset, from data sources to floodmaps")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("dfo_path", type=Path)
    parser.add_argument("tcs_path", type=Path)
    parser.add_argument("caravan_path", type=Path)
    parser.add_argument("ks_agg_labels_path", type=Path)
    parser.add_argument("desired_distribution_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)

N_CONTINENTS = 9
N_CLIMATE_ZONES = 18
CONTINENT_INDEX = list(range(1, N_CONTINENTS + 1))
CONTINENT_NAMES = {
    1: 'Africa',
    2: 'Europe and middle east',
    3: 'Russia',
    4: 'Asia',
    5: 'Oceania, pacific islands and south-east Asia',
    6: 'South America',
    7: 'North America 1',
    8: 'North America 2',
    9: 'Greenland',
}
CLIMATE_ZONE_INDEX = list(range(1, N_CLIMATE_ZONES + 1))

def ks_center_of(p: Path):
    footprint = gff.util.image_footprint(p)
    footprint = gff.util.convert_crs(footprint, "EPSG:3857", "EPSG:4326")
    return shapely.centroid(footprint)

def get_points_by_basin(basins: geopandas.GeoDataFrame, ks_points):
    basin_geoms = np.array(basins.geometry.values)[:, None]
    ks_points = ks_points[None]
    contains = shapely.contains(basin_geoms, ks_points)
    return [basins[c] for c in contains]

def main(args):
    dfo = geopandas.read_file(args.dfo_path)
    dfo.geometry = dfo.geometry.buffer(0)
    tcs = pandas.read_csv(args.tc_path / "titleyetal2021_280storms.csv", converters={"BASIN": str})

    basin_path = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    basins08_df = geopandas.read_file(basin_path / basin08_fname, engine="pyogrio")
    basin04_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev04.shp"
    basins04_df = geopandas.read_file(basin_path / basin04_fname, engine="pyogrio")
    basin01_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev01.shp"
    basins01_df = geopandas.read_file(basin_path / basin01_fname, engine="pyogrio")

    tcs_basins = gff.generate.basins.tcs_basins(dfo, basins08_df, tcs, args.tcs_path)
    coastal_basins = gff.generate.basins.coastal(basins08_df)
    basins_by_impact = gff.generate.basins.basins_by_impact(dfo, coastal_basins)

    basin_floods = pandas.merge(
        basins_by_impact,
        tcs_basins,
        how="inner",
        left_on=["ID", "HYBAS_ID"],
        right_on=["FLOOD_ID", "HYBAS_ID"],
    )
    with open(args.desired_distribution_path) as f:
        desired_distribution = json.load(f)
    ks_site_points = np.array([ks_center_of(path) for path in args.ks_agg_labels_path.glob("*.tif")])
    ks_by_continent = get_points_by_basin(basins01_df, ks_site_points)
    basins_by_impact["continent"] = basins_by_impact.HYBAS_ID.astype(str).str[0].astype(int)
    for i in CONTINENT_INDEX:
        cont_basins = basins_by_impact[basins_by_impact['continent'] == i]
        ks_by_lv04 = get_points_by_basin(basins04_df, ks_by_continent[i])
        clz_counts = gff.util.count_group(basins04_df, 'clz_cl_smj', CLIMATE_ZONE_INDEX)
        for j in CLIMATE_ZONE_INDEX:
            clz_basins = cont_basins[cont_basins['clz_cl_smj'] == j]
            for basin_row in clz_basins:




    # continent_counts = {i: 0 for i in CONTINENT_INDEX}
    # climate_zone_counts = {i: {j: 0 for j in CLIMATE_ZONE_INDEX} for i in CONTINENT_INDEX}
    # for basin_row in basins_by_impact.iterrows():
    #     continent = str(basin_row.HYBAS_ID)[0]
    #     clim_zone = basin_row.clz_cl_smj
    #     if continent_counts[continent] > desired_continent_count[continent]:
    #         continue
    #     if (climate_zone_counts[continent][clim_zone]) > 0.5


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
