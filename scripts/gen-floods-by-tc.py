import argparse
import datetime
from pathlib import Path
import sys

import geopandas
import numpy as np
import pandas
import rasterio
import shapely
import skimage
import tqdm
import xarray


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("dfo_path", type=Path)
    parser.add_argument("tc_path", type=Path)
    parser.add_argument("basin_flood_gpkg_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path, default=None)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--basin_level", type=int, default=8)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)


def overlap_1d(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))


def overlap_1d_np(min1, max1, min2, max2):
    return np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))


def main(args):
    dfo = geopandas.read_file(args.dfo_path)
    dfo.geometry = dfo.geometry.buffer(0)
    basin_floods = geopandas.read_file(args.basin_flood_gpkg_path, engine="pyogrio")
    tcs = pandas.read_csv(args.tc_path / "titleyetal2021_280storms.csv", converters={"BASIN": str})

    basin_path = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev{args.basin_level:02d}.shp"
    basins_df = geopandas.read_file(basin_path / basin_fname, engine="pyogrio")

    dfofmtstr = "%Y-%m-%d"
    tcfmtstr = "%Y%m%d%H"

    def dfo_to_ts(x):
        return datetime.datetime.strptime(x, dfofmtstr).timestamp()

    def tc_to_ts(x):
        return datetime.datetime.strptime(str(x), tcfmtstr).timestamp()

    dfo["BEGAN"] = dfo["BEGAN"].apply(dfo_to_ts)
    dfo["ENDED"] = dfo["ENDED"].apply(dfo_to_ts)
    tcs["FIRSTTRACKDATE"] = tcs["FIRSTTRACKDATE"].apply(tc_to_ts)
    tcs["LASTTRACKDATE"] = tcs["LASTTRACKDATE"].apply(tc_to_ts)
    tcs["PRELANDFALLPOINT"] = tcs["PRELANDFALLPOINT"].apply(tc_to_ts)
    tcs["key"] = tcs.YEAR.astype(str) + "_" + tcs.BASIN + "_" + tcs.STORMNAME

    # Drop anything before 2014
    dfo = dfo[dfo["BEGAN"] >= datetime.datetime(year=2014, month=1, day=1).timestamp()]

    time_overlap = overlap_1d_np(
        dfo["BEGAN"].values[:, None],
        dfo["ENDED"].values[:, None],
        tcs["FIRSTTRACKDATE"].values[None],
        tcs["LASTTRACKDATE"].values[None],
    )
    overlapping = list(zip(*(time_overlap > 0).nonzero()))
    coincide = []
    tcs_covered = []
    n_already_covered = 0
    for dfo_i, tc_i in tqdm.tqdm(overlapping):
        dfo_row = dfo.iloc[dfo_i]
        tc_row = tcs.iloc[tc_i]

        # Read tc footprint
        fname = f"{tc_row.key}_footprint.nc"
        nc = rasterio.open(args.tc_path / fname)
        footprint = nc.read()[0]
        footprint_np = skimage.measure.find_contours(footprint)[0]
        nc.transform.itransform(footprint_np)
        footprint_geom = shapely.Polygon(footprint_np)

        # Check if they overlap in space
        flood_spatial_overlap = shapely.intersection(dfo_row.geometry, footprint_geom)
        if flood_spatial_overlap.area > 0:
            tcs_covered.append(tc_row.key)

        # Check if overlap with existing locations (that we've already checked)
        basins_in_flood = basin_floods[basin_floods["ID"] == dfo_row.ID]
        already_covered = shapely.intersects(np.array(basins_in_flood.geometry), footprint_geom)

        if not any(already_covered) and flood_spatial_overlap.area > 0:
            if flood_spatial_overlap.geom_type == "MultiPolygon":
                flood_spatial_overlap = list(flood_spatial_overlap.geoms)[1]
            center = shapely.centroid(flood_spatial_overlap)
            basin_rows = basins_df[basins_df.geometry.contains(center)]
            basin_row = basin_rows.iloc[0]  # Perverse shape might make center not be on land.
            coincide.append(
                (dfo_row.ID, basin_row.HYBAS_ID, tc_row.YEAR, tc_row.BASIN, tc_row.STORMNAME)
            )
        if flood_spatial_overlap and any(already_covered):
            n_already_covered += 1

    all_tc = set(tcs.key.values)
    n_covered = len(set(tcs_covered))
    print(f"Out of {len(all_tc)}")
    print(f"Tropical cyclones overlapping DFO: {n_covered}")
    columns = ["FLOOD_ID", "HYBAS_ID", "BASIN", "YEAR", "STORMNAME"]
    out = pandas.DataFrame(coincide, columns=columns)
    out.to_csv(args.out_path, index=False)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
