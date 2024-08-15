import argparse
import datetime
import json
from pathlib import Path
import sys

import geopandas
import numpy as np
import pandas
import shapely
import tqdm

import gff.data_sources
import gff.generate.basins


def parse_args(argv):
    parser = argparse.ArgumentParser("Count stats using Tropical Cyclones")

    parser.add_argument("data_path", type=Path)
    parser.add_argument("dfo_path", type=Path)
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("tcs_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)


def main(args):
    # dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    tc_csv_path = args.tcs_path / "titleyetal2021_280storms.csv"
    tcs = pandas.read_csv(tc_csv_path, converters={"BASIN": str})

    # basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    # basins08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    # basins08_df = geopandas.read_file(
    #     basin_path / basins08_fname, use_arrow=True, engine="pyogrio"
    # )
    # gff.generate.basins.tcs_basins(dfo, basins08_df, tcs, args.tcs_path)

    # print("Tropical cyclones directly used")
    # tcs_simple = pandas.read_csv(args.tcs_path / "tc_basins.csv")
    # for i, tc_row in tcs_simple.iterrows():
    #     check_str = f"{tc_row.FLOOD_ID}-{tc_row.HYBAS_ID}-*-meta.json"
    #     n = len(list((args.data_path / "release" / "rois").glob(check_str)))
    #     if n > 0:
    #         print(check_str)

    print("Tropical cyclones covered")
    # Load all ROI flood dates and footprints
    roi_tss = []
    roi_footprints = []
    roi_folder = args.data_path / "rois"
    meta_fpaths = list(roi_folder.glob("*-meta.json"))
    for meta_fpath in tqdm.tqdm(meta_fpaths):
        with open(meta_fpath) as f:
            meta = json.load(f)

        tile_path = roi_folder / meta["visit_tiles"]
        tiles = geopandas.read_file(tile_path, use_arrow=True, engine="pyogrio")
        footprint = shapely.unary_union(tiles.geometry)
        roi_footprints.append(footprint)
        roi_ts = datetime.datetime.fromisoformat(meta["post_date"]).timestamp()
        roi_tss.append(roi_ts)
    roi_tss = np.array(roi_tss)[None]
    roi_footprints = np.array(roi_footprints)[None]
    exp_area = 0.8 * shapely.area(roi_footprints)

    # Get the tropical cyclones, start (ts), end (ts) and footprints as numpy arrays
    tcfmtstr = "%Y%m%d%H"

    def tc_to_ts(x):
        return datetime.datetime.strptime(str(x), tcfmtstr).timestamp()

    def tc_to_geom(key):
        return gff.generate.basins.tc_footprint(args.tcs_path, key)

    tcs["FIRSTTRACKDATE"] = tcs["FIRSTTRACKDATE"].apply(tc_to_ts)
    tcs["LASTTRACKDATE"] = tcs["LASTTRACKDATE"].apply(tc_to_ts)
    tcs["PRELANDFALLPOINT"] = tcs["PRELANDFALLPOINT"].apply(tc_to_ts)
    tcs["key"] = tcs.YEAR.astype(str) + "_" + tcs.BASIN + "_" + tcs.STORMNAME

    tc_start = tcs["FIRSTTRACKDATE"].values[:, None]
    tc_end = tcs["LASTTRACKDATE"].values[:, None]
    tc_footprints = tcs["key"].apply(tc_to_geom).values[:, None]

    in_time = (tc_start >= roi_tss) & (roi_tss <= tc_end)
    in_space = shapely.area(shapely.intersection(roi_footprints, tc_footprints)) >= exp_area

    used_tcs = (in_time & in_space).sum(axis=1) >= 1
    print("Tropical cyclones which cover a ROI: ", used_tcs.sum())


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
