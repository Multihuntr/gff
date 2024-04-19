import argparse
import collections
import datetime
import json
from pathlib import Path
import sqlite3
import sys
import warnings

import geopandas
import numpy as np
import pandas
import shapely
import torch

import gff.constants
import gff.data_sources
import gff.generate.basins
import gff.generate.floodmaps
import gff.generate.search
import gff.generate.util
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Generate the whole dataset, from data sources to floodmaps")

    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("dfo_path", type=Path)
    parser.add_argument("tcs_path", type=Path)
    # parser.add_argument("caravan_path", type=Path)
    parser.add_argument("ks_agg_labels_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)
    parser.add_argument("--export_s1", action="store_true")
    parser.add_argument("--sites", type=int, default=200)
    parser.add_argument(
        "--continent",
        type=int,
        default=None,
        help="If provided, will only try to generate floodmaps for this continent",
    )
    parser.add_argument(
        "--clim_zone",
        type=int,
        default=None,
        help="If provided, will only try to generate floodmaps for this climate zone",
    )

    return parser.parse_args(argv)


def get_points_by_basin(basins: geopandas.GeoDataFrame, ks_points):
    # TODO: remove
    basin_geoms = np.array(basins.geometry.values)[:, None]
    ks_points = ks_points[None]
    contains = shapely.contains(basin_geoms, ks_points)
    return [basins[c] for c in contains]


def mk_expected(
    basin_distr: dict, flood_distr: np.ndarray, n_sites: int, expect_floods: bool = True
):
    """
    Calculates how many sites we expect (min) for each continent/climate zone

    returns { <continent>: {<climate_zone>: int} }
    """
    if expect_floods:
        ratios = flood_distr / flood_distr.sum()
    else:
        total = 0
        for continent in gff.constants.HYDROATLAS_CONTINENT_NAMES:
            if continent in basin_distr:
                total += basin_distr[continent]["total"]
        cont_ratio = n_sites / total
    result = collections.defaultdict(lambda: {})
    for i, continent in enumerate(gff.constants.HYDROATLAS_CONTINENT_NAMES):
        if expect_floods:
            n_cont = n_sites * ratios[i]
        else:
            n_cont = basin_distr[continent]["total"] * cont_ratio
        for clim_zone in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES:
            if continent in basin_distr and clim_zone in basin_distr[continent]["zones"]:
                proportion = (
                    basin_distr[continent]["zones"][clim_zone] / basin_distr[continent]["total"]
                )
                result[continent][clim_zone] = int(proportion * n_cont)
    return result


def do_ks_assignments(agg_labels_path: Path, lvl4_basins: geopandas.GeoDataFrame):
    # Count how many of each
    meta_paths = []
    sites = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    tiles = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    site_dt_geoms = []
    d_fmt = "%Y-%m-%d"
    for path in agg_labels_path.glob("*-meta.json"):
        meta_paths.append(path)
        with open(path) as f:
            meta = json.load(f)

        k = meta["HYBAS_ID_4"]
        continent = int(str(k)[0])
        lvl4_row = lvl4_basins[lvl4_basins["HYBAS_ID"] == k].iloc[0]
        clim_zone = lvl4_row["clz_cl_smj"]
        sites[continent][clim_zone] += 1
        geoms = geopandas.read_file(path.with_name(meta["visit_tiles"]), engine="pyogrio")
        tiles[continent][clim_zone] += len(geoms)
        d = datetime.datetime.strptime(meta["info"]["sources"]["MS1"]["source_date"], d_fmt)
        footprint = shapely.convex_hull(shapely.union_all(geoms.geometry))
        footprint_4326 = gff.util.convert_crs(footprint, geoms.crs, "EPSG:4326")
        site_dt_geoms.append((k, d.timestamp(), footprint_4326))
    return meta_paths, sites, tiles, site_dt_geoms


def get_this_lvl4(shp: shapely.Geometry, lvl4_basins: geopandas.GeoDataFrame):
    with warnings.catch_warnings(action="ignore"):
        # It complains because we're doing an area calculation in EPSG:4326,
        # a non-area-preserving CRS. But, I don't care about the exact area.
        which_basin = np.argmax(lvl4_basins.geometry.intersection(shp).area)
    return lvl4_basins.iloc[which_basin]


def too_close(
    existing: list[tuple[str, int, shapely.Geometry]],
    to_check: tuple[str, int, shapely.Geometry],
    time_threshold: int = 3600 * 24 * 20,
    distance_threshold: float = 35 * 0.1,
):
    """Checks if to_check is too close in time and space to any of existing"""
    check_basin, check_ts, check_shp = to_check
    for basin, ts, shp in existing:
        # If the area to be searched is already covered by a site, don't bother with it.
        # Even if it could theoretically provide more (by happenstance choosing different S1 images)
        # it's just a low-value location.
        if shapely.intersection(check_shp, shp).area > 0:
            return True
        close_in_time = ts > check_ts - time_threshold and ts < check_ts + time_threshold
        if not close_in_time:
            continue

        # NOTE: By using EPSG:4326, we are (kind of) measuring distance in ERA5-Land pixels
        #   This is important because we only care about distance so that ERA5-Land doesn't
        #   overlap across sites and leak information
        points = shapely.ops.nearest_points(check_shp, shp)
        dist = shapely.distance(*points)
        if dist < distance_threshold:
            return True
    return False


def remove_too_close(
    existing: list[tuple[str, int, shapely.Geometry]],
    search_results: list[dict],
    hybas_id: int,
    time_threshold: int = 3600 * 24 * 20,
    distance_threshold: float = 40 * 0.1,
    n_img: int = 3,
):
    out = []
    for i in list(range(n_img - 1, len(search_results))):
        # Only include S1 images which have overlap with others and for which searching
        # would not result in a search area too close in time and space to existing areas.
        # NOTE: This couples with how generation uses search results.
        search_idx = gff.generate.util.find_intersection_results(search_results, i, n_img)
        if search_idx is not None:
            results = [search_results[j] for j in search_idx]
            shp_to_gen = gff.generate.util.search_result_footprint_intersection(results)
            first_res = search_results[i][0]
            ts = datetime.datetime.fromisoformat(first_res["properties"]["startTime"]).timestamp()
            map_loc = (hybas_id, ts, shp_to_gen)
            if not too_close(existing, map_loc, time_threshold, distance_threshold):
                out.append(search_results[i])
    return out


def main(args):
    # Load the various geometry datasources
    dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    tc_csv_path = args.tcs_path / "titleyetal2021_280storms.csv"
    tcs = pandas.read_csv(tc_csv_path, converters={"BASIN": str})

    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    basins08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    basins08_df = geopandas.read_file(basin_path / basins08_fname, engine="pyogrio")
    basins04_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev04.shp"
    basins04_df = geopandas.read_file(basin_path / basins04_fname, engine="pyogrio")
    basins01_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev01.shp"
    basins01_df = geopandas.read_file(basin_path / basins01_fname, engine="pyogrio")

    rivers_df = gff.generate.floodmaps.load_rivers(args.hydroatlas_path)

    # Load cached s1 indexes
    p = args.data_path / f"s1" / f"index.db"
    s1_index = sqlite3.connect(p, timeout=10)
    gff.generate.search.init_db(s1_index)

    # Create potential basins to search
    tc_cache = args.data_path / "tc_basins.csv"
    tcs_basins = gff.generate.basins.tcs_basins(dfo, basins08_df, tcs, args.tcs_path, tc_cache)
    coastal_basins = gff.generate.basins.coastal(basins08_df)
    basins_by_impact = gff.generate.basins.basins_by_impact(coastal_basins, dfo)

    # Determine expected distribution of sites which we will try to match
    basin_distr = gff.generate.basins.basin_distribution(args.data_path, basins04_df)
    flood_distr = gff.generate.basins.flood_distribution(dfo, basins01_df)
    expected_distr = mk_expected(basin_distr, flood_distr, n_sites=args.sites)

    # Start by adding all kurosiwo labels to current distribution
    ks_metas, ks_sites, ks_tiles, ks_maps = do_ks_assignments(args.ks_agg_labels_path, basins04_df)
    current_distr_sites = {
        i: {j: ks_sites[i][j] for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    w_negative = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    just_negative = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    current_distr_tiles = {
        i: {j: ks_tiles[i][j] for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    existing_maps = ks_maps.copy()
    added_sites = ks_metas.copy()

    # Then we search through basins, ordered, and add them if there's not "enough"
    torch.set_grad_enabled(False)
    flood_model_names = ["vit", "snunet", "vit+snunet"]
    flood_models = [
        gff.generate.floodmaps.model_runner(name, args.data_path) for name in flood_model_names
    ]
    basins_by_impact["continent"] = basins_by_impact.HYBAS_ID.astype(str).str[0].astype(int)
    if args.continent is not None:
        basins_by_impact = basins_by_impact[basins_by_impact["continent"] == args.continent]
    for i, basin_row in basins_by_impact.iterrows():
        if len(added_sites) >= args.sites:
            break

        # Check if we even want this one
        continent = basin_row.continent
        basin04_row = get_this_lvl4(basin_row.geometry, basins04_df)
        clim_zone = basin04_row["clz_cl_smj"]
        expected_count = expected_distr[continent][clim_zone]
        curr_count = current_distr_sites[continent][clim_zone]
        if args.clim_zone is not None and clim_zone != args.clim_zone:
            continue
        if curr_count >= expected_count + 3:
            # That's enough
            continue

        # Search for S1 images; cache, filter and choose approach type
        search_results = gff.generate.search.get_search_results(s1_index, basin_row)
        filtered_results, counts = gff.generate.search.filter_search_results(
            search_results, basin_row
        )
        asc = remove_too_close(existing_maps, filtered_results["asc"], basin04_row.HYBAS_ID)
        desc = remove_too_close(existing_maps, filtered_results["desc"], basin04_row.HYBAS_ID)
        key = f"{basin_row.ID}-{basin_row.HYBAS_ID}"
        print(f"Search results for {key}: {len(search_results)} | ({len(asc)}, {len(desc)})")
        if len(asc) > 0:
            approach_results = asc
        elif len(desc) > 0:
            approach_results = desc
        else:
            continue

        # Generate floodmaps for this basin_row
        for name, flood_model in zip(flood_model_names, flood_models):
            if not gff.generate.floodmaps.check_floodmap_exists(args.data_path, key, name):
                gff.generate.floodmaps.create_flood_maps(
                    args.data_path,
                    approach_results,
                    basin_row,
                    rivers_df,
                    flood_model,
                    export_s1=args.export_s1,
                    floodmap_folder=name,
                )
            else:
                print(f"{key} {name} floodmaps already exist.")

        # Each call to create_flood_maps can generate multiple flood maps.
        # We track three cases:
        #   1. Only one floodmap; has flooding.
        #   2. Multiple floodmaps; one with flooding, n without.
        #   3. Any number of floodmaps without any flooding.
        # Only #1 contributes to the distribution calculation. The others are FYIs.
        # NOTE: Using hoisted name to only count once; thus whichever was last in the list
        metas = gff.generate.floodmaps.load_metas(args.data_path, key, name)
        folder = args.data_path / "floodmaps" / name
        for meta in metas:
            geoms = geopandas.read_file(folder / meta["visit_tiles"], engine="pyogrio")
            hull = shapely.convex_hull(shapely.union_all(geoms.geometry))
            ts = datetime.datetime.fromisoformat(meta["post_date"]).timestamp()
            existing_maps.append((basin04_row["HYBAS_ID"], ts, hull))

        if any([meta["flooding"] for meta in metas]):
            current_distr_sites[continent][clim_zone] += 1
            geoms = geopandas.read_file(folder / metas[-1]["visit_tiles"], engine="pyogrio")
            current_distr_tiles[continent][clim_zone] += len(geoms)
            if len(metas) > 1:
                w_negative[continent][clim_zone] += 1
        else:
            just_negative[continent][clim_zone] += 1

    distributions = {
        "basins": basin_distr,
        "expected": expected_distr,
        "created": current_distr_sites,
        "with_neg": w_negative,
        "pure_neg": just_negative,
        "created_tiles": current_distr_tiles,
    }
    with open(args.data_path, "w") as f:
        json.dump(distributions, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
