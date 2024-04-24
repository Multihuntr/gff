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

    return parser.parse_args(argv)


def mk_expected(flood_distr: np.ndarray, n_sites: int, est_basins_per_site: int = 15):
    """
    Calculates how many sites we expect (min) for each continent/climate zone

    returns { <continent>: {<climate_zone>: int} }
    """
    total = 0
    for continent in gff.constants.HYDROATLAS_CONTINENT_NAMES:
        if continent in flood_distr:
            total += flood_distr[continent]["total"]
    cont_ratio = n_sites * est_basins_per_site / total
    result = collections.defaultdict(lambda: {})
    for i, continent in enumerate(gff.constants.HYDROATLAS_CONTINENT_NAMES):
        n_cont = flood_distr[continent]["total"] * cont_ratio
        for clim_zone in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES:
            if (
                continent in flood_distr
                and clim_zone in flood_distr[continent]["zones"]
                and flood_distr[continent]["total"] > 0
            ):
                proportion = (
                    flood_distr[continent]["zones"][clim_zone] / flood_distr[continent]["total"]
                )
                result[continent][clim_zone] = int(proportion * n_cont)
    return result


def do_ks_assignments(agg_labels_path: Path, basins: geopandas.GeoDataFrame):
    # Count how many of each
    meta_paths = []
    sites = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    tiles = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    when_where = []
    d_fmt = "%Y-%m-%d"
    for path in agg_labels_path.glob("*-meta.json"):
        meta_paths.append(path)
        with open(path) as f:
            meta = json.load(f)

        k = meta["HYBAS_ID_4"]
        # Add site/tile counts
        continent = int(str(k)[0])
        geoms = geopandas.read_file(path.with_name(meta["visit_tiles"]), engine="pyogrio")
        geoms_union = shapely.ops.unary_union(geoms.geometry)
        geoms_union_4326 = gff.util.convert_crs(geoms_union, geoms.crs, "EPSG:4326")
        overlap_counts = zone_counts_shp(basins, geoms_union_4326)
        for clim_zone, count in overlap_counts.iterrows():
            sites[continent][clim_zone] += count.item()
            tiles[continent][clim_zone] += len(geoms)

        # Add meta data of where/when the site is so that we can ensure no overlap
        d = datetime.datetime.strptime(meta["info"]["sources"]["MS1"]["source_date"], d_fmt)
        footprint = shapely.convex_hull(shapely.union_all(geoms.geometry))
        footprint_4326 = gff.util.convert_crs(footprint, geoms.crs, "EPSG:4326")
        when_where.append((d.timestamp(), footprint_4326))
    return meta_paths, sites, tiles, when_where


def get_this_lvl4(shp: shapely.Geometry, lvl4_basins: geopandas.GeoDataFrame):
    with warnings.catch_warnings(action="ignore"):
        # It complains because we're doing an area calculation in EPSG:4326,
        # a non-area-preserving CRS. But, I don't care about the exact area.
        which_basin = np.argmax(lvl4_basins.geometry.intersection(shp).area)
    return lvl4_basins.iloc[which_basin]


def zone_counts_shp(basins: geopandas.GeoDataFrame, shp: shapely.Geometry, threshold: float = 0.3):
    overlaps = gff.generate.basins.overlaps_at_least(basins.geometry, shp, threshold)
    overlap_basins = basins[overlaps]
    return overlap_basins[["HYBAS_ID", "clz_cl_smj"]].groupby("clz_cl_smj").count()


def estimate_zone_counts(
    basins: geopandas.GeoDataFrame,
    search_results: list[dict],
    search_idx: list[int],
    threshold: float = 0.3,
):
    results = [search_results[i] for i in search_idx]
    footprint = gff.generate.util.search_result_footprint_intersection(results)
    return zone_counts_shp(basins, footprint, threshold)


def main(args):
    # Load the various geometry datasources
    dfo = gff.data_sources.load_dfo(args.dfo_path, for_s1=True)
    tc_csv_path = args.tcs_path / "titleyetal2021_280storms.csv"
    tcs = pandas.read_csv(tc_csv_path, converters={"BASIN": str})

    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    basins08_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev08.shp"
    basins08_df = geopandas.read_file(
        basin_path / basins08_fname, use_arrow=True, engine="pyogrio"
    )
    basins04_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev04.shp"
    basins04_df = geopandas.read_file(
        basin_path / basins04_fname, use_arrow=True, engine="pyogrio"
    )
    # basins01_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev01.shp"
    # basins01_df = geopandas.read_file(basin_path / basins01_fname, engine="pyogrio")

    rivers_df = None  # Loading these is slow and doesn't need to happen early, so it's lazy

    # Create potential basins to search
    tc_cache = args.data_path / "tc_basins.gpkg"
    tcs_basins = gff.generate.basins.tcs_basins(dfo, basins08_df, tcs, args.tcs_path, tc_cache)
    coastal_basin_floods = gff.generate.basins.coastal_basin_x_floods(basins08_df, dfo)
    basin_floods = pandas.concat([coastal_basin_floods, tcs_basins])
    basin_floods = gff.generate.basins.by_impact(basin_floods)
    # NOTE: pandas complains about a fragmented dataframe. Not sure why, but it recommened this:
    basin_floods = basin_floods.copy()  # to de-fragment the dataframe.

    # Determine expected distribution of basins which we will try to match
    flood_cache = args.data_path / "flood_distribution.json"
    flood_distr = gff.generate.basins.flood_distribution(
        dfo, basins08_df, overlap_threshold=0.3, cache_path=flood_cache
    )
    expected_distr = mk_expected(flood_distr, n_sites=args.sites)

    # Start by adding all kurosiwo labels to current distribution
    ks_metas, ks_sites, ks_tiles, ks_maps = do_ks_assignments(args.ks_agg_labels_path, basins08_df)
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

    # Load cached s1 index
    p = args.data_path / f"s1" / f"index.db"
    s1_index = sqlite3.connect(p, timeout=10)
    gff.generate.search.init_db(s1_index)

    # Then we search through basins, ordered, and add them if there's not "enough"
    torch.set_grad_enabled(False)
    # flood_model_names = ["vit", "snunet", "vit+snunet"]
    flood_model_names = ["vit"]
    flood_models = [
        gff.generate.floodmaps.model_runner(name, args.data_path) for name in flood_model_names
    ]
    basin_floods["continent"] = basin_floods.HYBAS_ID.astype(str).str[0].astype(int)
    if args.continent is not None:
        basin_floods = basin_floods[basin_floods["continent"] == args.continent]
    for i, basin_row in basin_floods.iterrows():
        if len(added_sites) >= args.sites:
            break

        key = f"{basin_row.ID}-{basin_row.HYBAS_ID}"
        continent = basin_row.continent

        # Search for S1 images; cache, static filter and choose approach type
        search_results = gff.generate.search.get_search_results(s1_index, basin_row)
        filtered_results, _ = gff.generate.search.filter_search_results(search_results, basin_row)
        asc = filtered_results["asc"]
        desc = filtered_results["desc"]
        if len(asc) > 0:
            approach_results = asc
        elif len(desc) > 0:
            approach_results = desc
        else:
            msg = f"{len(search_results)} | ({len(asc)}, {len(desc)})"
            print(f"{key}: Not enough S1 images available. {msg}")
            continue

        # Identify viable image tuplets by overlap
        tuplets = gff.generate.util.find_intersecting_tuplets(approach_results, basin_row, n_img=3)
        if len(tuplets) == 0:
            print(f"{key}: No S1 images overlap.")
            continue

        # Remove tuplets that are too close to existing
        safe_tuplets = []
        closes = []
        for tuplet in tuplets:
            close = gff.generate.util.s1_too_close(existing_maps, approach_results, tuplet[-1])
            if close:
                closes.append((tuplet[-1], closes))
            else:
                safe_tuplets.append(tuplet)
        if len(safe_tuplets) == 0:
            print(f"{key}: All S1 image tuplets are too close to existing ones.")
            continue

        # Check if we believe this site will give us level 8 basins we want.
        wanted_tuplets = []
        for tuplet in safe_tuplets:
            # NOTE: This is necessary because different tuplets overlap different areas
            counts = estimate_zone_counts(basins08_df, approach_results, safe_tuplets[0])
            wanted_zones = 0
            for clim_zone, count in counts.iterrows():
                expected_count = expected_distr[continent][clim_zone]
                curr_count = current_distr_sites[continent][clim_zone]
                if (curr_count + count.item()) < expected_count + 3:
                    wanted_zones += 1
            if (wanted_zones / len(counts)) > 0.5:
                wanted_tuplets.append(tuplet)
        if len(wanted_tuplets) == 0:
            print(f"{key}: Already enough sites for this continent/climate zone")
            continue

        # Generate floodmaps for this basin_row
        for name, flood_model in zip(flood_model_names, flood_models):
            print(f"{key}: Generating {name} floodmaps")
            if not gff.generate.floodmaps.check_floodmap_exists(args.data_path, key, name):
                if rivers_df is None:
                    rivers_df = gff.generate.floodmaps.load_rivers(args.hydroatlas_path)

                gff.generate.floodmaps.create_flood_maps(
                    args.data_path,
                    approach_results,
                    basin_row,
                    rivers_df,
                    flood_model,
                    export_s1=args.export_s1,
                    floodmap_folder=name,
                    search_idxs=safe_tuplets,
                )

                meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(
                    args.data_path, key, name
                )
                for meta_fpath in meta_fpaths:
                    gff.generate.floodmaps.remove_tiles_outside(meta_fpath, basins04_df)
            else:
                print(f"    {name} floodmaps already exist.")

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
            existing_maps.append((ts, hull))

        if any([meta["flooding"] for meta in metas]):
            flood_meta = metas[-1]
            geoms = geopandas.read_file(folder / flood_meta["visit_tiles"], engine="pyogrio")
            overlap_counts = zone_counts_shp(basins08_df, shapely.ops.unary_union(geoms))
            for clim_zone, count in overlap_counts.iterrows():
                current_distr_sites[continent][clim_zone] += count.item()
                current_distr_tiles[continent][clim_zone] += len(geoms)
            if len(metas) > 1:
                w_negative[continent][clim_zone] += 1
        elif len(metas) > 0:
            just_negative[continent][clim_zone] += 1

    distributions = {
        "floods": flood_distr,
        "expected": expected_distr,
        "created": current_distr_sites,
        "with_neg": w_negative,
        "pure_neg": just_negative,
        "created_tiles": current_distr_tiles,
    }
    with open(args.data_path / "distributions.json", "w") as f:
        json.dump(distributions, f)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
