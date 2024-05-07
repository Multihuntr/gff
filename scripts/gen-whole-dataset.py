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
    parser.add_argument("--flood_model_names", "-m", nargs="+", type=str, default=["vit"])

    return parser.parse_args(argv)


def do_ks_assignments(agg_labels_path: Path, basins: geopandas.GeoDataFrame):
    # Count how many of each
    metas = collections.defaultdict(lambda: [])
    sites = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    tiles = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    when_where = []
    d_fmt = "%Y-%m-%d"
    for path in agg_labels_path.glob("*-meta.json"):
        with open(path) as f:
            meta = json.load(f)

        k = meta["HYBAS_ID_4"]
        # Add site/tile counts
        continent = int(str(k)[0])
        metas[continent].append(meta)
        geoms = geopandas.read_file(path.with_name(meta["visit_tiles"]), engine="pyogrio")
        geoms_union = shapely.ops.unary_union(geoms.geometry)
        geoms_union_4326 = gff.util.convert_crs(geoms_union, geoms.crs, "EPSG:4326")
        overlap_counts = gff.generate.basins.zone_counts_shp(basins, geoms_union_4326)
        for clim_zone, count in overlap_counts.iterrows():
            sites[continent][clim_zone] += count.item()
            tiles[continent][clim_zone] += len(geoms)

        # Add meta data of where/when the site is so that we can ensure no overlap
        d = datetime.datetime.strptime(meta["info"]["sources"]["MS1"]["source_date"], d_fmt)
        footprint = shapely.convex_hull(shapely.union_all(geoms.geometry))
        footprint_4326 = gff.util.convert_crs(footprint, geoms.crs, "EPSG:4326")
        when_where.append((d.timestamp(), footprint_4326))
    return metas, sites, tiles, when_where


def calc_completion(
    exp_basin_distr,
    curr_flood_distr,
    curr_noflood_distr,
    max_sites,
    curr_sites,
    max_rows,
    curr_row,
    continent=None,
):
    if continent is not None:
        incl = [continent]
    else:
        incl = [i for i in curr_flood_distr]
    # Calculate distr overlap
    n_basin_groups = 0
    n_basins_counted = 0
    total_sites, total_basins = 0, 0
    diff_basins, diff_sites = 0, 0
    for c in incl:
        # Basins
        n_basin_groups += len(exp_basin_distr[c])
        for z in exp_basin_distr[c]:
            curr_count = curr_flood_distr[c][z] + 0.5 * curr_noflood_distr[c][z]
            if curr_count >= exp_basin_distr[c][z]:
                n_basins_counted += 1
            diff_basins += max(0, exp_basin_distr[c][z] - curr_count)
            total_basins += exp_basin_distr[c][z]

        # Sites
        total_sites += max_sites[c]
        n_sites = sum([1 if meta["flooding"] else 0.5 for meta in curr_sites[c]])
        diff_sites += max(0, max_sites[c] - n_sites)
    basin_perc = 1 - diff_basins / total_basins
    site_perc = 1 - diff_sites / total_sites
    basin_groups_perc = n_basins_counted / n_basin_groups

    # Rows left
    row_perc = curr_row / max_rows

    return basin_perc, site_perc, basin_groups_perc, row_perc


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
    exp_cont_sites, exp_basin_distr = gff.generate.basins.mk_expected_distribution(
        flood_distr, n_sites=args.sites
    )

    # Start by adding all kurosiwo labels to current distribution
    ks_metas, ks_sites, ks_tiles, ks_maps = do_ks_assignments(args.ks_agg_labels_path, basins08_df)
    cur_basin_flood_distr = {
        i: {j: ks_sites[i][j] for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    cur_basin_noflood_distr = {
        i: {j: 0 for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    current_distr_tiles = {
        i: {j: ks_tiles[i][j] for j in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
        for i in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    existing_maps = ks_maps.copy()
    added_sites = ks_metas.copy()
    exported_meta_fpaths = []

    # Load cached s1 index
    p = args.data_path / f"s1" / f"index.db"
    s1_index = sqlite3.connect(p, timeout=10)
    gff.generate.search.init_db(s1_index)

    # Then we search through basins, ordered, and add them if there's not "enough"
    torch.set_grad_enabled(False)
    flood_model_names = args.flood_model_names
    flood_models = [
        gff.generate.floodmaps.model_runner(name, args.data_path) for name in flood_model_names
    ]
    # Filter by continent
    basin_floods["continent"] = basin_floods.HYBAS_ID.astype(str).str[0].astype(int)
    if args.continent is not None:
        basin_floods = basin_floods[basin_floods["continent"] == args.continent]

    # BIG MAIN LOOP
    completion_columns = [
        "Basins x floods covering climate zones",
        "Sites created",
        "Climate zones with enough basins",
        "Rows searched",
    ]
    print(" | ".join(completion_columns))
    for i, (_, basin_row) in enumerate(basin_floods.iterrows()):
        # Calculate how much has been completed through the search
        perc = _, _, basin_group_perc, _ = calc_completion(
            exp_basin_distr,
            cur_basin_flood_distr,
            cur_basin_noflood_distr,
            exp_cont_sites,
            added_sites,
            len(basin_floods),
            i,
            args.continent,
        )
        fmt_str = " | ".join([f"{{{i}:4.0%}}" for i in range(len(perc))])
        complete_str = fmt_str.format(*perc)
        # Exit if we've added enough
        if basin_group_perc >= 1:
            print(complete_str)
            calc_completion(
                exp_basin_distr,
                cur_basin_flood_distr,
                cur_basin_noflood_distr,
                exp_cont_sites,
                added_sites,
                len(basin_floods),
                i,
                args.continent,
            )
            break

        key = f"{basin_row.ID}-{basin_row.HYBAS_ID}"
        continent = basin_row.continent

        # Search for S1 images; cache, static filter and choose approach type
        may_conflict = args.continent is None
        search_results = gff.generate.search.get_search_results(s1_index, basin_row, may_conflict)
        filtered_results, _ = gff.generate.search.filter_search_results(search_results, basin_row)
        asc = filtered_results["asc"]
        desc = filtered_results["desc"]
        if len(asc) > 0:
            approach_results = asc
        elif len(desc) > 0:
            approach_results = desc
        else:
            msg = f"{len(search_results)} | ({len(asc)}, {len(desc)})"
            print(f"{complete_str} | {key}: Not enough S1 images available. {msg}")
            continue

        # Identify viable image tuplets by overlap
        tuplets = gff.generate.util.find_intersecting_tuplets(
            approach_results, basin_row, n_img=3, min_size=0.1
        )
        if len(tuplets) == 0:
            print(f"{complete_str} | {key}: No S1 images overlap.")
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
            print(f"{complete_str} | {key}: All S1 image tuplets are too close to existing ones.")
            continue

        # Check if we believe this site will give us level 8 basins we want.
        wanted_tuplets = []
        for tuplet in safe_tuplets:
            # NOTE: This is a for loop because different tuplets overlap different areas
            counts = gff.generate.basins.estimate_zone_counts(
                basins08_df, approach_results, safe_tuplets[0]
            )
            wanted_zones = 0
            for clim_zone, count in counts.iterrows():
                expected_count = exp_basin_distr[continent][clim_zone]
                curr_count = cur_basin_flood_distr[continent][clim_zone]
                if (curr_count + count.item()) < expected_count + 3:
                    wanted_zones += 1
            if len(counts) > 0 and (wanted_zones / len(counts)) > 0.5:
                wanted_tuplets.append(tuplet)
        if len(wanted_tuplets) == 0:
            print(f"{complete_str} | {key}: Already enough sites for this continent/climate zone")
            continue

        # Generate floodmaps for this basin_row
        for name, flood_model in zip(flood_model_names, flood_models):
            print(f"{complete_str} | {key}: Generating {name} floodmaps")
            # Ensure vit maps are created
            vit_meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(
                args.data_path, key, "vit"
            )

            if len(vit_meta_fpaths) == 0:
                if rivers_df is None:
                    rivers_df = gff.generate.floodmaps.load_rivers(args.hydroatlas_path)

                gff.generate.floodmaps.create_flood_maps(
                    args.data_path,
                    approach_results,
                    basin_row,
                    rivers_df,
                    flood_models[flood_model_names.index("vit")],
                    export_s1=args.export_s1,
                    floodmap_folder="vit",
                    search_idxs=safe_tuplets,
                )

                vit_meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(
                    args.data_path, key, "vit"
                )
                for meta_fpath in vit_meta_fpaths:
                    gff.generate.floodmaps.remove_tiles_outside(meta_fpath, basins04_df)

            # Now that vit floodmaps are ensured, generate another model's floodmaps
            # following the tiles generated by vit.
            meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(args.data_path, key, name)
            if len(meta_fpaths) == 0:
                if rivers_df is None:
                    rivers_df = gff.generate.floodmaps.load_rivers(args.hydroatlas_path)

                tiles_to_use = [
                    gff.generate.floodmaps.floodmap_tiles_from_meta(fpath)
                    for fpath in vit_meta_fpaths
                ]

                gff.generate.floodmaps.create_flood_maps(
                    args.data_path,
                    approach_results,
                    basin_row,
                    rivers_df,
                    flood_model,
                    export_s1=False,
                    floodmap_folder=name,
                    search_idxs=safe_tuplets,
                    prescribed_tiles=tiles_to_use,
                )
                meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(
                    args.data_path, key, name
                )
                for meta_fpath in meta_fpaths:
                    gff.generate.floodmaps.remove_tiles_outside(meta_fpath, basins04_df)
            else:
                print(f"    {name} floodmaps already exist.")

        # Each call to create_flood_maps can generate multiple flood maps.
        meta_fpaths = gff.generate.floodmaps.floodmap_meta_fpaths(args.data_path, key, "vit")
        exported_meta_fpaths.extend(meta_fpaths)
        metas = gff.generate.floodmaps.load_metas(args.data_path, key, "vit")
        folder = args.data_path / "floodmaps" / "vit"
        for meta in metas:
            added_sites[continent].append(meta)
            geoms = geopandas.read_file(folder / meta["visit_tiles"], engine="pyogrio")
            hull = shapely.convex_hull(shapely.union_all(geoms.geometry))
            ts = datetime.datetime.fromisoformat(meta["post_date"]).timestamp()
            existing_maps.append((ts, hull))

            basin_counts = gff.generate.basins.zone_counts_shp(
                basins08_df, shapely.ops.unary_union(geoms)
            )
            for clim_zone, count in basin_counts.iterrows():
                if meta["flooding"]:
                    cur_basin_flood_distr[continent][clim_zone] += count.item()
                else:
                    cur_basin_noflood_distr[continent][clim_zone] += count.item()

    if args.continent is None:
        out_txt = args.data_path / f"exported_metas_{args.continent}.txt"
    else:
        out_txt = args.data_path / f"exported_metas.txt"

    exported_meta_fnames = [fpath.name for fpath in exported_meta_fpaths]
    pandas.DataFrame(exported_meta_fnames).to_csv(out_txt, index=False, header=False)
    print("All done!")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
