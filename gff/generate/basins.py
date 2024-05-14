import collections
import datetime
import json
from pathlib import Path

import geopandas
import numpy as np
import pandas
import rasterio
import shapely
import skimage
import tqdm


import gff.constants
import gff.generate.util


def basin_distribution(data_folder: Path, basins: geopandas.GeoDataFrame):
    """
    Calculates the distribution of basins per continent/climate zone

    returns {
        <continent_index>: {
            'total': int,
            'zones': {
                <climate_zone_index>: int,
            }
        }
    }
    """
    fpath = data_folder / "desired_distribution.json"
    if fpath.exists():
        with fpath.open() as f:
            return json.load(f)

    basins["continent"] = basins.HYBAS_ID.astype(str).str[0].astype(int)
    counts = (
        basins[["HYBAS_ID", "continent", "clz_cl_smj"]]
        .groupby(["continent", "clz_cl_smj"])
        .count()
    )
    result = collections.defaultdict(lambda: {"total": 0, "zones": {}})
    for (continent, climate_zone), row in counts.iterrows():
        result[continent]["total"] += row.item()
        result[continent]["zones"][climate_zone] = row.item()
    return result


def overlaps_at_least(a, b, threshold: float = 0.3):
    overlaps = shapely.area(shapely.intersection(a, b)) / shapely.area(a)
    return overlaps > threshold


def flood_distribution(
    dfo: geopandas.GeoDataFrame,
    basins: geopandas.GeoDataFrame,
    overlap_threshold: float = 0.3,
    cache_path: Path = None,
):
    """Calculates the number of floods per continent/climate zone"""
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            load_distr = json.load(f)
        # Keys have been typed to strings; return to ints
        out_distr = {}
        for k, v in load_distr.items():
            out_distr[int(k)] = {
                "total": v["total"],
                "zones": {int(kz): vz for kz, vz in v["zones"].items()},
            }
        return out_distr
    # Add continent column
    basins["continent"] = basins.HYBAS_ID.astype(str).str[0].astype(int)

    # Add n_floods column
    basin_geom = np.array(basins.geometry.values)[:, None]
    dfo_geom = np.array(dfo.geometry.values)[None, :]
    basins["n_floods"] = overlaps_at_least(basin_geom, dfo_geom, overlap_threshold).sum(axis=1)

    # Add up floods
    counts = (
        basins[["HYBAS_ID", "continent", "clz_cl_smj", "n_floods"]]
        .groupby(["continent", "clz_cl_smj"])
        .agg({"n_floods": "sum"})
    )
    # Store distribution as dictionary
    result = collections.defaultdict(lambda: {"total": 0, "zones": {}})
    for (continent, climate_zone), row in counts.iterrows():
        result[continent]["total"] += row.item()
        result[continent]["zones"][climate_zone] = row.item()
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(result, f)
    return result


def overlap_1d_np(min1, max1, min2, max2):
    return np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))


def tcs_basins(
    dfo: geopandas.GeoDataFrame,
    basins: geopandas.GeoDataFrame,
    tcs: geopandas.GeoDataFrame,
    tc_path: Path,
    out_fpath: Path,
):
    if out_fpath.exists():
        return geopandas.read_file(out_fpath)

    dfofmtstr = "%Y-%m-%d"
    tcfmtstr = "%Y%m%d%H"

    def dfo_to_ts(x):
        return datetime.datetime.strptime(x, dfofmtstr).timestamp()

    def tc_to_ts(x):
        return datetime.datetime.strptime(str(x), tcfmtstr).timestamp()

    dfo_orig = dfo.copy()
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
    for dfo_i, tc_i in tqdm.tqdm(overlapping):
        dfo_row = dfo.iloc[dfo_i]
        tc_row = tcs.iloc[tc_i]

        # Read tc footprint
        fname = f"{tc_row.key}_footprint.nc"
        nc = rasterio.open(tc_path / fname)
        footprint = nc.read()[0]
        footprint_np = skimage.measure.find_contours(footprint)[0]
        nc.transform.itransform(footprint_np)
        footprint_geom = shapely.Polygon(footprint_np)

        # Check if they overlap in space
        flood_spatial_overlap = shapely.intersection(dfo_row.geometry, footprint_geom)
        if flood_spatial_overlap.area > 0:
            tcs_covered.append(tc_row.key)

            # Select central basin
            if flood_spatial_overlap.geom_type == "MultiPolygon":
                flood_spatial_overlap = list(flood_spatial_overlap.geoms)[1]
            center = shapely.centroid(flood_spatial_overlap)
            basin_rows = basins[basins.geometry.contains(center)]
            basin_row = basin_rows.iloc[0]

            coincide.append(
                (dfo_row.ID, basin_row.HYBAS_ID, tc_row.YEAR, tc_row.BASIN, tc_row.STORMNAME)
            )

    all_tc = set(tcs.key.values)
    n_covered = len(set(tcs_covered))
    print(f"Out of {len(all_tc)}")
    print(f"Tropical cyclones overlapping DFO: {n_covered}")
    columns = ["ID", "HYBAS_ID", "BASIN", "YEAR", "STORMNAME"]
    tc_index = pandas.DataFrame(coincide, columns=columns)
    w_basin = pandas.merge(basins, tc_index, how="inner", on="HYBAS_ID")
    dfo_no_geom = pandas.DataFrame(dfo_orig.drop(columns="geometry"))
    out = pandas.merge(w_basin, dfo_no_geom, how="inner", on="ID")
    out.to_file(out_fpath)
    return out


def coastal_basin_x_floods(
    basins: geopandas.GeoDataFrame, floods: geopandas.GeoDataFrame, flow_threshold: int = 100
):
    # Filter by anything more than a tiny creek
    basins = basins[basins["riv_tc_usu"] > flow_threshold]

    # Select basins that drain into the ocean
    basins = basins[(basins["NEXT_DOWN"] == 0) & (basins["ENDO"] == 0)]

    # Spatial join - only checks if they are touching
    floods = floods.to_crs(basins.crs)
    joined = basins.sjoin(floods, how="inner", rsuffix="flood")
    return joined


def by_impact(w_flood: geopandas.GeoDataFrame):
    # Sort floods by impact
    # What is the value of a life? Approximately 20000 displaced people, apparently
    # (Apologies for my morbid humour)
    w_flood["impact"] = w_flood["DEAD"] * 20000 + w_flood["DISPLACED"]

    # Sort by impact and potentially affected population
    w_flood = w_flood.sort_values(["impact", "pop_ct_usu"], ascending=False)

    return w_flood


def zone_counts_shp(basins: geopandas.GeoDataFrame, shp: shapely.Geometry, threshold: float = 0.3):
    overlaps = overlaps_at_least(basins.geometry, shp, threshold)
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


def mk_expected_distribution(flood_distr: np.ndarray, n_sites: int, est_basins_per_site: int = 10):
    """
    Calculates how many sites we expect (min) for each continent/climate zone

    returns { <continent>: {<climate_zone>: int} }
    """
    total = 0
    for continent in gff.constants.HYDROATLAS_CONTINENT_NAMES:
        if continent in flood_distr:
            total += flood_distr[continent]["total"]
    cont_ratio = n_sites / total
    exp_cont_sites = {}
    exp_cont_basins = collections.defaultdict(lambda: {})
    for i, continent in enumerate(gff.constants.HYDROATLAS_CONTINENT_NAMES):
        n_cont = flood_distr[continent]["total"] * cont_ratio
        exp_cont_sites[continent] = int(n_cont)
        for clim_zone in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES:
            if (
                continent in flood_distr
                and clim_zone in flood_distr[continent]["zones"]
                and flood_distr[continent]["total"] > 0
            ):
                proportion = (
                    flood_distr[continent]["zones"][clim_zone] / flood_distr[continent]["total"]
                )
                exp_cont_basins[continent][clim_zone] = int(
                    proportion * n_cont * est_basins_per_site
                )
    return exp_cont_sites, exp_cont_basins
