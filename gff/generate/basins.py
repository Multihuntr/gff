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


def flood_distribution(dfo: geopandas.GeoDataFrame, basins: geopandas.GeoDataFrame):
    """Calculates the number of floods per basin"""
    basin_geom = np.array(basins.geometry.values)[:, None]
    dfo_geom = np.array(dfo.geometry.values)[None, :]
    return shapely.intersects(basin_geom, dfo_geom).sum(axis=1)


def overlap_1d_np(min1, max1, min2, max2):
    return np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))


def tcs_basins(
    dfo: geopandas.GeoDataFrame,
    basins: geopandas.GeoDataFrame,
    tcs: geopandas.GeoDataFrame,
    tc_path: Path,
    out_fpath: str,
):
    if out_fpath.exists():
        return pandas.read_csv(out_fpath)

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
    columns = ["FLOOD_ID", "HYBAS_ID", "BASIN", "YEAR", "STORMNAME"]
    out = pandas.DataFrame(coincide, columns=columns)
    out.to_csv(out_fpath, index=False)
    return out


def coastal(basins: geopandas.GeoDataFrame):
    # Select basins that drain into the ocean
    return basins[(basins["NEXT_DOWN"] == 0) & (basins["ENDO"] == 0)]


def basins_by_impact(
    basins: geopandas.GeoDataFrame,
    floods: geopandas.GeoDataFrame,
):
    # Sort floods by impact
    # What is the value of a life? Approximately 20000 displaced people, apparently
    # (Apologies for my morbid humour)
    floods["impact"] = floods["DEAD"] * 20000 + floods["DISPLACED"]

    # Spatial join - only checks if they are touching
    floods = floods.to_crs(basins.crs)
    joined = basins.sjoin(floods, how="inner", rsuffix="flood")

    # Sort by impact and potentially affected population
    joined = joined.sort_values(["impact", "pop_ct_usu"], ascending=False)

    return joined
