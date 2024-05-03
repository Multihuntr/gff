import datetime
import json
import sqlite3
import time

import asf_search as asf
import geopandas
import shapely


def group_s1_results_by_day(results: list[dict]):
    """Groups search results by day, assuming results are already ordered in time."""
    _to_day = lambda r: datetime.datetime.fromisoformat(r["properties"]["startTime"]).day
    groups = []
    i = 0
    while i < len(results):
        group = [results[i]]
        first_day = _to_day(results[i])
        j = i + 1
        while j < len(results):
            next_day = _to_day(results[j])
            if first_day == next_day:
                group.append(results[j])
                j += 1
                i += 1
            else:
                break
        groups.append(group)
        i += 1
    assert all(
        [all([_to_day(r) == _to_day(group[0]) for r in group]) for group in groups]
    ), "Programmer error: not all days in a group are the same"
    return groups


def group_covers_geom(group, geom, prop=1.0):
    footprints = [shapely.from_geojson(json.dumps(r)) for r in group]
    group_footprint = shapely.union_all(footprints)
    return group_footprint.intersection(geom).area >= (geom.area * prop)


def min_timing(group, ref_ts, n):
    """
    Checks if the images within a group satisfy the requirements of running a model:
    that there are n-1 images before the reference timestamp (`ref_ts`), and one after
    """
    before, after = 0, 0
    for result in group:
        ts = datetime.datetime.fromisoformat(result[0]["properties"]["startTime"]).timestamp()
        if ts < ref_ts:
            before += 1
        else:
            after += 1
    return before >= n and after > 0


def do_search(geom, start_date, end_date):
    """Thin wrapper around asf.geo_search with project defaults"""
    for x in range(3):
        try:
            search_results = asf.geo_search(
                intersectsWith=geom.convex_hull.wkt,
                platform=asf.PLATFORM.SENTINEL1,
                processingLevel=asf.PRODUCT_TYPE.AMPLITUDE_GRD,
                start=start_date,
                end=end_date,
            )

            # Only include processing level GRD_HD
            only_grd = [r for r in search_results if r.properties["processingLevel"] == "GRD_HD"]
            only_grd.sort(key=lambda r: r.properties["startTime"])
            geojsons = [result.geojson() for result in only_grd]
            break
        except json.JSONDecodeError:
            backoff = 30
            print(f"Search failed. Retrying in {backoff}s")
            time.sleep(backoff)
    else:
        raise Exception("asf_search unavailable")
    return geojsons


def get_search_results(
    index: sqlite3.Connection, shp: geopandas.GeoSeries, proc_may_conflict: bool = True
):
    k = f"{shp.ID}-{shp.HYBAS_ID}"
    if proc_may_conflict:
        index.execute("BEGIN EXCLUSIVE")
    values = index.execute("SELECT json FROM results WHERE key LIKE ?", (k,)).fetchall()
    if len(values) > 0:
        index.commit()
        row = values[0]
        return json.loads(row[0])

    # Do search of asf
    start_date = datetime.datetime.fromisoformat(shp["BEGAN"])
    end_date = datetime.datetime.fromisoformat(shp["ENDED"])
    buffer = datetime.timedelta(days=60)
    geojsons = do_search(shp.geometry, start_date - buffer, end_date)

    # Write them to the index
    db_row = {"key": k, "jsons": json.dumps(geojsons)}
    index.execute("INSERT INTO results VALUES (:key, :jsons)", db_row)
    index.commit()

    return geojsons


def filter_search_results(
    results: list[dict],
    shp: geopandas.GeoSeries,
    n_required: int = 3,
    basin_prop: float = None,
):
    # Remove HH+HV results
    results = [r for r in results if r["properties"]["polarization"] == "VV+VH"]

    # Group by ascending/descending
    ascending = [r for r in results if r["properties"]["flightDirection"] == "ASCENDING"]
    descending = [r for r in results if r["properties"]["flightDirection"] == "DESCENDING"]
    ab = 1 if len(ascending) >= n_required else 0
    db = 1 if len(descending) >= n_required else 0

    # Group sentinel-1 images taken on same day.
    # When there are more than one sentinel images covering the same basin on the same
    # day, that means that the basin covers multiple S1 images, in successive captures.
    # So, we group those together and if we want a basin, we might need multiple downloads.
    ascending = group_s1_results_by_day(ascending)
    descending = group_s1_results_by_day(descending)
    ad = 1 if len(ascending) >= n_required else 0
    dd = 1 if len(descending) >= n_required else 0

    # Ensure that, within each group, enough of the basin is covered
    if basin_prop is not None:
        ascending = [g for g in ascending if group_covers_geom(g, shp.geometry, basin_prop)]
        descending = [g for g in descending if group_covers_geom(g, shp.geometry, basin_prop)]

    # Ensure minimum timing: n-1 images before BEGAN and 1 image after
    began_ts = datetime.datetime.fromisoformat(shp["BEGAN"]).timestamp()
    if not min_timing(ascending, began_ts, n_required - 1):
        ascending = []

    if not min_timing(descending, began_ts, n_required - 1):
        descending = []

    # Ensure at least n_required images available (don't know yet if we will use all three)
    if len(ascending) < n_required:
        ascending = []
        ac = 0
    else:
        ac = 1
    if len(descending) < n_required:
        descending = []
        dc = 0
    else:
        dc = 1

    return {
        "asc": ascending,
        "desc": descending,
    }, (ab, db, ad, dd, ac, dc)


def init_db(con: sqlite3.Connection):
    con.execute("BEGIN EXCLUSIVE")
    tables = con.execute("SELECT name FROM sqlite_master")
    if len(tables.fetchall()) == 0:
        con.execute("CREATE TABLE results (key, json")
    con.commit()
