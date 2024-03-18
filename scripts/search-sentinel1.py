import argparse
import datetime
import json
from pathlib import Path
import sys

import asf_search as asf
import geopandas
import shapely


def parse_args(argv):
    desc = "Search ASF sentinel-1 archives; build map basin->s1 images"
    parser = argparse.ArgumentParser(desc)

    parser.add_argument("gpkg_path", type=Path, help="File with (floods X basins) shps")
    parser.add_argument("out_path", type=Path)

    return parser.parse_args(argv)


def group_s1_results_by_day(results):
    """Groups results by day, assuming results are already ordered in time."""
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
    before, after = 0, 0
    for result in group:
        ts = datetime.datetime.fromisoformat(result[0]["properties"]["startTime"]).timestamp()
        if ts < ref_ts:
            before += 1
        else:
            after += 1
    return before >= n and after > 0


def main(args):
    shps = geopandas.read_file(args.gpkg_path, engine="pyogrio", where="BEGAN >= '2014-01-01'")
    results_json_path = args.out_path.with_name(f"{args.out_path.stem}-raw.json")
    if results_json_path.exists():
        with results_json_path.open("r") as f:
            index = json.load(f)
    else:
        index = {}

    try:
        for i, shp in shps.iterrows():
            k = f"{shp.ID}-{shp.HYBAS_ID}"
            if k in index:
                continue

            print(k)
            start_date = shp["BEGAN"].to_pydatetime()
            end_date = shp["ENDED"].to_pydatetime()

            buffer = datetime.timedelta(days=60)
            search_results = asf.geo_search(
                intersectsWith=shp.geometry.convex_hull.wkt,
                platform=asf.PLATFORM.SENTINEL1,
                processingLevel=asf.PRODUCT_TYPE.AMPLITUDE_GRD,
                start=start_date - buffer,
                end=end_date,
            )
            # Only include processing level GRD_HD
            only_grd = [r for r in search_results if r.properties["processingLevel"] == "GRD_HD"]
            only_grd.sort(key=lambda r: r.properties["startTime"])

            # Write them to the index
            index[k] = {
                "results": [result.geojson() for result in only_grd],
                "BEGAN": shp["BEGAN"].to_pydatetime().isoformat(),
                "ENDED": shp["ENDED"].to_pydatetime().isoformat(),
            }
    finally:
        with results_json_path.open("w") as f:
            json.dump(index, f)

    filtered_index = {}
    nab, ndb = 0, 0
    nad, ndd = 0, 0
    nac, ndc = 0, 0
    basin_prop = 0.05
    n_required = 3
    for i, shp in shps.iterrows():
        k = f"{shp.ID}-{shp.HYBAS_ID}"
        results = index[k]["results"]

        # Group by ascending/descending
        ascending = [r for r in results if r["properties"]["flightDirection"] == "ASCENDING"]
        descending = [r for r in results if r["properties"]["flightDirection"] == "DESCENDING"]
        if len(ascending) >= n_required:
            nab += 1
        if len(descending) >= n_required:
            ndb += 1

        # Group sentinel-1 images taken on same day.
        # When there are more than one sentinel images covering the same basin on the same
        # day, that means that the basin covers multiple S1 images, in successive captures.
        # So, we group those together and if we want a basin, we might need multiple downloads.
        ascending = group_s1_results_by_day(ascending)
        descending = group_s1_results_by_day(descending)
        if len(ascending) >= n_required:
            nad += 1
        if len(descending) >= n_required:
            ndd += 1

        # Ensure that, within each group, the whole basin is covered
        ascending = [g for g in ascending if group_covers_geom(g, shp.geometry, basin_prop)]
        descending = [g for g in descending if group_covers_geom(g, shp.geometry, basin_prop)]

        # Ensure minimum timing: n-1 images before BEGAN and 1 image after
        began_ts = shp["BEGAN"].to_pydatetime().timestamp()
        if not min_timing(ascending, began_ts, n_required - 1):
            ascending = []
        if not min_timing(descending, began_ts, n_required - 1):
            descending = []

        # Ensure at least n_required images available (don't know yet if we will use all three)
        if len(ascending) < n_required:
            ascending = []
        else:
            nac += 1
        if len(descending) < n_required:
            descending = []
        else:
            ndc += 1

        filtered_index[k] = {
            "asc": ascending,
            "desc": descending,
            "BEGAN": shp["BEGAN"].to_pydatetime().isoformat(),
            "ENDED": shp["ENDED"].to_pydatetime().isoformat(),
        }
    with args.out_path.open("w") as f:
        json.dump(filtered_index, f)

    print(
        f"Number of times at least {n_required} images that touch the (basin x flood) shapes are available"
    )
    print(f"Search results:           Ascending {nab:5d}  |  Descending {ndb:5d}")
    print(f"Grouped by day:           Ascending {nad:5d}  |  Descending {ndd:5d}")
    print(f"Filtered by {basin_prop:.0%} coverage: Ascending {nac:5d}  |  Descending {ndc:5d}")

    n_sites = len(
        set([k[5:] for k, v in filtered_index.items() if len(v["asc"]) > 0 or len(v["desc"]) > 0])
    )
    n_floods = len(
        set([k[:4] for k, v in filtered_index.items() if len(v["asc"]) > 0 or len(v["desc"]) > 0])
    )
    print(f"Number of unique sites:  {n_sites:5d}")
    print(f"Number of unique floods: {n_floods:5d}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
