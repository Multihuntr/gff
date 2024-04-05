import datetime
import json

import shapely


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
