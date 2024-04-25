import datetime


import geopandas
import shapely


def search_result_footprint_intersection(search_results: list[dict]):
    footprints = []
    for day_group in search_results:
        shps = [shapely.Polygon(g["geometry"]["coordinates"][0]) for g in day_group]
        day_shp = shapely.convex_hull(shapely.union_all(shps))
        footprints.append(day_shp)
    return shapely.intersection_all(footprints)


def first_index_result_after(search_results: list[dict], row: geopandas.GeoSeries):
    if isinstance(row["BEGAN"], str):
        start_ts = datetime.datetime.fromisoformat(row["BEGAN"]).timestamp()
    else:
        start_ts = row["BEGAN"].to_pydatetime().timestamp()
    first_after = None
    for i, result in enumerate(search_results):
        ts = datetime.datetime.fromisoformat(result[0]["properties"]["startTime"]).timestamp()
        if ts >= start_ts:
            first_after = i
            break
    return first_after


def find_intersecting_tuplets(
    search_results: list[dict], basin_row: geopandas.GeoSeries, n_img: int, min_size: float = 0
):
    """
    Finds all sets of S1 images that overlap with each other
    """
    first_after = first_index_result_after(search_results, basin_row)

    tuplets = []
    for i in range(first_after, len(search_results)):
        tuplet = find_intersecting_tuplet(search_results, i, n_img, min_size=min_size)
        if tuplet is not None:
            tuplets.append(tuplet)

    return tuplets


def find_intersecting_tuplet(
    search_results: list[dict], future_idx: int, n_img: int, min_size: float = 0
):
    """
    Finds the first set of search result indices ending at future_idx which have an intersection
    """
    # Try combinations until we find some intersection, preferring close timesteps
    searchable_idxs = []
    for i in reversed(range(future_idx)):
        if n_img == 2:
            searchable_idxs.append((i,))
        elif n_img == 3:
            for j in reversed(range(i)):
                searchable_idxs.append((i, j))
        else:
            raise NotImplementedError("Must use either 2 or 3 for n_img")

    for combo in searchable_idxs:
        result_idxs = [*list(reversed(combo)), future_idx]
        results = [search_results[i] for i in result_idxs]

        intersection = search_result_footprint_intersection(results)
        if intersection.area > min_size:
            return result_idxs
    return None


def s1_too_close(
    existing: list[tuple[str, int, shapely.Geometry]],
    search_results: list[dict],
    idx: int,
    time_threshold: int = 3600 * 24 * 20,
    distance_threshold: float = 35 * 0.1,
):
    """Checks if the search result at idx is too close in time and space to any of existing"""
    to_check = search_results[idx]
    check_ts = datetime.datetime.fromisoformat(to_check[0]["properties"]["startTime"]).timestamp()
    check_shp = shapely.Polygon(to_check[0]["geometry"]["coordinates"][0])
    for ts, shp in existing:
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
            return (ts, shp)
    return False
