import datetime


import shapely


def search_result_footprint_intersection(search_results: list[dict]):
    footprints = []
    for day_group in search_results:
        shps = [shapely.Polygon(g["geometry"]["coordinates"][0]) for g in day_group]
        day_shp = shapely.convex_hull(shapely.union_all(shps))
        footprints.append(day_shp)
    return shapely.intersection_all(footprints)


def initial_search_floodmap_idx(search_results: list[dict], start_ts: int, n_img: int):
    first_after = None
    for i, result in enumerate(search_results):
        ts = datetime.datetime.fromisoformat(result[0]["properties"]["startTime"]).timestamp()
        if ts >= start_ts:
            first_after = i
            break

    result_idx = find_intersection_results(search_results, first_after, n_img)
    return result_idx


def find_intersection_results(search_results: list[dict], future_idx: int, n_img: int):
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
        if intersection.area > 0:
            return result_idxs
    return None
