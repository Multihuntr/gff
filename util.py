import math

import geopandas
import shapely
import numpy as np
import pandas as pd


# Generic geometry utilities

def rounded_bounds(arg):
    if isinstance(arg, tuple) or (isinstance(arg, np.ndarray) and arg.shape == (4,)):
        xlo, ylo, xhi, yhi = arg
        return math.floor(xlo), math.floor(ylo), math.ceil(xhi), math.ceil(yhi)
    elif isinstance(arg, shapely.Geometry):
        return rounded_bounds(shapely.bounds(arg))
    else:
        raise ValueError('Bounds must be 4-tuple, 1D np.array or a shapely.Geometry')

def shapely_bounds_to_rasterio_window(bounds):
    xlo, ylo, xhi, yhi = bounds
    return ((ylo, yhi), (xlo, xhi))

def mk_pixel_geometries(width, height, x_offset=0, y_offset=0):
    """
    Create a grid of pixel geometries, stored in a vectorised Shapely array.
    """
    xs = np.arange(width)
    ys = np.arange(height)
    yss, xss = np.meshgrid(ys, xs)
    coords = np.array([ # Clockwise squares
        [xss  +x_offset, yss  +y_offset],
        [xss+1+x_offset, yss  +y_offset],
        [xss+1+x_offset, yss+1+y_offset],
        [xss  +x_offset, yss+1+y_offset],
    ]).transpose((2,3,0,1)) # shapes [4, 2, W, H] -> [W, H, 4, 2]
    return shapely.polygons(coords)

def mk_pixel_overlap_mask(geom, bounds):
    """
    Create a mask of shape specified by bounds, where each grid cell / pixel is
    the overlap proportion between `geom` and that grid cell / pixel.

    Note that the returned array starts at the (xlo, ylo) specified in `bounds`.

    Args:
        geom (shapely.Geometry): Geometry in pixel-space to find overlap
        bounds (tuple(int)): Standard bounds

    Returns:
        np.ndarray: Shaped [W, H], proportion of overlap as an array.
    """
    xlo, ylo, xhi, yhi = bounds
    pixel_geoms = mk_pixel_geometries(xhi-xlo, yhi-ylo, x_offset=xlo, y_offset=ylo)
    intersections = geom.intersection(pixel_geoms)

    # note: pixels are defined as being unit sized, so area(intersection) == proportional overlap
    return shapely.area(intersections)



# Basin selection utilities

def get_upstream_basins(basins, basin_id):
    """
    Create a geopandas dataset containing just the basins which feed into
    the basin_row

    Args:
        basins (geopandas.GeoDataFrame): All basin shapes
        basin_id (int): Row from basins of the basin in question

    Returns:
        geopandas.GeoDataFrame: All basin shapes that share a river
    """
    # Quickly filter to basins in the same river system
    basin_row = basins.loc[basins['HYBAS_ID'] == basin_id]
    basins_w_same_sink = basins[basins['NEXT_SINK'] == basin_row['NEXT_SINK'].item()]

    # Collect all upstream. Scans for upstream basins in waves.
    # Start with the basin in question
    incl_ids = {basin_id}
    include_list = [basins_w_same_sink[basins_w_same_sink['HYBAS_ID'] == basin_id]]
    while True:
        # Add new basins that feed into the latest layer of basins
        new_included = basins_w_same_sink[basins_w_same_sink['NEXT_DOWN'].isin(incl_ids)]
        include_list.append(new_included)

        # Update ids to check for the next layer of upstream basins.
        incl_ids = set(new_included['HYBAS_ID'].to_numpy().tolist())
        if len(incl_ids) == 0:
            break

    # Concat the geodataframes back into a single dataframe
    return geopandas.GeoDataFrame(pd.concat(include_list, ignore_index=True))
