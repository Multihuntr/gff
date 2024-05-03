import argparse
import collections
from pathlib import Path
import sys

import affine
import geopandas
import numpy as np
import rasterio
import shapely
import tqdm

import gff.constants
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Rasterize hydroatlas into a global tif")

    parser.add_argument(
        "reference_tif_path", type=Path, help="Rasterize to match this tif's res, crs, etc"
    )
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)


HYBAS_META_ATTR = [
    "HYBAS_ID",
    "NEXT_DOWN",
    "NEXT_SINK",
    "MAIN_BAS",
    "DIST_SINK",
    "DIST_MAIN",
    "SUB_AREA",
    "UP_AREA",
    "PFAF_ID",
    "ENDO",
    "COAST",
    "ORDER_",
    "SORT",
    "geometry",
]
HYBAS_CLASS_SIGNIFIER = ["cl", "id"]
HYBAS_ABSOLUTE_SIGNIFIER = ["m3", "mm", "mc", "ha", "tc", "cm", "mt", "ct", "ud"]
HYBAS_RELATIVE_SIGNIFIER = ["pc", "dg", "dk", "ix", "th", "kh", "pk", "mk", "dc"]

# The HYBAS properties are one of:
#   1. A class index
#   2. An absolute (summed) quantity like streamflow or yearly rainfall are larger for larger basins.
#       For these, we calculate the contribution of each basin to the pixel as:
#       the value in the basin multiplied by the pixel's proportion of the basin.
#       i.e. we assign some proportion of the summed quantity to this basin.
#   3. A relative (averaged) quantity like temperature or percent cover over the basin
#       For these, we calculate the contribution of each basin to the pixel as:
#       the value in the basin multiplied by the propotion of the pixel covered by that basin.
#       e.g. If three basins overlap a pixel, covering 10%, 30%, 60% of the pixel,
#           and with tree cover percentages of: 50%, 25%, 75%
#           Then our estimate of tree cover for this pixel is: 0.1*0.5+0.3*0.25+0.6*0.75 = 57.75%
# We need to handle these separately.

CHUNK_SIZE = 2048 * 16


def main(args):
    print("WARNING: This script requires ~300GB of RAM to run.")
    # Load reference metadata
    with rasterio.open(args.reference_tif_path) as ref_tif:
        ref_transform = ref_tif.transform
        ref_width = ref_tif.width
        ref_height = ref_tif.height
        ref_profile = ref_tif.profile

    Q = ref_transform
    if Q[1] != 0:
        raise NotImplementedError("Rotated rasters not supported")

    # Create a geotransform for the same area, but with half the resolution
    width, height = ref_width * 2 - 1, ref_height * 2 - 1
    res = (Q[0] / 2, Q[4] / 2)
    origin_x = Q[2] + Q[0] / 2
    origin_y = Q[5] + Q[4] / 2
    transform = rasterio.transform.from_origin(origin_x, origin_y, res[0], -res[1])

    # Load basin data
    basin_path = args.hydroatlas_path / "BasinATLAS" / f"BasinATLAS_v{args.hydroatlas_ver}_shp"
    basins12_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev12.shp"
    basins12_df = geopandas.read_file(
        basin_path / basins12_fname, use_arrow=True, engine="pyogrio"
    )

    # Create a grid of pixel geometries in ref_tif's crs
    ref_grid = gff.util.mk_box_grid(width, height)
    ref_grid_flat = ref_grid.ravel()
    ref_grid_indices = np.stack([g.ravel() for g in np.indices(ref_grid.shape)], axis=-1)
    gff.util.convert_affine_inplace(ref_grid, transform, dtype=np.float64)
    pixel_area = ref_grid[0, 0].area

    # Split apart dataframe into different types of variables
    basins12_df_plain = basins12_df.drop(columns=HYBAS_META_ATTR)
    cs = basins12_df_plain.columns
    cls_col_idxs = [i for i, c in enumerate(cs) if c.split("_")[1] in HYBAS_CLASS_SIGNIFIER]
    abs_col_idxs = [i for i, c in enumerate(cs) if c.split("_")[1] in HYBAS_ABSOLUTE_SIGNIFIER]
    rel_col_idxs = [i for i, c in enumerate(cs) if c.split("_")[1] in HYBAS_RELATIVE_SIGNIFIER]
    cls_cols = [cs[i] for i in cls_col_idxs]
    abs_cols = [cs[i] for i in abs_col_idxs]
    rel_cols = [cs[i] for i in rel_col_idxs]
    class_val_df = basins12_df[cls_cols]
    absolute_val_df = basins12_df[abs_cols]
    relative_val_df = basins12_df[rel_cols]

    # Shortlist pixel/basin pairs with a spatial tree search
    geom_tree = shapely.STRtree(basins12_df.geometry.values)
    basins12_areas = shapely.area(basins12_df.geometry.values)
    ps, bs = geom_tree.query(ref_grid_flat)
    cls_vals = class_val_df.iloc[bs].values.T
    # There are some that use -999 to indicate "not known"
    # This messes with my script, so I'm adding a pseudo-class "not known" on the end
    # In my opinion this is what they should have done in the first place.
    for c in range(len(cls_cols)):
        if -999 in cls_vals[c]:
            hi = cls_vals[c].max()
            cls_vals[c][cls_vals[c] == -999] = hi + 1
    abs_vals = absolute_val_df.iloc[bs].values.T
    rel_vals = relative_val_df.iloc[bs].values.T

    # Create raster
    raster = np.zeros((len(cs), *ref_grid.shape))
    abs_raster = np.zeros((len(abs_cols), *ref_grid.shape))
    rel_raster = np.zeros((len(rel_cols), *ref_grid.shape))
    vote_rasters = [
        np.zeros((*ref_grid.shape, cls_vals[c].max().item() + 1)) for c in range(len(cls_col_idxs))
    ]
    valid_mask = np.zeros_like(ref_grid, dtype=bool)
    desc = "Writing relative and absolute properties"
    for lo in tqdm.tqdm(range(0, len(ps), CHUNK_SIZE), desc=desc):
        hi = min(lo + CHUNK_SIZE, len(ps))
        psc, bsc = ps[lo:hi], bs[lo:hi]
        xsc, ysc = ref_grid_indices[psc].T
        valid_mask[xsc, ysc] = True

        # NOTE: We are comparing areas in EPSG:4326, which is not area-preserving.
        # But, each comparison is very local (approx +- 0.05 deg), so I don't care enough to change it.
        overlap_abs = shapely.intersection(ref_grid_flat[psc], geom_tree.geometries[bsc])
        overlap_area = shapely.area(overlap_abs)
        basin_proportion = overlap_area / basins12_areas[bsc]
        pixel_proportion = overlap_area / pixel_area

        # Because we have repeated indices, we need to use np.add.at for unbuffered inplace addition
        # (Normally it is buffered, meaning the last operation is the only one actually performed)
        # https://stackoverflow.com/a/23914036

        # The proportion of the basin contained within the pixel is the proportion
        # of the value from the basin to include for absolute properties.
        coords = (slice(None), xsc, ysc)
        avals = basin_proportion[None] * abs_vals[:, lo:hi]
        np.add.at(abs_raster, coords, avals)
        # The proportion of the pixel covered by the overlap is the weight
        # for calculating a weighted average for relative properties.
        # NOTE: this is not quite accurate for pixels without 100% of its area covered by basins
        #       e.g. islands
        rvals = pixel_proportion[None] * rel_vals[:, lo:hi]
        np.add.at(rel_raster, coords, rvals)
        # The proportion of the pixel covered by the overlap is
        # the fraction of a vote for the class in that basin.
        # This only accumulates votes. Determining the winner is in another loop.
        for i in range(len(cls_col_idxs)):
            np.add.at(vote_rasters[i], (xsc, ysc, cls_vals[i, lo:hi]), pixel_proportion)

    desc = "Aggregating votes for class properties"
    for i, c in enumerate(tqdm.tqdm(cls_col_idxs, desc=desc)):
        raster[c] = np.argmax(vote_rasters[i], axis=-1)

    # Write to raster
    raster[abs_col_idxs] = abs_raster
    raster[rel_col_idxs] = rel_raster
    raster[:, ~valid_mask] = np.nan

    # Write raster to disk
    out_fpath = args.data_path / gff.constants.HYDROATLAS_RASTER_FNAME
    out_profile = {
        **ref_profile,
        "count": len(cs),
        "width": width,
        "height": height,
        "transform": transform,
        "dtype": np.float32,
        "nodata": np.nan,
        "COMPRESS": "LERC",
        "MAX_Z_ERROR": 0.00001,
        "INTERLEAVE": "BAND",
        "TILED": "YES",
        "BLOCKXSIZE": 32,
        "BLOCKYSIZE": 32,
    }
    print("Writing to disk")
    with rasterio.open(out_fpath, "w", **out_profile) as tif:
        tif.descriptions = cs
        tif.write(raster.transpose((0, 2, 1)))

    # Do we want one-hot encoded classes?
    # I sure hope not!


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
