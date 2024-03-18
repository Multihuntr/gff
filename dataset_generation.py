import datetime
import functools
import itertools
import math
from pathlib import Path
import shutil
import subprocess

import affine
import geopandas
import numpy as np
import pandas as pd
import rasterio
import scipy
import shapely
import skimage
import torch

import constants
import my_kurosiwo
import util


def check_tifs_match(tifs):
    transforms = [tif.transform for tif in tifs]
    t0 = transforms[0]
    res = t0[0], t0[4]
    assert all(
        [np.allclose((t[0], t[4]), res) for t in transforms[1:]]
    ), "Not all tifs have same resolution"
    assert all([tif.crs == tifs[0].crs for tif in tifs[1:]]), "Not all tifs have the same CRS"


def ensure_s1(img_folder, prefix, results, delete_intermediate=False):
    d = datetime.datetime.fromisoformat(results[0]["properties"]["startTime"])
    d_str = d.strftime("%Y-%m-%d")
    filename = f"{prefix}-{d_str}.tif"
    out_fpath = img_folder / filename
    if out_fpath.exists():
        return out_fpath

    zip_fpaths, processed_fpaths = [], []
    for result in results:
        # Download image
        zip_fpath = my_kurosiwo.download_s1(img_folder, result)
        zip_fpaths.append(zip_fpath)

        # Run preprocessing on whole image
        dim_fname = Path(zip_fpath.with_name(zip_fpath.stem + "-processed.dim").name)
        my_kurosiwo.preprocess(img_folder, zip_fpath.name, dim_fname)
        # DIMAP is a weird format. You ask it to save at ".dim" and it actually saves elsewhere
        # Anyway, need to combine the vv and vh bands into a single file for later merge
        data_path = img_folder / "tmp" / dim_fname.with_suffix(".data")
        processed_fpath = img_folder / dim_fname.with_suffix(".tif")
        processed_fpaths.append(processed_fpath)
        if not processed_fpath.exists():
            print("Merging DIMAP to a single TIF.")
            subprocess.run(
                [
                    "gdal_merge.py",
                    "-separate",
                    "-o",
                    processed_fpath,
                    data_path / "Sigma0_VV.img",
                    data_path / "Sigma0_VH.img",
                ]
            )
            if delete_intermediate:
                shutil.rmtree(data_path)
                data_path.with_suffix(".dim").unlink()

    if len(processed_fpaths) > 1:
        print("Merging multiple captures from the same day into a single tif")
        subprocess.run(["gdal_merge.py", "-n", "0", "-o", out_fpath, *processed_fpaths])
    else:
        shutil.copy(processed_fpaths[0], out_fpath)

    if delete_intermediate:
        for p in zip_fpaths:
            p.unlink()
        for p in processed_fpaths:
            p.unlink()
    return out_fpath


def create_flood_maps(
    img_folder, search_results, basin_shp, rivers_df, flood_model, include_s1=True, n_img=3
):
    """
    Creates flood maps for a single basin x flood shape by starting along rivers
    and expanding the flood maps until there's no more flood, or the edge of the image
    is reached.

    Will download S1 images as needed. Since the flood time window is quite long, and we
    don't know when the flood actually happened, more than three S1 images may be downloaded.
    The flood_model is used to decide whether a flood is visible.

    This will stop when flooding is found, or we run out of sentinel images.
    """
    cross_term = f"{basin_shp.ID}-{basin_shp.HYBAS_ID}"

    # Should go through all viable sets. What is a viable set?
    # Let's assume we have a sequence of S1 images: [-3, -2, -1, 0, 1, 2, 3, 4]
    # Where 0 is the first time a S1 image has been taken after a flood BEGAN.

    # We start with [-2, -1, 0]
    # Run progressively_grow_floodmaps on that set, and then
    # If there is significant flooding, the next set is: [-3, -2, 0]
    # If there is not, the next set is: [-1, 0, 1]

    # If we go from significant flooding to not-significant flooding, then we are done.
    # Flood over.

    start_ts = basin_shp["BEGAN"].to_pydatetime().timestamp()
    first_after = None
    for i, result in enumerate(search_results):
        ts = datetime.datetime.fromisoformat(result[0]["properties"]["startTime"]).timestamp()
        if ts >= start_ts:
            first_after = i
            break
    open_set = [[first_after - 2, first_after - 1, first_after]]
    found_flooding = False
    flood_metas = []

    while len(open_set) > 0:
        search_idx = open_set.pop(0)

        s1_img_paths = []
        date_strs = []
        for i in search_idx:
            res = search_results[i]
            # Get image
            img_path = ensure_s1(img_folder / "s1", cross_term, res, delete_intermediate=True)
            s1_img_paths.append(img_path)
            # Get image date
            res_date = datetime.datetime.fromisoformat(res[0]["properties"]["startTime"])
            date_strs.append(res_date.strftime("%Y-%m-%d"))

        print(f"Searching using S1 from   {'  '.join(date_strs)}")

        filename = f"{cross_term}-{'-'.join(date_strs)}.tif"
        floodmap_path = img_folder / "floodmaps" / filename
        visit_tiles, flood_tiles, s1_export = progressively_grow_floodmaps(
            s1_img_paths,
            rivers_df,
            basin_shp.geometry,
            floodmap_path,
            flood_model,
            include_s1=include_s1,
            print_freq=200,
        )

        if len(flood_tiles) < 50:
            print(" No major flooding found.")
            # floodmap_path.unlink()
            # s1_export.unlink()
            if found_flooding:
                print(" Since previous flooding was found, we're done")
                break
            if all([i >= 0 and i < len(search_results) for i in new_search_idx]):
                open_set.append([idx + 1 for idx in search_idx])
        else:
            print(" Major flooding found.")
            meta = {
                "FLOOD": f"{basin_shp.ID}",
                "HYBAS_ID": f"{basin_shp.HYBAS_ID}",
                "pre2_date": search_results[search_idx[0]][0]["properties"]["startTime"],
                "pre1_date": search_results[search_idx[1]][0]["properties"]["startTime"],
                "post_date": search_results[search_idx[2]][0]["properties"]["startTime"],
                "s1": str(s1_export),
                "floodmap": str(floodmap_path),
                "visit_tiles": visit_tiles,
                "flood_tiles": flood_tiles,
            }
            flood_metas.append(meta)
            found_flooding = True

            if include_s1:
                with rasterio.open(s1_export, "r+") as s1_tif:
                    desc = []
                    for i, img in enumerate(["pre2", "pre1", "post"][-n_img:]):
                        for pol in ["vv", "vh"]:
                            desc.append(f"{img}-{pol}")
                        desc_keys = ["flightDirection", "pathNumber", "startTime", "orbit"]
                        desc_dict = {
                            k: search_results[search_idx[i]][0]["properties"][k] for k in desc_keys
                        }
                        s1_tif.update_tags(2 * i + 1, **desc_dict, polarisation="vv")
                        s1_tif.update_tags(2 * i + 2, **desc_dict, polarisation="vh")
                    s1_tif.descriptions = desc
            new_search_idx = [*[i - 1 for i in search_idx[:-1]], search_idx[-1]]
            if all([i >= 0 and i < len(search_results) for i in new_search_idx]):
                open_set.append(new_search_idx)


def progressively_grow_floodmaps(
    inp_img_paths: list[Path],
    rivers_df: geopandas.GeoDataFrame,
    geom: shapely.Geometry,
    floodmap_path: Path,
    flood_model: torch.nn.Module,
    tile_size: int = 224,
    include_s1: bool = False,
    print_freq: int = 0,
):
    """
    First run flood_model along riverway from geom, then expand the search as flooded areas are found.
    Stores all visited tiles to floodmap_path.

    Conditions:
     - all tifs in inp_img_paths must match resolution and CRS.
     - geom must be in CRS of tifs

    Thus the output is in the same CRS.

    The output will be aligned pixel-wise with the last input image, and the other inputs will
    be resampled to match
    """
    # Open and check tifs
    inp_tifs = [rasterio.open(p) for p in inp_img_paths]
    ref_tif = inp_tifs[-1]
    check_tifs_match(inp_tifs)

    # Create tile_grid and initial set of tiles
    viable_footprint = shapely.intersection_all([util.image_footprint(tif) for tif in inp_tifs])
    tile_grids, viable_footprint = mk_grid(viable_footprint, ref_tif.transform, tile_size)
    tiles = tiles_along_river_within_geom(rivers_df, geom, tile_grids[(0, 0)], ref_tif.crs)
    visited = np.zeros_like(tile_grids[(0, 0)]).astype(bool)
    grid_size = tile_grids[(0, 0)].shape
    offset_cache = {(0, 1): {}, (1, 0): {}, (1, 1): {}}
    if len(tiles) == 0:
        raise Exception("Provided geometry does not intersect any rivers")

    # Rasterio profile handling
    outxlo, _, _, outyhi = viable_footprint.bounds
    t = ref_tif.transform
    new_transform = rasterio.transform.from_origin(outxlo, outyhi, t[0], -t[4])
    floodmap_nodata = 255
    profile = {
        **ref_tif.profile,
        **constants.FLOODMAP_PROFILE_DEFAULTS,
        "nodata": floodmap_nodata,
        "transform": new_transform,
        "width": tile_grids[0, 0].shape[0] * tile_size,
        "height": tile_grids[0, 0].shape[1] * tile_size,
    }
    s1_tif = None
    s1_fpath = None
    if include_s1:
        s1_count = len(inp_img_paths) * ref_tif.profile["count"]
        s1_profile = {
            **ref_tif.profile,
            "COMPRESS": "DEFLATE",
            "ZLEVEL": 1,
            "PREDICTOR": 2,
            "count": s1_count,
            "transform": new_transform,
            "width": tile_grids[0, 0].shape[0] * tile_size,
            "height": tile_grids[0, 0].shape[1] * tile_size,
        }
        s1_fpath = floodmap_path.with_stem(f"{floodmap_path.stem}_s1")
        s1_tif = rasterio.open(s1_fpath, "w", **s1_profile)

    # Begin flood-fill search
    visit_tiles, flood_tiles = [], []
    with rasterio.open(floodmap_path, "w", **profile) as out_tif:
        n_visited = 0
        n_flooded = 0
        n_permanent = 0
        n_outside = 0
        while len(tiles) > 0:
            # Grab a tile and get logits
            tile_x, tile_y = tiles.pop(0)
            visited[tile_x, tile_y] = True
            n_visited += 1
            tile_geom = tile_grids[(0, 0)][tile_x, tile_y]
            visit_tiles.append(tile_geom)
            s1_inps, flood_logits = run_flood_model(inp_tifs, tile_geom, flood_model)
            s1_inps = [t[0].cpu().numpy() for t in s1_inps]
            if not s1_preprocess_edge_heuristic(s1_inps):
                n_outside += 1
                continue

            # Smooth logits out over adjacent tiles
            adjacent_logits = get_adjacent_logits(
                inp_tifs, tile_grids, tile_x, tile_y, flood_model, offset_cache
            )
            flood_logits = average_logits_towards_edges(flood_logits, adjacent_logits)

            # Write classes to disk
            flood_cls = flood_logits.argmax(axis=0)[None].astype(np.uint8)
            window = util.shapely_bounds_to_rasterio_window(tile_geom.bounds, out_tif.transform)
            if include_s1:
                s1_tif.write(np.concatenate([t for t in s1_inps]), window=window)
            out_tif.write(flood_cls, window=window)

            # Select new potential tiles (if not visited)
            if ((flood_cls == constants.KUROSIWO_PW_CLASS).mean() > 0.5) or (
                (flood_cls == constants.KUROSIWO_BG_CLASS).mean() < 0.1
            ):
                # Don't go into the ocean or large lakes
                n_permanent += 1
            elif check_flooded(flood_cls):
                flood_tiles.append(tile_geom)
                n_flooded += 1
                new_tiles = sel_new_tiles_big_window(tile_x, tile_y, *grid_size, add=5)
                for tile in new_tiles:
                    if not visited[tile] and (tile not in tiles):
                        tiles.append(tile)

            # Logging
            if print_freq > 0 and n_visited % print_freq == 0 or len(tiles) == 0:
                print(
                    f"{len(tiles):6d} tiles open,",
                    f"{n_visited:6d} tiles visited,",
                    f"{n_flooded:6d} tiles flooded,",
                    f"{n_permanent:6d} tiles in large bodies of water",
                    f"{n_outside:6d} tiles outside legal bounds,",
                )
    if include_s1:
        s1_tif.close()
    for tif in inp_tifs:
        tif.close()

    # Finally postprocess to clean up the edges
    with rasterio.open(floodmap_path) as out_tif:
        floodmaps = out_tif.read()
    nan_mask = floodmaps == floodmap_nodata
    floodmaps = postprocess_classes(floodmaps[0], mask=(~nan_mask)[0])[None]
    floodmaps[nan_mask] = floodmap_nodata
    with rasterio.open(floodmap_path, "w", **profile) as out_tif:
        out_tif.write(floodmaps)
    return visit_tiles, flood_tiles, s1_fpath


def major_upstream_riverways(basins_df, start, bounds, threshold=20000):
    """Creates a shape that covers all the major upstream riverways within bounds"""
    upstream = util.get_upstream_basins(basins_df, start["HYBAS_ID"])
    major = upstream[upstream["riv_tc_usu"] > threshold]
    return shapely.intersection(major.convex_hull(), bounds)


def mk_grid(geom: shapely.Geometry, transform: affine.Affine, gridsize: int):
    """Create a grid in CRS-space where each block is `gridsize` large"""
    # Translate geom into some pixel-space coordinates
    geom_in_px = shapely.ops.transform(lambda x, y: ~transform * (x, y), geom)

    # So that you can create a grid that is the correct size in pixel coordinates
    xlo, ylo, xhi, yhi = geom_in_px.bounds
    xlo, ylo, xhi, yhi = math.ceil(xlo), math.ceil(ylo), math.floor(xhi), math.floor(yhi)
    w_px, h_px = (xhi - xlo), (yhi - ylo)
    s = gridsize
    grids = {
        (0, 0): util.mk_box_grid(w_px, h_px, xlo, ylo, s, s),
        (1, 0): util.mk_box_grid(w_px - s, h_px, xlo + s // 2, ylo, s, s),
        (0, 1): util.mk_box_grid(w_px, h_px - s, xlo, ylo + s // 2, s, s),
        (1, 1): util.mk_box_grid(w_px - s, h_px - s, xlo + s // 2, ylo + s // 2, s, s),
    }

    # Then translate grid back into CRS so that they align correctly across images
    for grid in grids.values():
        util.convert_affine_inplace(grid, transform, dtype=np.float64)
    new_geom = shapely.box(xlo, ylo, xhi, yhi)
    pixel_aligned_geom = shapely.ops.transform(
        lambda x, y: transform * np.array((x, y), dtype=np.float64), new_geom
    )

    # Note this only works if all images these grids are applied to have the exact same resolution
    return grids, pixel_aligned_geom


def tiles_along_river_within_geom(rivers_df, geom, tile_grid, crs):
    """
    Select tiles from a tile_grid where the rivers (multiple LINEs) touch a geom (a POLYGON).
    """
    # River geoms within geom
    rivers_df = rivers_df.to_crs(crs)
    river = rivers_df[rivers_df.geometry.intersects(geom)]
    if len(river) == 0:
        return []

    # Combine river geoms and check for intersection with tile_grid
    intersects = shapely.union_all(river.geometry.values).intersects(tile_grid)

    # Return the tile coordinates of intersection as a list
    x, y = intersects.nonzero()
    return list(zip(x, y))


def run_flood_model(tifs, geom, model):
    # TODO: Make fancy combinations of models
    return my_kurosiwo.run_flood_vit_once(tifs, geom, model)


def _ensure_logits(tifs, offset_tile_grid, x, y, flood_model, offset_cache):
    if (x, y) in offset_cache:
        return
    w, h = offset_tile_grid.shape
    if x >= 0 and y >= 0 and x < w and y < h:
        offset_tile_geom = offset_tile_grid[x, y]
        _, offset_logits = run_flood_model(tifs, offset_tile_geom, flood_model)
        offset_cache[(x, y)] = offset_logits
    else:
        offset_cache[(x, y)] = None


def get_adjacent_logits(tifs, tile_grids, tile_x, tile_y, flood_model, offset_cache):
    """
    Given a set of tile_grids which are offset from one another by half a tile,
    run the flood model and add the tile to the offset_cache if it is not already there.
    Then return the tiles in all 8 adjacent directions.
    """

    # Tile grid offsets are half a tile offset as compared to original grid (positive direction),
    # thus +0,+0 is positioned to the bottom/right, and -1,-1 is positioned to the top/left
    lr_grid = tile_grids[(1, 0)]
    lr_cache = offset_cache[(1, 0)]
    _ensure_logits(tifs, lr_grid, tile_x - 1, tile_y + 0, flood_model, lr_cache)
    _ensure_logits(tifs, lr_grid, tile_x + 0, tile_y + 0, flood_model, lr_cache)

    ud_grid = tile_grids[(0, 1)]
    ud_cache = offset_cache[(0, 1)]
    _ensure_logits(tifs, ud_grid, tile_x + 0, tile_y - 1, flood_model, ud_cache)
    _ensure_logits(tifs, ud_grid, tile_x + 0, tile_y + 0, flood_model, ud_cache)

    di_grid = tile_grids[(1, 1)]
    di_cache = offset_cache[(1, 1)]
    _ensure_logits(tifs, di_grid, tile_x - 1, tile_y - 1, flood_model, di_cache)
    _ensure_logits(tifs, di_grid, tile_x + 0, tile_y - 1, flood_model, di_cache)
    _ensure_logits(tifs, di_grid, tile_x - 1, tile_y + 0, flood_model, di_cache)
    _ensure_logits(tifs, di_grid, tile_x + 0, tile_y + 0, flood_model, di_cache)

    return {
        "le": lr_cache[(tile_x - 1, tile_y + 0)],
        "ri": lr_cache[(tile_x + 0, tile_y + 0)],
        "up": ud_cache[(tile_x + 0, tile_y - 1)],
        "do": ud_cache[(tile_x + 0, tile_y + 0)],
        "tl": di_cache[(tile_x - 1, tile_y - 1)],
        "tr": di_cache[(tile_x + 0, tile_y - 1)],
        "bl": di_cache[(tile_x - 1, tile_y + 0)],
        "br": di_cache[(tile_x + 0, tile_y + 0)],
    }


@functools.lru_cache(1)
def _weight_matrices(h, w):
    """Creates a grid sized (h, w) of weights to apply to corners/edges of adjacent tiles"""
    # The -1 ensures that it's 1 at the edges and approaches 0 at the center
    spaces = np.array([(0, h // 2 - 0.5, h - 1), (0, w // 2 - 0.5, w - 1)])
    eval_coords = tuple(np.indices((h, w)))

    # Spaces describe the y and x independently (like for meshgrid)
    # Then lores is shaped like (len(yspace), len(xspace)), with lores at each coordinate
    # And eval_coords are the pixel coordinates as (Y, X) (like from meshgrid)
    def interp(lores):
        return scipy.interpolate.RegularGridInterpolator(spaces, lores)(eval_coords)

    # Create a grid of weight matrices, where the position in the grid indicates
    # where the weight matrix should be applied. e.g.
    # result[0, 0] is the weight matrix for top-left
    # result[1, 2] is the weight matrix for right, etc.
    result = np.array(
        [
            [
                interp([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                interp([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
                interp([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            ],
            [
                interp([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
                interp([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                interp([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            ],
            [
                interp([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
                interp([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
                interp([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
            ],
        ]
    )
    return result


def average_logits_towards_edges(logits, adjacent):

    c, h, w = logits.shape
    h2, w2 = h // 2, w // 2
    weights = _weight_matrices(h, w)
    slices = {
        "tl": (slice(0, h2), slice(0, w2)),
        "up": (slice(0, h2), slice(None)),
        "tr": (slice(0, h2), slice(w2, None)),
        "le": (slice(None), slice(0, w2)),
        "ri": (slice(None), slice(w2, None)),
        "bl": (slice(h2, None), slice(0, w2)),
        "do": (slice(h2, None), slice(None)),
        "br": (slice(h2, None), slice(w2, None)),
    }

    def f(tile, slc, islc, weight):
        if tile is None:
            return 0
        return tile[:, *islc] * weight[slc]

    out = np.zeros_like(logits)

    # Take the bottom-left corner of the top-right adjacent tile, and multiply by the weights
    # And similarly for the others. Complicated slightly by the fact they may not exist.
    out[:, *slices["tl"]] += f(adjacent["tl"], slices["tl"], slices["br"], weights[0, 0])
    out[:, *slices["up"]] += f(adjacent["up"], slices["up"], slices["do"], weights[0, 1])
    out[:, *slices["tr"]] += f(adjacent["tr"], slices["tr"], slices["bl"], weights[0, 2])
    out[:, *slices["le"]] += f(adjacent["le"], slices["le"], slices["ri"], weights[1, 0])
    out += logits * weights[1, 1]
    out[:, *slices["ri"]] += f(adjacent["ri"], slices["ri"], slices["le"], weights[1, 2])
    out[:, *slices["bl"]] += f(adjacent["bl"], slices["bl"], slices["tr"], weights[2, 0])
    out[:, *slices["do"]] += f(adjacent["do"], slices["do"], slices["up"], weights[2, 1])
    out[:, *slices["br"]] += f(adjacent["br"], slices["br"], slices["tl"], weights[2, 2])

    if False:
        out_tl = np.zeros_like(logits)
        out_tl[:, *slices["tl"]] += f(adjacent["tl"], slices["tl"], slices["br"], weights[0, 0])
        debug.save_as_rgb(out_tl, "debug_tl_weighted.png")
        out_up = np.zeros_like(logits)
        out_up[:, *slices["up"]] += f(adjacent["up"], slices["up"], slices["do"], weights[0, 1])
        debug.save_as_rgb(out_up, "debug_up_weighted.png")
        out_tr = np.zeros_like(logits)
        out_tr[:, *slices["tr"]] += f(adjacent["tr"], slices["tr"], slices["bl"], weights[0, 2])
        debug.save_as_rgb(out_tr, "debug_tr_weighted.png")
        out_le = np.zeros_like(logits)
        out_le[:, *slices["le"]] += f(adjacent["le"], slices["le"], slices["ri"], weights[1, 0])
        debug.save_as_rgb(out_le, "debug_le_weighted.png")
        out_ri = np.zeros_like(logits)
        out_ri[:, *slices["ri"]] += f(adjacent["ri"], slices["ri"], slices["le"], weights[1, 2])
        debug.save_as_rgb(out_ri, "debug_ri_weighted.png")
        out_bl = np.zeros_like(logits)
        out_bl[:, *slices["bl"]] += f(adjacent["bl"], slices["bl"], slices["tr"], weights[2, 0])
        debug.save_as_rgb(out_bl, "debug_bl_weighted.png")
        out_do = np.zeros_like(logits)
        out_do[:, *slices["do"]] += f(adjacent["do"], slices["do"], slices["up"], weights[2, 1])
        debug.save_as_rgb(out_do, "debug_do_weighted.png")
        out_br = np.zeros_like(logits)
        out_br[:, *slices["br"]] += f(adjacent["br"], slices["br"], slices["tl"], weights[2, 2])
        debug.save_as_rgb(out_br, "debug_br_weighted.png")

        debug.save_as_rgb(logits * weights[1, 1], "debug_centre.png")
        debug.save_as_rgb(logits, "debug_orig.png")
        debug.save_as_rgb(out, "debug_out.png")

        debug.save_as_rgb(adjacent["tl"], "debug_tl.png")
        debug.save_as_rgb(adjacent["up"], "debug_up.png")
        debug.save_as_rgb(adjacent["tr"], "debug_tr.png")
        debug.save_as_rgb(adjacent["le"], "debug_le.png")
        debug.save_as_rgb(adjacent["ri"], "debug_ri.png")
        debug.save_as_rgb(adjacent["bl"], "debug_bl.png")
        debug.save_as_rgb(adjacent["do"], "debug_do.png")
        debug.save_as_rgb(adjacent["br"], "debug_br.png")

    return out


def s1_preprocess_edge_heuristic(tensors, threshold=0.05):
    """Uses a heuristic to check if the tensor is not at the edge (i.e. False if at the edge)"""
    # The tensors are at the edge if they have a significant proportion of 0s
    return np.all([(t < 1e-5).sum() < (t.size * threshold) for t in tensors])


def check_flooded(tile, threshold=0.03):
    return (tile == constants.KUROSIWO_FLOOD_CLASS).mean() > threshold


def sel_new_tiles_big_window(tile_x, tile_y, xsize, ysize, add=3):
    xlo = max(0, tile_x - add)
    xhi = min(xsize, tile_x + add)
    ylo = max(0, tile_y - add)
    yhi = min(ysize, tile_y + add)
    return list(itertools.product(range(xlo, xhi), range(ylo, yhi)))


def load_rivers(hydroatlas_path: Path, threshold: int = 20000):
    river_path = hydroatlas_path / "filtered_rivers.gpkg"
    if river_path.exists():
        return geopandas.read_file(river_path, engine="pyogrio")

    rivers = []
    for raw_path in hydroatlas_path.glob("**/RiverATLAS_v10_*.shp"):
        r = geopandas.read_file(raw_path, engine="pyogrio", where=f"riv_tc_usu > {threshold}")
        rivers.append(r)
    rivers_df = geopandas.GeoDataFrame(pd.concat(rivers, ignore_index=True), crs=rivers[0].crs)
    rivers_df.to_file(river_path, engine="pyogrio")
    return rivers_df


def postprocess_classes(class_map, mask=None, size_threshold=50):
    # Smooth edges
    kernel = skimage.morphology.disk(radius=2)
    smoothed = skimage.filters.rank.majority(class_map, kernel, mask=mask)

    # Remove "small" holes in non-background classes
    h, w = smoothed.shape
    for cls_id in range(1, 3):
        for shp_np in skimage.measure.find_contours(smoothed == cls_id):
            if np.linalg.norm(shp_np[0] - shp_np[-1]) > 2:
                # This happens if the contour does not form a polygon
                # (e.g. cutting off a corner of the image)
                continue
            elif len(shp_np) <= 2:
                # This happens when there's a single pixel
                # Note: skimage contours places coordinates at half-pixels
                y, x = np.round(shp_np.mean(axis=0)).astype(np.int32)
                ylo, xlo = max(0, y - 2), max(0, x - 2)
                yhi, xhi = min(h, y + 3), min(w, x + 3)
                slices = (slice(ylo, yhi), slice(xlo, xhi))
                majority, _ = scipy.stats.mode(smoothed[slices], axis=None)
                smoothed[y, x] = majority
            else:
                shp = shapely.Polygon(shp_np)
                if shp.area < size_threshold:
                    ylo, xlo, yhi, xhi = util.rounded_bounds(shp.bounds)
                    ylo, xlo = max(0, ylo - 2), max(0, xlo - 2)
                    yhi, xhi = min(h, yhi + 3), min(w, xhi + 3)
                    slices = (slice(ylo, yhi), slice(xlo, xhi))
                    shp_mask = skimage.draw.polygon2mask(
                        (yhi - ylo, xhi - xlo), shp_np - (ylo, xlo)
                    )
                    majority, _ = scipy.stats.mode(smoothed[slices], axis=None)
                    smoothed[slices][shp_mask] = majority

    return smoothed
