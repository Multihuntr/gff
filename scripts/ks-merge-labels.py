import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import asf_search as asf
import geopandas
import numpy as np
import rasterio
import shapely
import tqdm

import gff.constants
import gff.generate.floodmaps
import gff.generate.search
import gff.util


def parse_args(argv):
    parser = argparse.ArgumentParser("Group KuroSiwo targets into individual TIF files")

    parser.add_argument("ks_folder", type=Path, help="Folder containing KuroSiwo targets")
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--hydroatlas_ver", type=int, default=10)
    parser.add_argument("--export_ks_s1", action="store_true")

    return parser.parse_args(argv)


def mk_tif_profile(path, info, geoms):
    footprint = shapely.union_all([geom for geom in geoms]).convex_hull
    xlo, ylo, xhi, yhi = footprint.bounds
    first_tif = rasterio.open(path / f"{list(info['datasets'].keys())[0]}.tif")
    T0 = first_tif.transform
    res = resx, resy = T0[0], -T0[4]
    transform = rasterio.transform.from_origin(xlo, yhi, *res)
    return {
        **gff.constants.FLOODMAP_PROFILE_DEFAULTS,
        "height": abs((yhi - ylo) / resy),
        "width": abs((xhi - xlo) / resx),
        "count": 1,
        "crs": first_tif.crs,
        "transform": transform,
    }


def check_fully_covers(result: dict, geom: shapely.Geometry):
    res_geom = shapely.Polygon(result["geometry"]["coordinates"][0])
    return res_geom.covers(geom)


def parse_date_fname(fpath: Path):
    # *-YYYY-MM-DD.tif
    parts = fpath.stem.split("-")
    d_str = "-".join(parts[-3:])
    return datetime.datetime.fromisoformat(d_str)


def standardise_iso_fmt(s):
    return datetime.datetime.fromisoformat(s).isoformat()


class NoS1Exception(Exception):
    pass


def main(args):
    basin_path = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    basin04_fname = f"BasinATLAS_v{args.hydroatlas_ver}_lev04.shp"
    basins04_df = geopandas.read_file(basin_path / basin04_fname, engine="pyogrio")
    d_fmt = "%Y-%m-%d"

    # Collect tiles into groups based on date/aoi
    print("Grouping...")
    groups = collections.defaultdict(lambda: [])
    for i, p in enumerate(args.ks_folder.glob("**/info.json")):
        with p.open() as f:
            info = json.load(f)
        d = datetime.datetime.fromisoformat(info["flood_date"])
        geom = shapely.from_wkt(info["geom"])
        groups[(d, info["actid"], info["aoiid"])].append((p.parent, info, geom))
        if i % 1000 == 0:
            print(f"{i:5d} tiles processed | {len(groups):2d} groups found", end="\r")
    print(f"{i} tiles processed | {len(groups)} groups found")

    # Rewrite tile data into one file per group
    print("Writing merged tifs...")
    lbl_folder: Path = args.ks_folder / "merged-labels"
    lbl_folder.mkdir(exist_ok=True)
    s1_folder: Path = args.ks_folder / "merged-s1"
    s1_folder.mkdir(exist_ok=True)
    for k, group in tqdm.tqdm(list(groups.items())):
        paths, infos, geoms = zip(*group)

        # Organise filenames
        (d, actid, aoiid) = k
        id_str = f"{actid}-{aoiid}"
        post_d = datetime.datetime.fromisoformat(infos[0]["sources"]["MS1"]["source_date"])
        fname = f"{id_str}-{post_d.strftime(d_fmt)}.tif"
        fname = Path(fname)
        geoms_fname = fname.with_name(fname.stem + "-geoms.gpkg")
        meta_fname = fname.with_name(fname.stem + "-meta.json")

        # Create GPKG: Filter tiles based on assigned lvl4 basin
        geoms_np = np.array(geoms)
        geoms_np_4326 = gff.util.convert_crs(geoms_np, "EPSG:3857", basins04_df.crs)
        hybas_id, geom_mask = gff.util.tile_mask_for_basin(geoms_np_4326, basins04_df)
        geoms_np_masked = geoms_np[~geom_mask]
        geoms_df = geopandas.GeoDataFrame(
            geoms_np_masked, columns=["geometry"], geometry="geometry", crs="EPSG:3857"
        )
        geoms_df.to_file(lbl_folder / geoms_fname)

        # Create label TIFs: Write tiles
        profile = mk_tif_profile(paths[0], infos[0], geoms)
        with rasterio.open(lbl_folder / fname, "w", **profile) as out_tif:
            out_tif.update_tags(date=d.isoformat())
            to_iter = list(zip(group, geom_mask))
            for (path, info, geom), mask in tqdm.tqdm(to_iter, desc="Labels", leave=False):
                if mask:
                    continue
                # Add target tile
                target_file = [name for name in info["datasets"] if "MK0_MLU_" in name][0]
                with rasterio.open(path / f"{target_file}.tif") as in_tif:
                    target_data = in_tif.read()
                window = gff.util.shapely_bounds_to_rasterio_window(geom.bounds, out_tif.transform)
                out_tif.write(target_data, window=window)

        # Create merged s1 TIFs: Write tiles
        s1_profile = {**profile, **gff.constants.S1_PROFILE_DEFAULTS}
        if args.export_ks_s1:
            try:
                # Create s1 images: asf_search
                combined_geom = shapely.unary_union(geoms_np_4326)
                existing_s1 = list((args.data_path / "s1").glob(f"{id_str}-*"))
                if len(existing_s1) >= 1:
                    s1_fpath = existing_s1[0]
                    s1_d = parse_date_fname(s1_fpath)
                else:
                    last_date = datetime.datetime.fromisoformat(
                        infos[0]["sources"]["MS1"]["source_date"]
                    )
                    no_img_buffer = datetime.timedelta(days=7)
                    search_window = datetime.timedelta(days=60)
                    search_end = last_date - no_img_buffer
                    search_start = search_end - search_window
                    search_results = gff.generate.search.do_search(
                        combined_geom, search_start, search_end
                    )
                    chosen_result = None
                    for result in reversed(search_results):
                        if check_fully_covers(result, combined_geom):
                            chosen_result = result
                            break
                    if chosen_result is None:
                        raise NoS1Exception(f"No S1 image was found for site: {actid}-{aoiid}")
                    s1_fpath = gff.generate.floodmaps.ensure_s1(
                        args.data_path, id_str, [result], delete_intermediate=True
                    )
                    s1_d = datetime.datetime.fromisoformat(
                        chosen_result["properties"]["startTime"]
                    )

                pre1_date = s1_d.isoformat()
            except NoS1Exception as e:
                print("Falling back to using kurosiwos own S1 images")
                d_str = infos[0]["sources"]["SL1"]["source_date"]
                s1_fname = f"{actid}-{aoiid}-{d_str}.tif"
                with rasterio.open(s1_folder / s1_fname, "w", **s1_profile) as out_tif:
                    out_tif.descriptions = ("VV", "VH")
                    for band_idx, polarisation in zip(range(1, 3), ["VV", "VH"]):
                        ds_name = f"SL1_I{polarisation}"
                        iter = tqdm.tqdm(list(zip(group, geom_mask)), desc="S1", leave=False)
                        for (path, info, geom), mask in iter:
                            if mask:
                                continue
                            s1_file = [name for name in info["datasets"] if ds_name in name][0]
                            with rasterio.open(path / f"{s1_file}.tif") as in_tif:
                                s1_data = in_tif.read()
                            window = gff.util.shapely_bounds_to_rasterio_window(
                                geom.bounds, out_tif.transform
                            )
                            out_tif.write(s1_data[0], band_idx, window=window)
                pre1_date = standardise_iso_fmt(d_str)
        else:
            pre1_date = standardise_iso_fmt(infos[0]["sources"]["SL1"]["source_date"])

        # Create meta.json

        meta = {
            "type": "kurosiwo",
            "key": f'{infos[0]["actid"]}-{infos[0]["aoiid"]}',
            "HYBAS_ID_4": hybas_id.item(),
            "actid": infos[0]["actid"],
            "aoiid": infos[0]["aoiid"],
            "pre1_date": pre1_date,
            "post_date": standardise_iso_fmt(infos[0]["sources"]["MS1"]["source_date"]),
            "floodmap": str(fname),
            "visit_tiles": str(geoms_fname),
            "flood_tiles": str(geoms_fname),
            "info": infos[0],
            "flooding": True,
        }
        with open(lbl_folder / meta_fname, "w") as f:
            json.dump(meta, f)

    return


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
