import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import shapely
import tqdm

import gff.util as util


def parse_args(argv):
    parser = argparse.ArgumentParser("Group KuroSiwo targets into individual TIF files")

    parser.add_argument("ks_folder", type=Path, help="Folder containing KuroSiwo targets")
    parser.add_argument("hydroatlas_path", type=Path)
    parser.add_argument("--group_by_date", action="store_true")
    parser.add_argument("--hydroatlas_ver", type=int, default=10)

    return parser.parse_args(argv)


def mk_tif_profile(path, info, geoms):
    footprint = shapely.union_all([geom for geom in geoms]).convex_hull
    xlo, ylo, xhi, yhi = footprint.bounds
    first_tif = rasterio.open(path / f"{list(info['datasets'].keys())[0]}.tif")
    T0 = first_tif.transform
    res = resx, resy = T0[0], -T0[4]
    transform = rasterio.transform.from_origin(xlo, yhi, *res)
    return {
        "height": abs((yhi - ylo) / resy),
        "width": abs((xhi - xlo) / resx),
        "count": 1,
        "crs": "EPSG:3857",
        "transform": transform,
        "dtype": "uint8",
        "compress": "PACKBITS",
        "nodata": 255,
    }


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
        if args.group_by_date:
            groups[d].append((p.parent, info))
        else:
            groups[(d, info["aoiid"])].append((p.parent, info, geom))
        if i % 1000 == 0:
            print(f"{i} tiles processed | {len(groups)} groups found", end="\r")
    print(f"{i} tiles processed | {len(groups)} groups found")

    # Rewrite tile data into one file per group
    print("Writing merged tifs...")
    lbl_folder: Path = args.ks_folder / "merged-labels"
    lbl_folder.mkdir(exist_ok=True)
    group_basin_counts = collections.defaultdict(lambda: [])
    group_aoi_fnames = collections.defaultdict(lambda: [])
    for k, group in tqdm.tqdm(list(groups.items())):
        paths, infos, geoms = zip(*group)

        # Organise filenames
        if args.group_by_date:
            d = k
            fname = f"{d.strftime(d_fmt)}.tif"
        else:
            (d, aoiid) = k
            fname = f"{d.strftime(d_fmt)}_{aoiid}.tif"
        fname = Path(fname)
        geoms_fname = fname.with_name(fname.stem + "-geoms.gpkg")
        meta_fname = fname.with_name(fname.stem + "-meta.json")

        # Create GPKG: Filter tiles based on assigned lvl4 basin
        geoms_np = np.array(geoms)
        geoms_np_4326 = util.convert_crs(geoms_np, "EPSG:3857", basins04_df.crs)
        hybas_id, geom_mask = util.tile_mask_for_basin(geoms_np_4326, basins04_df)
        geoms_np_masked = geoms_np[~geom_mask]
        geoms_df = geopandas.GeoDataFrame(
            geoms_np_masked, columns=["geometry"], geometry="geometry", crs="EPSG:3857"
        )
        geoms_df.to_file(lbl_folder / geoms_fname)
        group_basin_counts[d].append((hybas_id, len(geoms_df)))
        group_aoi_fnames[d].append((hybas_id, fname, geoms_fname, meta_fname))

        # Create TIF: Write tiles
        profile = mk_tif_profile(paths[0], infos[0], geoms)
        with rasterio.open(lbl_folder / fname, "w", **profile) as out_tif:
            out_tif.update_tags(date=d.isoformat())
            for (path, info, geom), mask in zip(group, geom_mask):
                if mask:
                    continue
                target_file = [name for name in info["datasets"] if "MK0_MLU_" in name][0]
                with rasterio.open(path / f"{target_file}.tif") as in_tif:
                    target_data = in_tif.read()
                window = util.shapely_bounds_to_rasterio_window(geom.bounds, out_tif.transform)
                out_tif.write(target_data, window=window)

        # Create meta.json
        def ks_to_isodate(s):
            return datetime.datetime.strptime(s, d_fmt).isoformat()

        meta = {
            "type": "kurosiwo",
            "HYBAS_ID_4": hybas_id.item(),
            "pre2_date": ks_to_isodate(infos[0]["sources"]["SL2"]["source_date"]),
            "pre1_date": ks_to_isodate(infos[0]["sources"]["SL1"]["source_date"]),
            "post_date": ks_to_isodate(infos[0]["sources"]["MS1"]["source_date"]),
            "floodmap": str(fname),
            "visit_tiles": str(geoms_fname),
            "flood_tiles": str(geoms_fname),
            "info": infos[0],
            "flooding": True,
        }
        with open(lbl_folder / meta_fname, "w") as f:
            json.dump(meta, f)

    # Remove aois outside the lvl4 basin the group was mostly assigned to
    return
    # TODO: Allow if far enough away; disallow if within 3 days of each other
    # NOTE: I have manually checked all of kurosiwo labels, and this isn't necessary
    print("Checking/removing aois across basin boundaries")
    if not args.group_by_date:
        for d, v in group_basin_counts.items():
            counts = collections.defaultdict(lambda: 0)
            for hybas_id, count in v:
                counts[hybas_id] += count
            majority_hybas_id = list(counts.keys())[np.argmax(counts.values())]
            for hybas_id, fname, geoms_fname, meta_fname in group_aoi_fnames[d]:
                if hybas_id != majority_hybas_id:
                    print(f"Removing {fname}")
                    (lbl_folder / fname).unlink()
                    (lbl_folder / geoms_fname).unlink()
                    (lbl_folder / meta_fname).unlink()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
