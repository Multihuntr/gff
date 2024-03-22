import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapely

import gff.util as util

def parse_args(argv):
    parser = argparse.ArgumentParser("Group KuroSiwo targets into individual TIF files")

    parser.add_argument("kw_folder", type=Path, help="Folder containing KuroSiwo targets")

    return parser.parse_args(argv)


def main(args):
    groups = collections.defaultdict(lambda: [])
    for p in args.kw_folder.glob("**/info.json"):
        with p.open() as f:
            info = json.load(f)
        d = datetime.datetime.fromisoformat(info["flood_date"])
        info['geom'] = shapely.from_wkt(info['geom'])
        groups[d].append((p.parent, info))

    d_fmt = '%Y-%m-%d'
    for d, group in groups.items():
        paths, infos = zip(*group)
        footprint = shapely.union_all([info['geom'] for info in infos]).convex_hull
        xlo, ylo, xhi, yhi = footprint.bounds
        first_tif = rasterio.open(paths[0] / f'{list(infos[0]['datasets'].keys())[0]}.tif')
        T0 = first_tif.transform
        res = resx, resy = T0[0], -T0[4]
        T = rasterio.transform.from_origin(xlo, yhi, *res)

        profile = {
            'height': abs((yhi-ylo)/resy),
            'width': abs((xhi-xlo)/resx),
            'count': 1,
            'crs': 'EPSG:3857',
            'transform': T,
            'dtype': 'uint8',
            'compress': 'PACKBITS',
            'nodata': 255
        }
        with rasterio.open(args.kw_folder / f'{d.strftime(d_fmt)}.tif', 'w', **profile) as out_tif:
            for path, info in group:
                target_file = [name for name in info['datasets'] if 'MK0_MLU_' in name][0]
                with rasterio.open(path / f'{target_file}.tif') as in_tif:
                    target_data = in_tif.read()
                window = util.shapely_bounds_to_rasterio_window(info['geom'].bounds, out_tif.transform)
                out_tif.write(target_data, window=window)



if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
