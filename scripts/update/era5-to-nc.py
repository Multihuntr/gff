import argparse
import datetime
import sys
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import rasterio
import tqdm
import xarray

import gff.constants


def parse_args(argv):
    parser = argparse.ArgumentParser("Convert era5 data from .tif to .nc")

    parser.add_argument("data_folder", type=Path)

    return parser.parse_args(argv)


def convert_era5_tif_to_nc(in_fpath, out_fpath, keys):
    tif = rasterio.open(in_fpath)
    im_data = tif.read()
    im_data = einops.rearrange(im_data, "(N C) H W -> C N H W", N=366, C=len(keys))
    start_date = datetime.datetime.fromisoformat(tif.descriptions[0][:10])
    end_date = datetime.datetime.fromisoformat(tif.descriptions[-1][:10])
    dr = pd.date_range(start_date, end_date, 366)

    ys = np.arange(tif.shape[0], dtype=np.float32) + 0.5
    ycoords = np.stack([np.zeros_like(ys), ys], axis=-1)
    tif.transform.itransform(ycoords)
    ycoords = ycoords[:, 1]

    xs = np.arange(tif.shape[1], dtype=np.float32) + 0.5
    xcoords = np.stack([xs, np.zeros_like(xs)], axis=-1)
    tif.transform.itransform(xcoords)
    xcoords = xcoords[:, 0]

    arr = xarray.Dataset(
        {
            k: xarray.DataArray(
                im_data[i],
                dims=("time", "latitude", "longitude"),
                coords={"time": dr, "latitude": ycoords, "longitude": xcoords},
            )
            for i, k in enumerate(keys)
        }
    )
    arr.to_netcdf(out_fpath, encoding={k: {"zlib": True} for k in keys})


def main(args):
    era5_fpaths = list(args.data_folder.glob("*-era5.tif"))
    for fpath in tqdm.tqdm(era5_fpaths, "ERA5"):
        convert_era5_tif_to_nc(fpath, fpath.with_suffix(".nc"), gff.constants.ERA5_BANDS)

    era5l_fpaths = list(args.data_folder.glob("*-era5-land.tif"))
    for fpath in tqdm.tqdm(era5l_fpaths, "ERA5"):
        convert_era5_tif_to_nc(fpath, fpath.with_suffix(".nc"), gff.constants.ERA5L_BANDS)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
