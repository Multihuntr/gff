import argparse
from pathlib import Path
import sys

import einops
import numpy as np
import rasterio
import tqdm

import gff.constants
import gff.normalisation


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "Calculate the normalisation parameters for various data sources"
    )

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def get_stats_era5(fpaths, desc, bands):
    n_bands = len(bands)
    _sum = np.zeros(n_bands, dtype=np.float64)
    sum_sqr = np.zeros(n_bands, dtype=np.float128)
    count = np.zeros(n_bands, dtype=np.int64)
    for fpath in tqdm.tqdm(fpaths, desc=desc):
        with rasterio.open(fpath) as tif:
            scales = np.array(tif.scales)[:, None, None]
            offsets = np.array(tif.offsets)[:, None, None]
            for idx, window in tqdm.tqdm(list(tif.block_windows(1)), leave=False, desc=fpath.name):
                data = tif.read(window=window)
                data = data * scales + offsets
                data = einops.rearrange(data, "(n c) h w -> n c h w", c=n_bands)
                nan_mask = data == tif.nodata
                data[nan_mask] = 0
                _sum += data.sum(axis=(0, 2, 3))
                sum_sqr += (data**2).sum(axis=(0, 2, 3))
                count += (~nan_mask).sum()
    mean = _sum / count
    std = np.sqrt(sum_sqr / count - mean**2)
    return mean, std


def get_stats_hydroatlas(fpath, ignore_bands):
    with rasterio.open(fpath) as tif:
        band_names = tif.descriptions
        ignore_idxs = []
        for i, c in enumerate(tif.descriptions):
            if c.split("_")[1] in ignore_bands:
                ignore_idxs.append(i)

        n_bands = len(tif.descriptions)
        _sum = np.zeros(n_bands, dtype=np.float64)
        sum_sqr = np.zeros(n_bands, dtype=np.longdouble)
        count = np.zeros(n_bands, dtype=np.int64)

        for idx, window in tqdm.tqdm(list(tif.block_windows(1)), leave=False, desc=fpath.name):
            data = tif.read(window=window)
            _sum += np.nansum(data, axis=(1, 2))
            sum_sqr += np.nansum(data**2, axis=(1, 2))
            count += (data != tif.nodata).sum()

    mean = _sum / count
    std = np.sqrt(sum_sqr / count - mean**2)
    mean[ignore_idxs] = 0
    std[ignore_idxs] = 1
    return mean, std, band_names


def main(args):
    # Normalise ERA5, ERA5-Land and HydroATLAS by reading the whole dataset
    era5_folder = args.data_path / gff.constants.ERA5_FOLDER
    era5_files = list(era5_folder.glob("*.tif"))
    era5_bands = gff.constants.ERA5_BANDS
    era5_mean, era5_std = get_stats_era5(era5_files, "ERA5", era5_bands)
    gff.normalisation.save("era5_norm.csv", era5_bands, era5_mean, era5_std)

    era5l_folder = args.data_path / gff.constants.ERA5L_FOLDER
    era5l_files = list(era5l_folder.glob("*.tif"))
    era5l_bands = gff.constants.ERA5L_BANDS
    era5l_mean, era5l_std = get_stats_era5(era5l_files, "ERA5-Land", era5l_bands)
    gff.normalisation.save("era5_land_norm.csv", era5l_bands, era5l_mean, era5l_std)

    hydroatlas_fpath = args.data_path / gff.constants.HYDROATLAS_RASTER_FNAME
    hydroatlas_mean, hydroatlas_std, hydroatlas_bands = get_stats_hydroatlas(
        hydroatlas_fpath, ignore_bands=["cl", "id"]
    )

    gff.normalisation.save(
        "hydroatlas_norm.csv", hydroatlas_bands, hydroatlas_mean, hydroatlas_std
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
