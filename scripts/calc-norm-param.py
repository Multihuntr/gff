import argparse
import datetime
import json
from pathlib import Path
import sys

import einops
import numpy as np
import geopandas
import pandas
import rasterio
import tqdm
import xarray

import gff.constants
import gff.data_sources
import gff.normalisation
import gff.util


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
    count = 0
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
                # TODO: Check that locations where one property is null, all are null
                count += (~np.any(nan_mask, axis=1)).sum().item()
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
        count = 0

        for idx, window in tqdm.tqdm(list(tif.block_windows(1)), leave=False, desc=fpath.name):
            data = tif.read(window=window)
            _sum += np.nansum(data, axis=(1, 2))
            sum_sqr += np.nansum(data**2, axis=(1, 2))
            # TODO: Check that locations where one property is null, all are null
            count += np.any((data != tif.nodata), axis=0).sum().item()

    mean = _sum / count
    std = np.sqrt(sum_sqr / count - mean**2)
    mean[ignore_idxs] = 0
    std[ignore_idxs] = 1
    return mean, std, band_names


def get_stats_s1_dem(folder: Path):
    n_bands = 2
    s1_sums = []
    s1_sum_sqrs = []
    s1_counts = []
    dem_sums = []
    dem_sum_sqrs = []
    dem_counts = []
    hand_sums = []
    hand_sum_sqrs = []
    hand_counts = []
    test_fnames = []
    for i in range(gff.constants.N_PARTITIONS):
        s1_sums.append(np.zeros(n_bands, dtype=np.float64))
        s1_sum_sqrs.append(np.zeros(n_bands, dtype=np.float128))
        s1_counts.append(np.zeros(1, dtype=np.int64))
        dem_sums.append(np.zeros(1, dtype=np.float64))
        dem_sum_sqrs.append(np.zeros(1, dtype=np.float128))
        dem_counts.append(np.zeros(1, dtype=np.int64))
        hand_sums.append(np.zeros(1, dtype=np.float64))
        hand_sum_sqrs.append(np.zeros(1, dtype=np.float128))
        hand_counts.append(np.zeros(1, dtype=np.int64))
        fpath = folder / "partitions" / f"floodmap_partition_{i}.txt"
        test_fnames.append(pandas.read_csv(fpath, header=None)[0].values.tolist())

    fpaths = list((folder / "rois").glob("*-meta.json"))
    for j, meta_fpath in enumerate(tqdm.tqdm(fpaths, desc="Files")):
        with open(meta_fpath) as f:
            meta = json.load(f)

        incl_partition = []
        for i, fold_fnames in enumerate(test_fnames):
            if meta_fpath.name not in fold_fnames:
                incl_partition.append(i)

        s1_stem = gff.util.get_s1_stem_from_meta(meta)
        s1_path = meta_fpath.parent / f"{s1_stem}-s1.tif"
        s1_tif = rasterio.open(s1_path)
        visit_tiles = geopandas.read_file(
            meta_fpath.parent / meta["visit_tiles"], engine="pyogrio", use_arrow=True
        )
        fmap_path = meta_fpath.parent / Path(meta["floodmap"])
        dem_path = fmap_path.with_name(fmap_path.stem + "-dem-local.tif")
        dem_tif = rasterio.open(dem_path)
        hand_path = fmap_path.with_name(fmap_path.stem + "-hand.tif")
        hand_tif = rasterio.open(hand_path)
        for i, tile_row in tqdm.tqdm(
            visit_tiles.iterrows(), desc="Tiles", total=len(visit_tiles), leave=False
        ):
            s1_data = gff.util.get_tile(s1_tif, tile_row.geometry.bounds, align=True)
            s1_summed = s1_data.sum(axis=(1, 2))
            s1_sum_sqred = (s1_data**2).sum(axis=(1, 2))
            H, W = s1_data[0].shape
            s1_counted = H * W

            if np.isnan(s1_summed).sum() > 0:
                raise Exception("S1 has nan. That shouldn't happen")

            dem = gff.util.get_tile(dem_tif, tile_row.geometry.bounds, align=True)
            dem_summed = np.nansum(dem)
            dem_sum_sqred = np.nansum(dem**2)
            dem_counted = (~np.isnan(dem)).sum()

            hand = gff.util.get_tile(hand_tif, tile_row.geometry.bounds, align=True)
            hand_summed = np.nansum(hand)
            hand_sum_sqred = np.nansum(hand**2)
            hand_counted = (~np.isnan(hand)).sum()

            for idx in incl_partition:
                s1_sums[idx] += s1_summed
                s1_sum_sqrs[idx] += s1_sum_sqred
                s1_counts[idx] += s1_counted
                dem_sums[idx] += dem_summed
                dem_sum_sqrs[idx] += dem_sum_sqred
                dem_counts[idx] += dem_counted
                hand_sums[idx] += hand_summed
                hand_sum_sqrs[idx] += hand_sum_sqred
                hand_counts[idx] += hand_counted

    s1_means, s1_stds = [], []
    dem_means, dem_stds = [], []
    hand_means, hand_stds = [], []
    for i in range(gff.constants.N_PARTITIONS):
        s1_means.append(s1_sums[i] / s1_counts[i])
        s1_stds.append(np.sqrt(s1_sum_sqrs[i] / s1_counts[i] - s1_means[i] ** 2))
        dem_means.append(dem_sums[i] / dem_counts[i])
        dem_stds.append(np.sqrt(dem_sum_sqrs[i] / dem_counts[i] - dem_means[i] ** 2))
        hand_means.append(hand_sums[i] / hand_counts[i])
        hand_stds.append(np.sqrt(hand_sum_sqrs[i] / hand_counts[i] - hand_means[i] ** 2))

    return s1_means, s1_stds, dem_means, dem_stds, hand_means, hand_stds


def get_stats_glofas(folder: Path):
    bands = ["dis24", "rowe", "swir"]
    glofas_sums = []
    glofas_sum_sqrs = []
    glofas_counts = []
    test_fnames = []
    for i in range(gff.constants.N_PARTITIONS):
        glofas_sums.append(np.zeros(len(bands), dtype=np.float64))
        glofas_sum_sqrs.append(np.zeros(len(bands), dtype=np.float128))
        glofas_counts.append(np.zeros(len(bands), dtype=np.int64))
        fpath = folder / "partitions" / f"floodmap_partition_{i}.txt"
        test_fnames.append(pandas.read_csv(fpath, header=None)[0].values.tolist())

    fpaths = list((folder / "rois").glob("*-meta.json"))
    for j, meta_fpath in enumerate(tqdm.tqdm(fpaths, desc="Files")):
        with open(meta_fpath) as f:
            meta = json.load(f)

        incl_partition = []
        for i, fold_fnames in enumerate(test_fnames):
            if meta_fpath.name not in fold_fnames:
                incl_partition.append(i)

        d_str = datetime.datetime.fromisoformat(meta["post_date"]).strftime("%Y-%m-%d")
        glofas_fname = f'{meta["key"]}_{d_str}.nc'
        dataset = xarray.open_dataset(meta_fpath.parent / glofas_fname)

        data = np.array([getattr(dataset, band).values for band in bands])
        glofas_summed = np.nansum(data, axis=(1, 2, 3))
        glofas_sum_sqred = np.nansum(data**2, axis=(1, 2, 3))
        glofas_counted = (~np.isnan(data)).sum(axis=(1, 2, 3))

        for idx in incl_partition:
            glofas_sums[idx] += glofas_summed
            glofas_sum_sqrs[idx] += glofas_sum_sqred
            glofas_counts[idx] += glofas_counted

    glofas_means, glofas_stds = [], []
    for i in range(gff.constants.N_PARTITIONS):
        glofas_means.append(glofas_sums[i] / glofas_counts[i])
        glofas_stds.append(np.sqrt(glofas_sum_sqrs[i] / glofas_counts[i] - glofas_means[i] ** 2))

    return glofas_means, glofas_stds


def main(args):
    # Normalise ERA5, ERA5-Land and HydroATLAS by reading the whole dataset
    era5_folder = args.data_path / gff.constants.ERA5_FOLDER
    era5_files = list(era5_folder.glob("*.tif"))
    era5_bands = gff.constants.ERA5_BANDS
    era5_mean, era5_std = get_stats_era5(era5_files, "ERA5", era5_bands)
    gff.normalisation.save(args.data_path, "era5_norm.csv", era5_bands, era5_mean, era5_std)

    era5l_folder = args.data_path / gff.constants.ERA5L_FOLDER
    era5l_files = list(era5l_folder.glob("*.tif"))
    era5l_bands = gff.constants.ERA5L_BANDS
    era5l_mean, era5l_std = get_stats_era5(era5l_files, "ERA5-Land", era5l_bands)
    gff.normalisation.save(
        args.data_path, "era5_land_norm.csv", era5l_bands, era5l_mean, era5l_std
    )

    hydroatlas_fpath = args.data_path / gff.constants.HYDROATLAS_RASTER_FNAME
    hydroatlas_mean, hydroatlas_std, hydroatlas_bands = get_stats_hydroatlas(
        hydroatlas_fpath, ignore_bands=["cl", "id"]
    )

    gff.normalisation.save(
        args.data_path, "hydroatlas_norm.csv", hydroatlas_bands, hydroatlas_mean, hydroatlas_std
    )

    s1_mean, s1_std, dem_mean, dem_std, hand_mean, hand_std = get_stats_s1_dem(args.data_path)
    glofas_mean, glofas_std = get_stats_glofas(args.data_path)

    for x in range(gff.constants.N_PARTITIONS):
        gff.normalisation.save(
            args.data_path, f"s1_norm_{x}.csv", ["VV", "VH"], s1_mean[x], s1_std[x]
        )
        gff.normalisation.save(
            args.data_path, f"dem_norm_{x}.csv", ["dem"], dem_mean[x], dem_std[x]
        )
        gff.normalisation.save(
            args.data_path, f"hand_norm_{x}.csv", ["hand"], hand_mean[x], hand_std[x]
        )
        gff.normalisation.save(
            args.data_path,
            f"glofas_norm_{x}.csv",
            ["dis24", "rowe", "swir"],
            glofas_mean[x],
            glofas_std[x],
        )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
