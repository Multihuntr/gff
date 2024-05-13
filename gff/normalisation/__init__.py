from pathlib import Path

import numpy as np
import pandas

_here = Path(__file__).parent
_dtypes = {"mean": np.float32, "std": np.float32}


def get_norm_w_keys(fpath: Path, keys: list[str]):
    df = pandas.read_csv(fpath, dtype=_dtypes)
    idxs = [df.band.values.tolist().index(c) for c in keys]
    return df.iloc[idxs]


def get_era5_norm(keys: list[str]):
    return get_norm_w_keys(_here / "era5_norm.csv", keys)


def get_era5_land_norm(keys: list[str]):
    return get_norm_w_keys(_here / "era5_land_norm.csv", keys)


def get_hydroatlas_norm(keys: list[str]):
    return get_norm_w_keys(_here / "hydroatlas_norm.csv", keys)


def get_dem_norm(fold: int):
    return pandas.read_csv(_here / f"dem_norm_{fold}.csv", dtype=_dtypes)


def get_s1_norm(fold: int):
    return pandas.read_csv(_here / f"s1_norm_{fold}.csv", dtype=_dtypes)


def save(fname, bands, means, stds):
    df = pandas.DataFrame({"band": bands, "mean": means, "std": stds})
    df.to_csv(_here / fname, index=False)
