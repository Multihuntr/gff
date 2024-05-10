import pandas
from pathlib import Path

_here = Path(__file__).parent


def era5_norm():
    return pandas.read_csv(_here / "era5_norm.csv", index=False)


def era5_land_norm():
    return pandas.read_csv(_here / "era5_land_norm.csv", index=False)


def hydroatlas_norm():
    return pandas.read_csv(_here / "hydroatlas_norm.csv", index=False)


def dem_norm(fold: int):
    return pandas.read_csv(_here / f"dem_norm_{fold}.csv", index=False)


def s1_norm(fold: int):
    return pandas.read_csv(_here / f"s1_norm_{fold}.csv", index=False)


def save(fname, bands, means, stds):
    df = pandas.DataFrame({"band": bands, "mean": means, "std": stds})
    df.to_csv(_here / fname, index=False)
