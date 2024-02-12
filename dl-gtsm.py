import argparse
from pathlib import Path
import sys
import xarray
import zipfile

import cdsapi
import tqdm


def parse_args(argv):
    parser = argparse.ArgumentParser("Download global GTSM data")

    parser.add_argument("data_path", type=Path, help="Output path for GTSM data")

    return parser.parse_args(argv)


def main(args):
    # Setup
    c = cdsapi.Client()
    scratch_path: Path = args.data_path / "scratch"
    scratch_path.mkdir(exist_ok=True)

    # In case it fails halfway through
    existing_files = args.data_path.glob("reanalysis_waterlevel_dailymax_????_??_v1.nc")
    existing_years = set([int(p.stem[31:-6]) for p in existing_files])
    for year in tqdm.tqdm(range(1979, 2019)):
        if year in existing_years:
            continue
        # Read data from cds
        dl_path = scratch_path / "download.zip"
        c.retrieve(
            "sis-water-level-change-timeseries-cmip6",
            {
                "format": "zip",
                "variable": "total_water_level",
                "experiment": "reanalysis",
                "temporal_aggregation": "daily_maximum",
                "year": str(year),
                "month": [f"{d:02d}" for d in range(1, 13)],
            },
            dl_path,
        )

        # Extract zips
        with zipfile.ZipFile(dl_path) as f:
            f.extractall(args.data_path)
        dl_path.unlink()

    # Combine all the netcdf files into a single one to save on extra indexes
    ds = xarray.open_mfdataset(args.data_path / "*.nc")
    # For some reason this takes like half an hour to run?
    ds.to_netcdf(args.data_path / "gtsm.nc")

    # Cleanup
    for old_file in args.data_path.glob("reanalysis_waterlevel_dailymax_????_??_v1.nc"):
        old_file.unlink()
    scratch_path.rmdir()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
