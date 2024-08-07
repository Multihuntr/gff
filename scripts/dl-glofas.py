import os
import sys
import tqdm
import datetime

import argparse
from pathlib import Path

import cdsapi
import xarray
import zipfile

import geopandas as gpd

# example usage:
# python dl-glofas.py --data_path ~/NFS_era5/gff/


def parse_args(argv):
    parser = argparse.ArgumentParser("Download Glofas data at all ROIs.")

    parser.add_argument("--data_path", type=Path, help="GFF data folder")
    parser.add_argument("--max_back", type=int, default=20, help="maximum days to go back in time")
    parser.add_argument(
        "--first_sample", type=int, default=0, help="first sample to download GloFas data for"
    )
    parser.add_argument(
        "--last_sample", type=int, default=int(1e9), help="last sample to download GloFas data for"
    )

    return parser.parse_args(argv)


def convert(date_time):
    format = "%Y-%m-%d"
    return datetime.datetime.strptime(date_time, format)


def query(c, year, month, day, lon_min, lat_min, lon_max, lat_max, path):
    c.retrieve(
        "cems-glofas-historical",
        {
            "system_version": "version_4_0",
            "variable": [
                "river_discharge_in_the_last_24_hours",
                "runoff_water_equivalent",
                "soil_wetness_index",
            ],
            "format": "netcdf4.zip",
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "hyear": year,
            "hmonth": month,
            "hday": day,
            "area": [lat_max, lon_min, lat_min, lon_max],
        },
        path,
    )


def main(args):
    # Setup
    c = cdsapi.Client()
    glofas_path: Path = args.data_path / "glofas-daily"
    scratch_path = glofas_path / "scratch"
    glofas_path.mkdir(exist_ok=True)
    scratch_path.mkdir(exist_ok=True)

    # fetch meta data to download files for
    locations = gpd.read_file(args.data_path / "extras" / "locations.gpkg")

    # iterate through samples specified in meta file
    for row in tqdm.tqdm(locations[args.first_sample : args.last_sample].itertuples()):
        id = row.key
        date = convert(row.date)
        area = row.geometry
        out_path = glofas_path / f"{id}_{str(date).split(' ')[0]}.nc"

        # skip existing files
        if os.path.isfile(out_path):
            print(f"Data for sample ID {id} already exists. Skipping.")
            continue

        try:
            for date_back in range(0, args.max_back):
                updated = date - datetime.timedelta(days=date_back)
                y, m, d = (
                    updated.strftime("%Y"),
                    updated.strftime("%B").lower(),
                    updated.strftime("%d"),
                )

                # call to Copernicus CDS API
                dl_path = scratch_path / f"{id}_{date_back}.zip"
                query(c, y, m, d, *area.bounds, dl_path)

                # Extract zips
                with zipfile.ZipFile(dl_path) as f:
                    f.extractall(scratch_path)
                    (scratch_path / "data.nc").rename(scratch_path / f"{id}_{date_back}.nc")
                dl_path.unlink()

            # Combine all the netcdf files into a single one to save on extra indexes
            ds = xarray.open_mfdataset(
                str(scratch_path / f"{id}_*.nc"), concat_dim="time", combine="nested"
            ).load()
            ds.sortby(ds.time).to_netcdf(out_path)

            # Cleanup
            for old_file in scratch_path.glob(f"{id}_*.nc"):
                old_file.unlink()
        except Exception as e:
            print(f"Failed fetching data for sample ID {id} due to {e}")

    print("Done downloading GloFas data.")
    scratch_path.rmdir()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
    exit()
