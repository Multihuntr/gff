import datetime
import json
from pathlib import Path

import geopandas
import numpy as np
import pandas
import shapely
import torch
import torch.nn
import torch.utils.data

import gff.constants
import gff.data_sources
import gff.util


class DebugFloodForecastDataset(torch.utils.data.Dataset):
    def __init__(self, folder, C):
        self.folder = folder
        self.n_examples = C["n_examples"]
        self.n_times = C["n_times"]
        self.data_sources = C["data_sources"]

        self.bogus_geom = shapely.Point([124.225, 9.9725])
        # self.floodmaps = np.random.randint(0, 3, size=(self.n_examples, 1, 224, 224))
        self.floodmaps = np.zeros((self.n_examples, 1, 224, 224), dtype=np.int64)
        self.floodmaps[:, :, :112, :112] = 1
        self.floodmaps[:, :, 112:, 112:] = 2

        if "era5" in C["data_sources"]:
            n_era5 = len(C["era5_keys"])
            self.era5 = np.random.randn(self.n_examples, self.n_times, n_era5, 32, 32)
            self.era5 = self.era5.astype(np.float32)

        if "era5_land" in C["data_sources"]:
            n_era5_land = len(C["era5_land_keys"])
            self.era5_land = np.random.randn(self.n_examples, self.n_times, n_era5_land, 32, 32)
            self.era5_land = self.era5_land.astype(np.float32)

        if "hydroatlas_basin" in C["data_sources"]:
            n_hydroatlas = len(C["hydroatlas_keys"])
            self.hydroatlas_basin = np.random.randn(self.n_examples, n_hydroatlas, 32, 32)
            self.hydroatlas_basin = self.hydroatlas_basin.astype(np.float32)

        if "dem_context" in C["data_sources"]:
            self.dem_context = np.random.randn(self.n_examples, 1, 32, 32)
            self.dem_context = self.dem_context.astype(np.float32)
        if "dem_local" in C["data_sources"]:
            self.dem_local = np.random.randn(self.n_examples, 1, 224, 224)
            self.dem_local = self.dem_local.astype(np.float32)
        if "s1" in C["data_sources"]:
            self.s1 = np.random.randn(self.n_examples, 2, 224, 224)
            self.s1 = self.s1.astype(np.float32)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        raster_data = {k: getattr(self, k)[idx] for k in self.data_sources}
        return {
            "floodmap": self.floodmaps[idx],
            "continent": 1,
            "geom": self.bogus_geom,
            "context_geom": self.bogus_geom,
            **raster_data,
        }


class FloodForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: Path,
        ext_folders: list[Path],
        C: dict,
        is_train: str,
        split: int,
    ):
        self.folder = folder
        self.ext_folders = ext_folders
        self.C = C
        self.floodmap_path = self.folder / "floodmaps" / self.C["floodmap"]
        self.is_train = is_train
        self.split = split
        self.incl_list = self.read_partition()
        self.metas = self.load_tile_metas()
        self.tiles = self.load_tile_geoms(self.metas)

    def read_partition(self):
        # Split index defines the partition index to use as test.
        test_partition_fname = f"floodmap_partition_{self.split}.txt"
        partition_paths = list((self.folder / "partitions").glob("floodmap_partition_?.txt"))
        self.floodmap_fnames = []
        for fpath in partition_paths:
            partition_is_train = fpath.name != test_partition_fname
            if self.is_train == partition_is_train:
                fnames = pandas.read_csv(fpath, header=None)[0].values.tolist()
                self.floodmap_fnames.extend(fnames)

    def mk_context_geom(self, geom: shapely.Geometry):
        w = h = gff.constants.CONTEXT_DEGREES
        lon, lat = geom.centroid
        lonlo, latlo = lon - w / 2, lat + h / 2
        lonhi, lathi = lon + w / 2, lat - h / 2
        return shapely.box(lonlo, latlo, lonhi, lathi)

    def load_tile_metas(self):
        metas = []
        for meta_path in self.floodmap_fnames:
            with open(meta_path) as f:
                meta = json.load(f)
            metas.append(meta)

    def load_tile_geoms(self, metas: list[dict]):
        tiles = []
        for meta in metas:
            # Get tiles
            p = self.floodmap_path / meta["visit_tiles"]
            visit_tiles = geopandas.read_file(p, use_arrow=True, engine="pyogrio")

            # Extract geometries
            geoms = list(visit_tiles.geometry)
            context_geoms = [self.mk_context_geom(geom) for geom in geoms]

            # Create a simple list of tuples; one per example tile
            meta_parallel = [meta] * len(geoms)
            crs_parallel = [visit_tiles.crs] * len(geoms)
            pairs = list(zip(meta_parallel, crs_parallel, geoms, context_geoms))
            tiles.extend(pairs)

    def get_s1_path(self, meta):
        if self.s1_exported:
            s1_folder = "s1-export"
        else:
            s1_folder = "s1"

        date_str = datetime.datetime.fromisoformat(meta["pre1_date"]).strftime("%Y-%m-%d")
        if meta["type"] == "kurosiwo":
            s1_fname = f"{meta['info']['actiid']}-{meta['info']['aoiid']}-{date_str}.tif"
        elif meta["type"] == "generated":
            s1_fname = f"{meta['FLOOD']}-{meta['HYBAS_ID']}-{date_str}.tif"

        return self.folder / s1_folder / s1_fname

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        meta, crs, geom, context_geom = self.tiles[idx]
        floodmap_path = self.floodmap_path / meta["floomaps"]
        s1_path = self.get_s1_path(meta)
        hybas_key = "HYBAS_ID" if "HYBAS_ID" in meta else "HYBAS_ID_4"
        continent = int(str(meta[hybas_key])[0])
        result = {
            "floodmap": gff.util.get_tile(floodmap_path, geom),
            "continent": continent,
            "geom": geom,
            "context_geom": context_geom,
        }

        # Add in various data sources
        # Weather data
        last_name = gff.constants.KUROSIWO_S1_NAMES[-2]
        next_name = gff.constants.KUROSIWO_S1_NAMES[-1]
        weather_start = datetime.datetime.fromisoformat(meta[f"{last_name}_date"])
        weather_end = datetime.datetime.fromisoformat(meta[f"{next_name}_date"])
        if "era5_land" in self.C["data_sources"]:
            result["era5_land"] = gff.data_sources.load_era5(
                self.ext_folders["era5_land"],
                context_geom,
                gff.constants.CONTEXT_RESOLUTION,
                weather_start,
                weather_end,
                era5_land=True,
            )
        if "era5" in self.C["data_sources"]:
            result["era5"] = gff.data_sources.load_era5(
                self.ext_folders["era5"],
                context_geom,
                gff.constants.CONTEXT_RESOLUTION,
                weather_start,
                weather_end,
                era5_land=False,
            )

        # (Relatively) static soil attributes
        if "hydroatlas_basin" in self.C["data_sources"]:
            fpath = self.folder / gff.constants.HYDROATLAS_RASTER_FNAME
            result["hydroatlas_basin"] = gff.data_sources.load_pregenerated_raster(
                fpath, context_geom, gff.constants.CONTEXT_RESOLUTION, keys=self.hydroatlas_keys
            )

        # DEM at coarse and fine scales
        if "dem_context" in self.C["data_sources"]:
            fpath = self.folder / gff.constants.COARSE_DEM_FNAME
            result["dem_context"] = gff.data_sources.load_pregenerated_raster(
                fpath, context_geom, gff.constants.CONTEXT_RESOLUTION
            )
        if "dem_local" in self.C["data_sources"]:
            dem = gff.data_sources.get_dem(geom, crs, self.folder)
            result["dem_local"] = gff.util.resample_xr(
                dem, geom, gff.constants.CONTEXT_RESOLUTION, method="linear"
            )

        # Sentinel 1 (assumed to be a proxy for soil moisture)
        if "s1" in self.C["data_sources"]:
            result["s1"] = gff.util.get_tile(s1_path, geom)

        return result


def custom_collate_fn(original_batch, as_list=["continent", "geom", "context_geom"]):
    keys = list(original_batch[0].keys())
    result = {key: [] for key in keys}
    for item in original_batch:
        for key, value in item.items():
            if key in as_list:
                result[key].append(value)
            else:
                result[key].append(torch.tensor(value))
    for key in keys:
        if key not in as_list:
            result[key] = torch.stack(result[key])
    return result


def create(C, generator):
    if C["dataset"] == "debug_dataset":
        train_ds = DebugFloodForecastDataset(C["data_folder"], C)
        test_ds = DebugFloodForecastDataset(C["data_folder"], C)
    elif C["dataset"] == "forecast_dataset":
        train_ds = FloodForecastDataset(
            C["data_folder"], C["ext_folders"], C, is_train=True, split=C["test_partition"]
        )
        test_ds = FloodForecastDataset(
            C["data_folder"], C["ext_folders"], C, is_train=False, split=C["test_partition"]
        )
    else:
        raise NotImplementedError(f"Dataset {C['dataset']} not supported")

    kwargs = {k: C[k] for k in ["batch_size", "num_workers"]}
    train_dl = torch.utils.data.DataLoader(
        train_ds, shuffle=True, **kwargs, generator=generator, collate_fn=custom_collate_fn
    )
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs, collate_fn=custom_collate_fn)

    return train_dl, test_dl
