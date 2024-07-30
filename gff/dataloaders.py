import datetime
import json
from pathlib import Path

import geopandas
import numpy as np
import pandas
import rasterio
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

        if C["half_res_context"]:
            c_res = 16
        else:
            c_res = 32

        self.bogus_geom = shapely.Point([124.225, 9.9725])
        if C["checkerboard"]:
            self.floodmaps = np.zeros((self.n_examples, 1, 224, 224), dtype=np.int64)
            self.floodmaps[:, :, :112, :112] = 1
            self.floodmaps[:, :, 112:, 112:] = 2
        else:
            self.floodmaps = np.random.randint(0, 3, size=(self.n_examples, 1, 224, 224))
        self.floodmaps = gff.util.flatten_classes(self.floodmaps, C["n_classes"])

        if "era5" in C["data_sources"]:
            n_era5 = len(C["era5_keys"])
            self.era5 = np.random.randn(self.n_examples, self.n_times, n_era5, c_res, c_res)
            self.era5 = self.era5.astype(np.float32)

        if "era5_land" in C["data_sources"]:
            n_era5_land = len(C["era5_land_keys"])
            self.era5_land = np.random.randn(
                self.n_examples, self.n_times, n_era5_land, c_res, c_res
            )
            self.era5_land = self.era5_land.astype(np.float32)

        if "hydroatlas_basin" in C["data_sources"]:
            n_hydroatlas = len(C["hydroatlas_keys"])
            self.hydroatlas_basin = np.random.randn(self.n_examples, n_hydroatlas, c_res, c_res)
            self.hydroatlas_basin = self.hydroatlas_basin.astype(np.float32)

        if "dem_context" in C["data_sources"]:
            self.dem_context = np.random.randn(self.n_examples, 1, c_res, c_res)
            self.dem_context = self.dem_context.astype(np.float32)
        if "dem_local" in C["data_sources"]:
            self.dem_local = np.random.randn(self.n_examples, 1, 224, 224)
            self.dem_local = self.dem_local.astype(np.float32)
        if "hand" in C["data_sources"]:
            self.hand = np.random.randn(self.n_examples, 1, 224, 224)
            self.hand = self.hand.astype(np.float32)
        if "s1" in C["data_sources"]:
            self.s1 = np.random.randn(self.n_examples, 2, 224, 224)
            self.s1 = self.s1.astype(np.float32)
            self.s1_lead_days = np.random.randint(0, 60, (self.n_examples,))
            self.s1_lead_days = self.s1_lead_days.astype(np.int64)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        raster_data = {k: getattr(self, k)[idx] for k in self.data_sources}
        example = {
            "floodmap": self.floodmaps[idx],
            "continent": 1,
            "geom": self.bogus_geom,
            "context_geom": self.bogus_geom,
            **raster_data,
        }
        if "s1" in example:
            example["s1_lead_days"] = self.s1_lead_days[idx]
        return example


class FloodForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: Path,
        C: dict,
        meta_fnames: list[str],
    ):
        self.folder = folder
        self.C = C
        self.floodmap_path = self.folder / "rois"
        self.meta_fnames = sorted(meta_fnames)
        self.metas = self.load_tile_metas()
        self.tiles = self.load_tile_geoms(self.metas)
        self.hydroatlas_band_idxs = None
        self.dem_nodata = None

    def mk_context_geom(self, geom: shapely.Geometry):
        if self.C["half_res_context"]:
            degrees = gff.constants.CONTEXT_DEGREES / 2
        else:
            degrees = gff.constants.CONTEXT_DEGREES
        w = h = degrees
        lon, lat = shapely.get_coordinates(geom.centroid)[0]
        lonlo, latlo = lon - w / 2, lat + h / 2
        lonhi, lathi = lon + w / 2, lat - h / 2
        return shapely.box(lonlo, latlo, lonhi, lathi)

    def load_tile_metas(self):
        metas = []
        for meta_fname in self.meta_fnames:
            # Some generated floodmaps mightn't be available due to not having ERA5
            meta_fpath = self.floodmap_path / meta_fname
            if meta_fpath.exists():
                with open(meta_fpath) as f:
                    meta = json.load(f)
                metas.append(meta)
        return metas

    def load_tile_geoms(self, metas: list[dict]):
        tiles = []
        for meta in metas:
            # Get tiles
            p = self.floodmap_path / meta["visit_tiles"]
            visit_tiles = geopandas.read_file(p, use_arrow=True, engine="pyogrio")

            # Extract geometries
            # Local could be any CRS, but context is always EPSG:4326 (to match ERA5)
            geoms = list(visit_tiles.geometry)
            geoms_4326 = list(visit_tiles.to_crs("EPSG:4326").geometry)
            context_geoms = [self.mk_context_geom(geom) for geom in geoms_4326]

            # Create a simple list of tuples; one per example tile
            meta_parallel = [meta] * len(geoms)
            crs_parallel = [visit_tiles.crs] * len(geoms)
            pairs = list(zip(meta_parallel, crs_parallel, geoms, context_geoms))
            tiles.extend(pairs)
        return tiles

    def get_s1_lead_days(self, meta):
        pre1_date = datetime.datetime.fromisoformat(meta["pre1_date"])
        post_date = datetime.datetime.fromisoformat(meta["post_date"])
        return (post_date - pre1_date).days

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        meta, crs, geom, context_geom = self.tiles[idx]
        floodmap_path = self.floodmap_path / meta["floodmap"]
        hybas_key = "HYBAS_ID" if "HYBAS_ID" in meta else "HYBAS_ID_4"
        continent = int(str(meta[hybas_key])[0])
        if self.C["half_res_context"]:
            context_res = (gff.constants.CONTEXT_RESOLUTION // 2,) * 2
        else:
            context_res = (gff.constants.CONTEXT_RESOLUTION,) * 2

        cache_local = self.C["cache_local_in_ram"]

        targ = gff.util.get_tile(floodmap_path, geom.bounds, align=True, cache=cache_local)
        targ = targ.astype(np.int64)
        targ = gff.util.flatten_classes(targ, self.C["n_classes"])
        result = {
            "floodmap": targ,
            "continent": continent,
            "crs": crs,
            "geom": geom,
            "context_geom": context_geom,
            "s1_lead_days": self.get_s1_lead_days(meta),
            "fpaths": {"floodmap": floodmap_path},
        }

        # Add in various data sources
        # Weather data
        future_name = gff.constants.KUROSIWO_S1_NAMES[-1]
        weather_end = datetime.datetime.fromisoformat(meta[f"{future_name}_date"].rstrip("Z"))
        weather_start = weather_end - datetime.timedelta(days=(self.C["weather_window"] - 1))
        if "era5_land" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-era5-land.tif")
            data = gff.data_sources.load_exported_era5(
                fpath,
                context_geom,
                context_res,
                weather_start,
                weather_end,
                keys=self.C["era5_land_keys"],
                cache_in_ram=self.C["cache_context_in_ram"],
            )
            result["era5_land"] = np.array(data, dtype=np.float32)
            result["fpaths"]["era5_land"] = fpath
        if "era5" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-era5.tif")
            data = gff.data_sources.load_exported_era5(
                fpath,
                context_geom,
                context_res,
                weather_start,
                weather_end,
                keys=self.C["era5_keys"],
                cache_in_ram=self.C["cache_context_in_ram"],
            )
            result["era5"] = np.array(data, dtype=np.float32)
            result["fpaths"]["era5"] = fpath
        if "glofas" in self.C["data_sources"]:
            d_str = weather_end.strftime("%Y-%m-%d")
            fpath = self.floodmap_path / f"{meta['key']}_{d_str}.nc"
            data = gff.data_sources.load_glofas(
                fpath,
                context_geom,
                context_res,
                weather_start,
                weather_end,
                bands=self.C["glofas_keys"],
                cache_in_ram=self.C["cache_context_in_ram"],
            )
            result["glofas"] = np.array(data, dtype=np.float32)
            result["fpaths"]["glofas"] = fpath

        # (Relatively) static soil attributes
        if "hydroatlas_basin" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-hydroatlas.tif")
            if self.hydroatlas_band_idxs is None:
                with rasterio.open(fpath) as tif:
                    # NOTE: only works because the band idxs are identical across hydroatlas rasters.
                    self.hydroatlas_band_idxs = tuple(
                        [1 + tif.descriptions.index(k) for k in self.C["hydroatlas_keys"]]
                    )
            data = gff.data_sources.load_pregenerated_raster(
                fpath,
                context_geom,
                context_res,
                self.hydroatlas_band_idxs,
                self.C["cache_context_in_ram"],
            )
            result["hydroatlas_basin"] = np.array(data, dtype=np.float32)
            result["fpaths"]["hydroatlas_basin"] = fpath

        # DEM at coarse and fine scales
        if "dem_context" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-dem-context.tif")
            data = gff.data_sources.load_pregenerated_raster(
                fpath,
                context_geom,
                context_res,
                (1,),
                self.C["cache_context_in_ram"],
            )
            if self.dem_nodata is None:
                with rasterio.open(fpath) as tif:
                    self.dem_nodata = tif.nodata
            nan_mask = data == self.dem_nodata
            result["dem_context"] = np.array(data, dtype=np.float32)
            result["dem_context"][nan_mask] = np.nan
            result["fpaths"]["dem_context"] = fpath
        if "dem_local" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-dem-local.tif")
            data = gff.util.get_tile(fpath, geom.bounds, align=True, cache=cache_local)
            result["dem_local"] = np.array(data, dtype=np.float32)
            result["fpaths"]["dem_local"] = fpath
        if "hand" in self.C["data_sources"]:
            fpath = floodmap_path.with_name(floodmap_path.stem + "-hand.tif")
            data = gff.util.get_tile(fpath, geom.bounds, align=True, cache=cache_local)
            result["hand"] = np.array(data, dtype=np.float32)
            result["fpaths"]["hand"] = fpath

        # Sentinel 1 (assumed to be a proxy for soil moisture)
        if "s1" in self.C["data_sources"]:
            s1_stem = gff.util.get_s1_stem_from_meta(meta)
            s1_path = self.floodmap_path / f"{s1_stem}-s1.tif"
            result["s1"] = gff.util.get_tile(s1_path, geom.bounds, align=True, cache=cache_local)
            result["fpaths"]["s1"] = fpath
            if np.isnan(result["s1"]).sum() > 0:
                raise Exception("NO! It can't be!")

        return result


def sometimes_things_are_lists(
    original_batch, as_list=["continent", "crs", "geom", "context_geom", "fpaths"]
):
    """A custom collation function which lets things be lists instead of tensors"""
    keys = list(original_batch[0].keys())
    result = {key: [] for key in keys}
    for item in original_batch:
        for key, value in item.items():
            if key in as_list:
                result[key].append(value)
            else:
                result[key].append(torch.tensor(np.array(value)))
    for key in keys:
        if key not in as_list:
            result[key] = torch.stack(result[key])
    return result


def read_partitions(folder: Path, fold: int):
    test_partition = fold
    val_partition = (test_partition + 1) % gff.constants.N_PARTITIONS
    val_partition_fname = f"floodmap_partition_{val_partition}.txt"
    test_partition_fname = f"floodmap_partition_{test_partition}.txt"
    train_fnames = []
    val_fnames = []
    test_fnames = []
    for fpath in list((folder / "partitions").glob("floodmap_partition_?.txt")):
        fnames = pandas.read_csv(fpath, header=None)[0].values.tolist()
        if fpath.name == test_partition_fname:
            test_fnames.extend(fnames)
        elif fpath.name == val_partition_fname:
            val_fnames.extend(fnames)
        else:
            train_fnames.extend(fnames)

    return train_fnames, val_fnames, test_fnames


def create_all(C, generator):
    data_folder = Path(C["data_folder"]).expanduser()
    if C["dataset"] == "debug_dataset":
        train_ds = DebugFloodForecastDataset(data_folder, C)
        val_ds = DebugFloodForecastDataset(data_folder, C)
        test_ds = DebugFloodForecastDataset(data_folder, C)
    elif C["dataset"] == "forecast_dataset":
        train_fnames, val_fnames, test_fnames = read_partitions(data_folder, C["fold"])
        train_ds = FloodForecastDataset(data_folder, C, meta_fnames=train_fnames)
        val_ds = FloodForecastDataset(data_folder, C, meta_fnames=val_fnames)
        test_ds = FloodForecastDataset(data_folder, C, meta_fnames=test_fnames)
    else:
        raise NotImplementedError(f"Dataset {C['dataset']} not supported")

    kwargs = {k: C[k] for k in ["batch_size", "num_workers"]}
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        **kwargs,
        generator=generator,
        collate_fn=sometimes_things_are_lists,
    )
    val_dl = torch.utils.data.DataLoader(val_ds, **kwargs, collate_fn=sometimes_things_are_lists)
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs, collate_fn=sometimes_things_are_lists)

    return train_dl, val_dl, test_dl


def create_test(C):
    data_folder = Path(C["data_folder"]).expanduser()
    if C["dataset"] == "debug_dataset":
        test_ds = DebugFloodForecastDataset(data_folder, C)
    elif C["dataset"] == "forecast_dataset":
        _, _, test_fnames = read_partitions(data_folder, C["fold"])
        test_ds = FloodForecastDataset(data_folder, C, meta_fnames=test_fnames)
    else:
        raise NotImplementedError(f"Dataset {C['dataset']} not supported")

    kwargs = {k: C[k] for k in ["batch_size", "num_workers"]}
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs, collate_fn=sometimes_things_are_lists)

    return test_dl
