import json
from pathlib import Path
import rasterio
import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import tqdm

import geopandas

import gff.constants
import gff.data_sources
import gff.util


def get_tilewise_count(tile: torch.tensor, cls_idx: int):
    count = (tile == cls_idx).sum()
    return count


def model_inference(model_folder: Path, model: nn.Module, dataloader: torch.utils.data.DataLoader):
    out_folder = model_folder / "inference"
    if out_folder.exists():
        return out_folder
    out_folder.mkdir()

    device = next(model.parameters()).device
    for example in tqdm.tqdm(dataloader, desc="Inferencing"):
        example = gff.util.recursive_todevice(example, device)
        pred = model(example)
        pred_oh = pred.argmax(dim=1).detach().cpu().numpy()

        for i, pred_map in enumerate(pred_oh):
            fmap_fpath = example["fpaths"][i]["floodmap"]
            out_fmap_fpath = out_folder / Path(fmap_fpath).name

            #
            if not out_fmap_fpath.exists():
                with rasterio.open(fmap_fpath) as lbl_tif:
                    profile = lbl_tif.profile
                with rasterio.open(out_fmap_fpath, "w", **profile) as out_tif:
                    pass

            with rasterio.open(out_fmap_fpath, "r+") as out_tif:
                window = gff.util.shapely_bounds_to_rasterio_window(
                    example["geom"][i].bounds, out_tif.transform
                )
                out_tif.write(pred_map[None], window=window)
    return out_folder


def evaluate_floodmaps(fnames: list[str], pred_path: Path, targ_path: Path, n_classes: int):
    overall_f1 = MulticlassF1Score(
        n_classes, average="none", ignore_index=-100, validate_args=False
    )
    overall_count = 0
    continent_f1 = {
        k: MulticlassF1Score(n_classes, average="none", ignore_index=-100, validate_args=False)
        for k in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    clim_zone_f1 = {
        k: MulticlassF1Score(n_classes, average="none", ignore_index=-100, validate_args=False)
        for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES
    }
    continent_counts = {k: 0 for k in gff.constants.HYDROATLAS_CONTINENT_NAMES}
    clim_zone_counts = {k: 0 for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
    tilewise_mse = MeanSquaredError()
    overall_cm = MulticlassConfusionMatrix(n_classes, ignore_index=-100, normalize="true")

    for meta_fname in tqdm.tqdm(fnames, "Files"):
        # Load data from meta: meta.json, tiles, continent
        meta_fpath = targ_path / meta_fname
        if not meta_fpath.exists():
            continue
        with open(meta_fpath) as f:
            meta = json.load(f)
        visit_tiles = geopandas.read_file(
            meta_fpath.parent / meta["visit_tiles"], engine="pyogrio", use_arrow=True
        )
        fmap_fname = Path(meta["floodmap"])
        hybas_key = "HYBAS_ID" if "HYBAS_ID" in meta else "HYBAS_ID_4"
        continent = int(str(meta[hybas_key])[0])

        # Get ready to read tiles
        hydroatlas_tif = rasterio.open(targ_path / f"{fmap_fname.stem}-hydroatlas.tif")
        pred_tif = rasterio.open(pred_path / fmap_fname)
        targ_tif = rasterio.open(targ_path / fmap_fname)
        for i, tile_row in tqdm.tqdm(list(visit_tiles.iterrows()), "Tiles", leave=False):
            # Read pred/targ tiles
            geom = tile_row.geometry
            window = gff.util.shapely_bounds_to_rasterio_window(geom.bounds, pred_tif.transform)
            pred_tile = torch.as_tensor(pred_tif.read(window=window))
            targ_tile = torch.as_tensor(targ_tif.read(window=window))
            targ_tile = gff.util.flatten_classes(targ_tile, n_classes)

            # Increment metrics
            overall_f1.update(pred_tile, targ_tile)
            overall_cm.update(pred_tile, targ_tile)
            clim_zone = gff.data_sources.get_climate_zone(hydroatlas_tif, geom, pred_tif.crs)
            continent_f1[continent].update(pred_tile, targ_tile)
            continent_counts[continent] += 1
            if clim_zone is not None:
                # This can happen if the hydroatlas raster doesn't quite reach the coast.
                # It's a border of like ~3 tiles that won't be counted.
                clim_zone_f1[clim_zone].update(pred_tile, targ_tile)
                clim_zone_counts[clim_zone] += 1
            overall_count += 1
            tilewise_mse.update(
                get_tilewise_count(pred_tile, n_classes - 1),
                get_tilewise_count(targ_tile, n_classes - 1),
            )
        hydroatlas_tif.close()
        pred_tif.close()
        targ_tif.close()

    return (
        {
            "f1": {
                "overall": tuple(f1v.item() for f1v in overall_f1.compute()),
                "continent": {
                    k: tuple(f1v.item() for f1v in v.compute()) for k, v in continent_f1.items()
                },
                "clim_zone": {
                    k: tuple(f1v.item() for f1v in v.compute()) for k, v in clim_zone_f1.items()
                },
            },
            "tilewise_mse": tilewise_mse.compute().item(),
            "counts": {
                "overall": overall_count,
                "continent": continent_counts,
                "clim_zone": clim_zone_counts,
            },
        },
        overall_cm,
    )
