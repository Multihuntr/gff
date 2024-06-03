import json
from pathlib import Path
from matplotlib import pyplot as plt
import rasterio
import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import tqdm

import geopandas
import yaml

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


def flatten_classes(n_classes, pred_tile, targ_tile, *args, **kwargs):
    pred_tile = gff.util.flatten_classes(pred_tile, n_classes)
    targ_tile = gff.util.flatten_classes(targ_tile, n_classes)
    return pred_tile, targ_tile


def processing_blockout_fnc(cache_folder, block):
    def blockout_inner(n_classes, pred_tile, targ_tile, fmap_fname, window):
        if n_classes == 2 and block == "kurosiwo-pw":
            mask = targ_tile == gff.constants.KUROSIWO_PW_CLASS
            pred_tile[mask] = -100
            targ_tile[mask] = -100
        elif n_classes == 2 and block == "worldcover-water":
            worldcover_tif_fpath = cache_folder / f"{fmap_fname.stem}-worldcover.tif"
            tif = gff.util.tif_data_ram(worldcover_tif_fpath)
            cover = tif.read(window=window)
            # Mask permanent water
            mask = (cover == gff.constants.WORLDCOVER_PW_CLASS)[None]
            pred_tile[mask] = -100
            targ_tile[mask] = -100
        pred_tile = gff.util.flatten_classes(pred_tile, n_classes)
        targ_tile = gff.util.flatten_classes(targ_tile, n_classes)
        return pred_tile, targ_tile

    return blockout_inner


def evaluate_floodmaps(
    fnames: list[str],
    pred_path: Path,
    targ_path: Path,
    n_classes: int,
    coast_masks: dict[str, list[bool]] = None,
    extra_processing: callable = flatten_classes,
    device: str = "cpu",  # Torchmetrics is slow; cuda speeds it up dramatically
):
    m = make_metrics(n_classes, device)
    overall_count = 0

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

        # Open files
        hydroatlas_tif = gff.util.tif_data_ram(targ_path / f"{fmap_fname.stem}-hydroatlas.tif")
        pred_tif = gff.util.tif_data_ram(pred_path / fmap_fname)
        targ_tif = gff.util.tif_data_ram(targ_path / fmap_fname)
        for i, tile_row in tqdm.tqdm(list(visit_tiles.iterrows()), "Tiles", leave=False):
            # Read pred/targ tiles
            geom = tile_row.geometry
            window = gff.util.shapely_bounds_to_rasterio_window(geom.bounds, pred_tif.transform)
            pred_tile = torch.as_tensor(pred_tif.read(window=window)).to(device)
            targ_tile = torch.as_tensor(targ_tif.read(window=window)).to(device)
            pred_tile, targ_tile = extra_processing(
                n_classes, pred_tile, targ_tile, fmap_fname, window
            )

            # Update metrics
            clim_zone = gff.data_sources.get_climate_zone(hydroatlas_tif, geom, pred_tif.crs)
            overall_count += 1
            if coast_masks is None:
                coast_mask = None
            else:
                coast_mask = coast_masks[meta_fname][i]
            update_metrics(m, pred_tile, targ_tile, n_classes, continent, clim_zone, coast_mask)

    return (
        compute_metrics(m, overall_count),
        m["overall_cm"],
    )


def make_metrics(n_classes, device):
    return {
        "overall_f1": MulticlassF1Score(
            n_classes, average="none", ignore_index=-100, validate_args=False
        ).to(device),
        "overall_count": 0,
        "continent_f1": {
            k: MulticlassF1Score(
                n_classes, average="none", ignore_index=-100, validate_args=False
            ).to(device)
            for k in gff.constants.HYDROATLAS_CONTINENT_NAMES
        },
        "clim_zone_f1": {
            k: MulticlassF1Score(
                n_classes, average="none", ignore_index=-100, validate_args=False
            ).to(device)
            for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES
        },
        "continent_counts": {k: 0 for k in gff.constants.HYDROATLAS_CONTINENT_NAMES},
        "clim_zone_counts": {k: 0 for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES},
        "tilewise_mse": MeanSquaredError().to(device),
        "overall_cm": MulticlassConfusionMatrix(
            n_classes, ignore_index=-100, normalize="true", validate_args=False
        ).to(device),
        "coast": MulticlassF1Score(
            n_classes, average="none", ignore_index=-100, validate_args=False
        ).to(device),
        "inland": MulticlassF1Score(
            n_classes, average="none", ignore_index=-100, validate_args=False
        ).to(device),
        "coast_count": 0,
    }


def update_metrics(m, pred, targ, n_classes, continent, clim_zone, coast_mask):
    m["overall_f1"].update(pred, targ)
    m["overall_cm"].update(pred, targ)
    m["continent_f1"][continent].update(pred, targ)
    m["continent_counts"][continent] += 1
    if clim_zone is not None:
        # This can happen if the hydroatlas raster doesn't quite reach the coast.
        # It's a border of like ~3 tiles that won't be counted.
        m["clim_zone_f1"][clim_zone].update(pred, targ)
        m["clim_zone_counts"][clim_zone] += 1
    m["tilewise_mse"].update(
        get_tilewise_count(pred, n_classes - 1),
        get_tilewise_count(targ, n_classes - 1),
    )
    if coast_mask is not None:
        if coast_mask:
            m["coast"].update(pred, targ)
            m["coast_count"] += 1
        else:
            m["inland"].update(pred, targ)


def compute_metrics(m, overall_count):
    return {
        "f1": {
            "overall": tuple(f1v.item() for f1v in m["overall_f1"].compute()),
            "continent": {
                k: tuple(f1v.item() for f1v in v.compute()) for k, v in m["continent_f1"].items()
            },
            "clim_zone": {
                k: tuple(f1v.item() for f1v in v.compute()) for k, v in m["clim_zone_f1"].items()
            },
            "coast": tuple(f1v.item() for f1v in m["coast"].compute()),
            "inland": tuple(f1v.item() for f1v in m["inland"].compute()),
        },
        "tilewise_mse": m["tilewise_mse"].compute().item(),
        "counts": {
            "overall": overall_count,
            "continent": m["continent_counts"],
            "clim_zone": m["clim_zone_counts"],
            "coast": m["coast_count"],
        },
    }


def save_results(fpath, results):
    with open(fpath, "w") as f:
        yaml.safe_dump(results, f)


def save_cm(cm, n_cls, title, fpath):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    cm.plot(ax=ax, labels=gff.constants.KUROSIWO_CLASS_NAMES[:n_cls])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fpath.with_suffix(".png"))
    fig.savefig(fpath.with_suffix(".eps"))
    plt.close(fig)
