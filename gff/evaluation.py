from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassF1Score
import tqdm

import gff.constants
import gff.data_sources
import gff.dataloaders
import gff.util


def get_tilewise_count(tile: torch.tensor, cls: int):
    count = (tile == cls).sum()
    return count


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    device = next(model.parameters()).device
    c = model.n_predict  # num predicted classes

    overall_f1 = MulticlassF1Score(c, average="macro", ignore_index=-100).to(device)
    overall_count = 0
    continent_f1 = {
        k: MulticlassF1Score(c, average="macro", ignore_index=-100).to(device)
        for k in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    clim_zone_f1 = {
        k: MulticlassF1Score(c, average="macro", ignore_index=-100).to(device)
        for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES
    }
    continent_counts = {k: 0 for k in gff.constants.HYDROATLAS_CONTINENT_NAMES}
    clim_zone_counts = {k: 0 for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES}
    tilewise_mse = MeanSquaredError().to(device)
    for example in tqdm.tqdm(dataloader, desc="Evaluating"):
        example = gff.util.recursive_todevice(example, device)
        targ = example.pop("floodmap")
        pred = model(example)  # [B, C, H, W]

        overall_f1(pred, targ[:, 0])
        pred_oh = pred.argmax(dim=1)
        for b, (pred_map, targ_map) in enumerate(zip(pred_oh, targ[:, 0])):
            continent = example["continent"][b]
            hydroatlas_fpath = example["fpaths"][b]["hydroatlas_basin"]
            clim_zone = gff.data_sources.get_climate_zone(
                hydroatlas_fpath, example["geom"][b], example["crs"][b]
            )
            continent_f1[continent](pred_map, targ_map)
            clim_zone_f1[clim_zone](pred_map, targ_map)
            continent_counts[continent] += 1
            clim_zone_counts[clim_zone] += 1
            overall_count += 1
            tilewise_mse(get_tilewise_count(pred_map, c - 1), get_tilewise_count(targ_map, c - 1))

    return {
        "f1": {
            "overall": overall_f1.compute().item(),
            "continent": {k: v.compute().item() for k, v in continent_f1.items()},
            "clim_zone": {k: v.compute().item() for k, v in clim_zone_f1.items()},
        },
        "tilewise_mse": tilewise_mse.compute().item(),
        "counts": {
            "overall": overall_count,
            "continent": continent_counts,
            "clim_zone": clim_zone_counts,
        },
    }


def evaluate_model_from_fnames(C: dict, data_folder: Path, model: nn.Module, fnames: list[str]):
    test_ds = gff.dataloaders.FloodForecastDataset(data_folder, C, meta_fnames=fnames)
    kwargs = {k: C[k] for k in ["batch_size", "num_workers"]}
    collate_fn = gff.dataloaders.sometimes_things_are_lists
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs, collate_fn=collate_fn)

    return evaluate_model(model, test_dl)
