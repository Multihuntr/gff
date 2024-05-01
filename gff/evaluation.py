from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics
import tqdm

import gff.constants
import gff.data_sources
import gff.dataloaders
import gff.util


def get_tilewise_count(tile: torch.tensor):
    count = (tile == gff.constants.KUROSIWO_FLOOD_CLASS).sum()
    return count


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    device = next(model.parameters()).device
    overall_f1 = torchmetrics.classification.MulticlassF1Score(3, average="macro").to(device)
    continent_f1 = {
        k: torchmetrics.classification.MulticlassF1Score(3, average="macro").to(device)
        for k in gff.constants.HYDROATLAS_CONTINENT_NAMES
    }
    clim_zone_f1 = {
        k: torchmetrics.classification.MulticlassF1Score(3, average="macro").to(device)
        for k in gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES
    }
    tilewise_mse = torchmetrics.MeanSquaredError().to(device)
    folder = Path(dataloader.dataset.folder).expanduser()
    for example in tqdm.tqdm(dataloader, desc="Evaluating"):
        example = gff.util.recursive_todevice(example, device)
        target = example.pop("floodmap")
        pred = model(example)  # [B, C, H, W]

        pred_oh = pred.argmax(dim=1)
        overall_f1(pred_oh, target[:, 0])
        for b, (pred_map, targ_map) in enumerate(zip(pred_oh, target[:, 0])):
            continent = example["continent"][b]
            clim_zone = gff.data_sources.get_climate_zone(folder, example["geom"][b])
            continent_f1[continent](pred_map, targ_map)
            clim_zone_f1[clim_zone](pred_map, targ_map)
            tilewise_mse(get_tilewise_count(pred_map), get_tilewise_count(targ_map))

    return {
        "overall": overall_f1.compute().item(),
        "continent": {k: v.compute().item() for k, v in continent_f1.items()},
        "clim_zone": {k: v.compute().item() for k, v in clim_zone_f1.items()},
    }
