import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchmetrics
import torchmetrics.classification
import tqdm

import gff.util


def nans_to_min(r: np.ndarray):
    if len(r.shape) == 4:
        channelwise_min = np.nanmin(r, axis=(0, 2, 3))[None, :, None]
        r[:, :, *np.any(np.isnan(r), axis=(0, 1)).nonzero()] = channelwise_min
    else:
        channelwise_min = np.nanmin(r, axis=(1, 2))[:, None]
        r[:, *np.any(np.isnan(r), axis=0).nonzero()] = channelwise_min


def rescale_channelwise(r: np.ndarray):
    if len(r.shape) == 4:
        r_min = r.min(axis=(0, 2, 3), keepdims=True)
        r_max = r.max(axis=(0, 2, 3), keepdims=True)
    else:
        r_min = r.min(axis=(1, 2), keepdims=True)
        r_max = r.max(axis=(1, 2), keepdims=True)
    return (r - r_min) / (r_max - r_min)


def cls_to_rgb(arr: np.ndarray):
    colours = np.array([(18, 19, 19), (60, 125, 255), (255, 64, 93)]) / 256
    out_arr = np.zeros((*arr.shape, 3))
    for cls_idx in range(3):
        out_arr[(arr == cls_idx)] = colours[cls_idx][None]
    return out_arr


class CustomWriter:
    """A wrapper for the default writer that handles data wrangling for this project"""

    def __init__(self, writer):
        self.writer = writer
        self.count = 0

    def step(self):
        self.count += 1

    def write_imgs(self, ex: dict, pred: torch.Tensor, targ: torch.Tensor):
        # Pred vs target
        most_water_ex = ((targ == 1) | (targ == 2)).sum(axis=(1, 2, 3)).argmax()
        pred_cls = pred[most_water_ex].argmax(dim=0).numpy()
        pred_rgb = cls_to_rgb(pred_cls)
        targ_cls = targ[most_water_ex, 0].numpy()
        targ_rgb = cls_to_rgb(targ_cls)
        pred_v_targ = np.stack([pred_rgb, targ_rgb])
        self.writer.add_images("pred_vs_targ", pred_v_targ, self.count, dataformats="NHWC")

        context_rasters = []
        for k in ["era5", "era5_land"]:
            if k in ex:
                r = ex[k][0][:, :3].numpy()
                nans_to_min(r)
                r = rescale_channelwise(r)
                context_rasters.extend(r)
        if "hydroatlas_basin" in ex:
            r = ex["hydroatlas_basin"][0][:3].numpy()
            nans_to_min(r)
            r = rescale_channelwise(r)
            context_rasters.append(r)
        if "dem_context" in ex:
            r = ex["dem_context"][0].repeat((3, 1, 1)).numpy()
            nans_to_min(r)
            r = rescale_channelwise(r)
            context_rasters.append(r)
        self.writer.add_images("context", np.stack(context_rasters), self.count)

        local_rasters = []
        if "dem_local" in ex:
            r = ex["dem_local"][0].repeat((3, 1, 1)).numpy()
            nans_to_min(r)
            r = rescale_channelwise(r)
            local_rasters.append(r)
        if "s1" in ex:
            r = ex["s1"][0].numpy()
            r = rescale_channelwise(r)
            r = np.stack([*r, np.zeros_like(r[0])], axis=0)
            local_rasters.append(r)
        self.writer.add_images("local", np.stack(local_rasters), self.count)

    def write_scalars(self, loss: float, f1: float):
        self.writer.add_scalar("Loss/train_batch", loss, self.count)
        self.writer.add_scalar("F1/train_batch", f1, self.count)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    n_classes: int,
    optim: torch.optim.Optimizer,
    custom_writer: CustomWriter,
    scalar_freq: int,
    img_freq: int,
):
    model.train()
    f1_all = torchmetrics.classification.MulticlassF1Score(
        n_classes, average="macro", ignore_index=-100
    )
    f1_prog = torchmetrics.classification.MulticlassF1Score(
        n_classes, average="macro", ignore_index=-100
    )
    loss_all = torchmetrics.MeanMetric()
    loss_prog = torchmetrics.MeanMetric()
    # Move everything to whatever device the model is on
    device = next(model.parameters()).device
    for thing in [f1_all, f1_prog, loss_all, loss_prog, criterion]:
        thing.to(device)
    for i, example_cpu in enumerate(tqdm.tqdm(dataloader, desc="Training", leave=False)):
        example = gff.util.recursive_todevice(example_cpu, device)
        targ = example.pop("floodmap")
        pred = model(example)
        targ[targ > 2] = -100  # To make criterion ignore weird numbers from Kuro Siwo

        if n_classes == 2:
            # Flatten target labels to just no-water/water
            targ[targ == 2] = 1

        optim.zero_grad()
        loss = criterion(pred, targ[:, 0])
        loss.backward()
        optim.step()

        f1_all(pred, targ[:, 0])
        f1_prog(pred, targ[:, 0])
        loss_all(loss)
        loss_prog(loss)
        if (i % scalar_freq) == 0 and i != 0:
            this_loss = loss_prog.compute().cpu().item()
            this_f1 = f1_prog.compute().cpu().item()
            custom_writer.write_scalars(this_loss, this_f1)
            loss_prog.reset()
            f1_prog.reset()
        if (i % img_freq) == 0 and i != 0:
            custom_writer.write_imgs(example_cpu, pred.detach().cpu(), targ.detach().cpu())
        custom_writer.step()
    return loss_all.compute().cpu().item(), f1_all.compute().cpu().item()


def val_epoch(model, dataloader, criterion, n_classes, limit: int = None):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        f1 = torchmetrics.classification.MulticlassF1Score(
            n_classes, average="macro", ignore_index=-100
        )
        f1.to(device)
        loss_all = torchmetrics.MeanMetric()
        loss_all.to(device)
        total_steps = len(dataloader) if limit is None else limit
        for i, example in enumerate(
            tqdm.tqdm(dataloader, desc="Val-ing", leave=False, total=total_steps)
        ):
            example = gff.util.recursive_todevice(example, device)
            targ = example.pop("floodmap")
            pred = model(example)
            targ[targ > 2] = -100

            if n_classes == 2:
                # Flatten target labels to just no-water/water
                targ[targ == 2] = 1

            loss = criterion(pred, targ[:, 0])

            f1(pred, targ[:, 0])
            loss_all(loss)
            if limit is not None and i >= limit:
                break
    return loss_all.compute().cpu().item(), f1.compute().cpu().item()


def training_loop(C, model_folder, model: nn.Module, dataloaders, checkpoint=None):
    assert C["n_classes"] in [2, 3], "Num classes must be 2 (no-water/) or 3 (bg/perm.water/flood)"

    optim = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda epoch: C["lr_decay"])
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    cls_weight = torch.tensor([0.5, 2, 2])[: C["n_classes"]]
    criterion = nn.CrossEntropyLoss(weight=cls_weight, ignore_index=-100)

    writer = torch.utils.tensorboard.SummaryWriter(model_folder)
    custom_writer = CustomWriter(writer)

    def do_save(epoch):
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            model_folder / f"checkpoint_{epoch:03d}.th",
        )

    train_dl, val_dl = dataloaders
    hoisted_epoch = start_epoch
    try:
        for epoch in tqdm.tqdm(range(start_epoch, C["epochs"]), desc="Training epochs"):
            hoisted_epoch = epoch
            train_loss, train_f1 = train_epoch(
                model,
                train_dl,
                criterion,
                C["n_classes"],
                optim,
                custom_writer,
                scalar_freq=C["scalar_freq"],
                img_freq=C["img_freq"],
            )
            val_loss, val_f1 = val_epoch(model, val_dl, criterion, C["n_classes"], limit=256)

            writer.add_scalar("Loss/train_epoch", train_loss, custom_writer.count)
            writer.add_scalar("F1/train_epoch", train_f1, custom_writer.count)
            writer.add_scalar("Loss/val", val_loss, custom_writer.count)
            writer.add_scalar("F1/val", val_f1, custom_writer.count)

            scheduler.step()
    finally:
        writer.close()
        do_save(hoisted_epoch)

    return model
