from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchmetrics
import torchmetrics.classification
import tqdm

import gff.constants
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
        prop_water = ((targ == 1) | (targ == 2)).sum(axis=(1, 2, 3)) / targ[0].numel()
        half_water_ex = np.abs(prop_water - 0.5).argmin()
        pred_cls = pred[half_water_ex].argmax(dim=0).numpy()
        pred_rgb = cls_to_rgb(pred_cls)
        targ_cls = targ[half_water_ex, 0].numpy()
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

    def write_scalars(self, loss: float, avg_f1: float, f1s: list[float]):
        self.writer.add_scalar("Loss/train_batch", loss, self.count)
        for i, f1 in enumerate(f1s):
            self.writer.add_scalar(f"F1/train_batch_{i}", f1, self.count)
        self.writer.add_scalar("F1/train_batch", avg_f1, self.count)


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
        n_classes, average="none", ignore_index=-100
    )
    f1_prog = torchmetrics.classification.MulticlassF1Score(
        n_classes, average="none", ignore_index=-100
    )
    loss_all = torchmetrics.MeanMetric()
    loss_prog = torchmetrics.MeanMetric()
    cm = torchmetrics.classification.MulticlassConfusionMatrix(
        n_classes, ignore_index=-100, normalize="true"
    )
    # Move everything to whatever device the model is on
    device = next(model.parameters()).device
    for thing in [f1_all, f1_prog, loss_all, loss_prog, cm, criterion]:
        thing.to(device)
    pbar = tqdm.tqdm(dataloader, desc="Training", leave=False)
    for i, example_cpu in enumerate(pbar):
        example = gff.util.recursive_todevice(example_cpu, device)
        targ = example.pop("floodmap")
        pred = model(example)

        optim.zero_grad()
        loss = criterion(pred, targ[:, 0])
        loss.backward()
        optim.step()

        f1_all.update(pred, targ[:, 0])
        f1_prog.update(pred, targ[:, 0])
        cm.update(pred, targ[:, 0])
        loss_all.update(loss)
        loss_prog.update(loss)
        if (i % scalar_freq) == 0 and i != 0:
            this_loss = loss_prog.compute().cpu()
            this_f1 = f1_prog.compute().cpu().numpy().tolist()
            avg_f1 = sum(this_f1) / len(this_f1)
            custom_writer.write_scalars(this_loss, avg_f1, this_f1)
            loss_prog.reset()
            f1_prog.reset()
            pbar.set_description(f"Training [{avg_f1:4.2f}]")
        if (i % img_freq) == 0 and i != 0:
            custom_writer.write_imgs(example_cpu, pred.detach().cpu(), targ.detach().cpu())
        custom_writer.step()
    return (loss_all.compute().cpu().item(), f1_all.compute().cpu().numpy().tolist(), cm)


def val_epoch(model, dataloader, criterion, n_classes, limit: int = None):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        f1 = torchmetrics.classification.MulticlassF1Score(
            n_classes, average="none", ignore_index=-100
        )
        loss_all = torchmetrics.MeanMetric()
        cm = torchmetrics.classification.MulticlassConfusionMatrix(
            n_classes, ignore_index=-100, normalize="true"
        )
        for thing in [f1, loss_all, cm, criterion]:
            thing.to(device)
        total_steps = len(dataloader) if limit is None else limit
        pbar = tqdm.tqdm(dataloader, desc="Val-ing", leave=False, total=total_steps)
        for i, example in enumerate(pbar):
            example = gff.util.recursive_todevice(example, device)
            targ = example.pop("floodmap")
            pred = model(example)

            loss = criterion(pred, targ[:, 0])

            f1.update(pred, targ[:, 0])
            cm.update(pred, targ[:, 0])
            loss_all.update(loss)
            if limit is not None and i >= limit:
                break
    return (loss_all.compute().cpu().item(), f1.compute().cpu().numpy().tolist(), cm)


def training_loop(C, model_folder, model: nn.Module, dataloaders, checkpoint=None):
    assert C["n_classes"] in [2, 3], "Num classes must be 2 (bg/water) or 3 (bg/perm.water/flood)"

    optim = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda epoch: C["lr_decay"])
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    cls_weight = torch.tensor(C.get("class_weights", (1.0,) * 3))[: C["n_classes"]]
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
        pbar = tqdm.tqdm(range(start_epoch, C["epochs"]), desc="Epochs")
        for epoch in pbar:
            hoisted_epoch = epoch
            train_loss, train_f1s, train_cm = train_epoch(
                model,
                train_dl,
                criterion,
                C["n_classes"],
                optim,
                custom_writer,
                scalar_freq=C["scalar_freq"],
                img_freq=C["img_freq"],
            )
            val_loss, val_f1s, val_cm = val_epoch(
                model, val_dl, criterion, C["n_classes"], limit=256
            )

            # Write statistics to Tensorboard
            writer.add_scalar("Loss/train_epoch", train_loss, custom_writer.count)
            writer.add_scalar("Loss/val", val_loss, custom_writer.count)
            for i in range(C["n_classes"]):
                writer.add_scalar(f"F1/train_epoch_{i}", train_f1s[i], custom_writer.count)
                writer.add_scalar(f"F1/val_{i}", val_f1s[i], custom_writer.count)
            train_f1 = sum(train_f1s) / len(train_f1s)
            val_f1 = sum(val_f1s) / len(val_f1s)
            writer.add_scalar("F1/train_epoch", train_f1, custom_writer.count)
            writer.add_scalar("F1/val", val_f1, custom_writer.count)
            pbar.set_description(f"Epochs [val F1: {val_f1:4.2f}]")

            # Draw confustion matrix to Tensorboard
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            train_cm.plot(ax=axs[0], labels=gff.constants.KUROSIWO_CLASS_NAMES[: C["n_classes"]])
            val_cm.plot(ax=axs[1], labels=gff.constants.KUROSIWO_CLASS_NAMES[: C["n_classes"]])
            axs[0].set_title("Train")
            axs[1].set_title("Val")
            fig.tight_layout()
            writer.add_figure("confusion_matrix", fig, custom_writer.count)
            plt.close(fig)

            scheduler.step()
            do_save(epoch)
    finally:
        writer.close()
        do_save(hoisted_epoch)

    return model
