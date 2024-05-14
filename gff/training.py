import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchmetrics
import tqdm

import gff.util


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optim: torch.optim.Optimizer,
    write_scalars: callable,
    write_imgs: callable,
    scalar_freq: int,
    img_freq: int,
):
    losses = []
    device = next(model.parameters()).device
    f1 = torchmetrics.classification.MulticlassF1Score(3, average="macro", ignore_index=-100)
    f1.to(device)
    f1_prog = torchmetrics.classification.MulticlassF1Score(3, average="macro", ignore_index=-100)
    f1_prog.to(device)
    criterion = criterion.to(device)
    model.train()
    for i, example_cpu in enumerate(tqdm.tqdm(dataloader, desc="Training", leave=False)):
        example = gff.util.recursive_todevice(example_cpu, device)
        targ = example.pop("floodmap")
        pred = model(example)
        targ[targ > 2] = -100  # To make criterion ignore weird numbers from Kuro Siwo

        optim.zero_grad()
        loss = criterion(pred, targ[:, 0])
        loss.backward()
        optim.step()

        f1(pred, targ[:, 0])
        f1_prog(pred, targ[:, 0])
        loss_float = loss.cpu().item()
        losses.append(loss_float)
        if ((i + 1) % scalar_freq) == 0:
            this_f1 = f1_prog.compute().cpu().item()
            write_scalars(loss_float, this_f1)
            f1_prog = torchmetrics.classification.MulticlassF1Score(
                3, average="macro", ignore_index=-100
            )
            f1_prog.to(device)
        if ((i + 1) % img_freq) == 0:
            write_imgs(example_cpu, pred.detach().cpu(), targ.detach().cpu())
    return sum(losses) / len(losses), f1.compute().cpu().item()


def test_epoch(model, dataloader, criterion, limit: int = None):
    model.eval()
    with torch.no_grad():
        losses = []
        device = next(model.parameters()).device
        f1 = torchmetrics.classification.MulticlassF1Score(3, average="macro", ignore_index=-100)
        f1.to(device)
        for i, example in enumerate(tqdm.tqdm(dataloader, desc="Testing", leave=False)):
            example = gff.util.recursive_todevice(example, device)
            targ = example.pop("floodmap")
            pred = model(example)
            targ[targ > 2] = -100

            loss = criterion(pred, targ[:, 0])

            f1(pred, targ[:, 0])
            losses.append(loss.cpu().item())
            if limit is not None and i >= limit:
                break
    return sum(losses) / len(losses), f1.compute()


def auto_incr_scalar_writer_closure(writer):

    def add_next(loss: float, f1: float):
        writer.add_scalar("Loss/train_batch", loss, add_next.count)
        writer.add_scalar("F1/train_batch", f1, add_next.count)
        add_next.count += 1

    add_next.count = 0
    return add_next


def cls_to_rgb(arr: np.ndarray):
    colours = np.array([(18, 19, 19), (60, 125, 255), (255, 64, 93)]) / 256
    out_arr = np.zeros((*arr.shape, 3))
    for cls_idx in range(3):
        out_arr[(arr == cls_idx)] = colours[cls_idx][None]
    return out_arr


def auto_incr_img_writer_closure(writer):

    def add_next(ex: dict, pred: torch.tensor, targ: torch.tensor):
        pred_cls = pred[0].argmax(dim=0).numpy()
        pred_rgb = cls_to_rgb(pred_cls)
        targ_cls = targ[0, 0].numpy()
        targ_rgb = cls_to_rgb(targ_cls)
        context_rasters = []
        for k in ["era5", "era5_land"]:
            if k in ex:
                r = ex[k][0][:, :3].numpy()
                context_rasters.extend(r)
        if "hydroatlas" in ex:
            r = ex["hydroatlas"][0][:3].numpy()
            context_rasters.append(r)
        if "dem_context" in ex:
            r = ex["dem_context"][0].repeat((3, 1, 1)).numpy()
            context_rasters.append(r)

        local_rasters = []
        if "dem_local" in ex:
            r = ex["dem_local"][0].repeat((3, 1, 1)).numpy()
            local_rasters.append(r)
        if "s1" in ex:
            r = ex["s1"][0].numpy()
            r = np.stack([*r, np.zeros_like(r[0])], axis=0)
            local_rasters.append(r)

        writer.add_images("context", np.stack(context_rasters), add_next.count)
        writer.add_images("local", np.stack(local_rasters), add_next.count)
        writer.add_images(
            "pred_vs_targ", np.stack([pred_rgb, targ_rgb]), add_next.count, dataformats="NHWC"
        )
        add_next.count += 1

    add_next.count = 0
    return add_next


def training_loop(C, model_folder, model: nn.Module, dataloaders, checkpoint=None):
    optim = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda epoch: C["lr_decay"])
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    writer = torch.utils.tensorboard.SummaryWriter(model_folder)
    write_scalars = auto_incr_scalar_writer_closure(writer)
    write_imgs = auto_incr_img_writer_closure(writer)

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
                optim,
                write_scalars,
                write_imgs,
                scalar_freq=C["scalar_freq"],
                img_freq=C["img_freq"],
            )
            test_loss, test_f1 = test_epoch(model, val_dl, criterion, limit=256)

            writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            writer.add_scalar("F1/train_epoch", train_f1, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("F1/test", test_f1, epoch)

            scheduler.step()
    finally:
        writer.close()
        do_save(hoisted_epoch)

    return model
