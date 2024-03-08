from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np


def save_as_greyscale(arr, fname="debug.png"):
    assert len(arr.shape) == 2, arr.shape
    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    Image.fromarray(arr).convert("L").save(fname)


def save_as_rgb(arr, fname="debug.png"):
    assert len(arr.shape) == 3, arr.shape
    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    arr = arr.transpose((1, 2, 0))
    Image.fromarray(arr).convert("RGB").save(fname)


def plot_item(item, folder=Path("fig")):
    folder.mkdir(parents=True, exist_ok=True)

    # Series
    t = len(item["input"]["series"])
    fig, ax = plt.subplots(1, 1)
    seriess = item["input"]["sparse"][
        np.repeat(item["input"]["valid_mask"][0:1] == 1, t, axis=0)
    ].reshape((t, -1))
    ax.plot(range(t), seriess, c="gray")
    ax.plot((t + item["input"]["td_lead"] - 1,), item["target"]["series"], c="red", marker="x")
    print(np.abs(seriess.sum(axis=0) - item["input"]["series"].sum(axis=0)) < 1e-5)
    ax.plot(range(t), item["input"]["series"], c="cyan")
    fig.savefig(folder / "series.png")
    plt.close(fig)

    # Imgs
    for k in ["gtsm", "era5", "valid_mask"]:
        for i in range(item["input"][k].shape[0]):
            save_as_greyscale(item["input"][k][i, 0], folder / f"{k}_{i:02d}.png")
    save_as_greyscale(item["input"]["ls_mask"][0], folder / "ls_mask.png")
