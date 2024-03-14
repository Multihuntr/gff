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
