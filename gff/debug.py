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


def save_as_geojson(shps, fname="debug.geojson"):
    import json
    import shapely

    if isinstance(shps, shapely.Geometry):
        shps = np.array([shps])

    geojson = {
        "type": "FeatureCollection",
        "name": "test",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "geometry": eval(shapely.to_geojson(s))} for s in shps.flatten()
        ],
    }
    with open(fname, "w") as f:
        json.dump(geojson, f)
