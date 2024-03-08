import numpy as np

KUROSIWO_CLASS_NAMES = ["background", "permanent water", "flood"]
KUROSIWO_FLOOD_CLASS = KUROSIWO_CLASS_NAMES.index("flood")
FLOODMAP_BLOCK_SIZE = 224
FLOODMAP_PROFILE_DEFAULTS = {
    "COMPRESS": "DEFLATE",
    "ZLEVEL": 1,
    "PREDICTOR": 2,
    "count": 3,
    "descriptions": KUROSIWO_CLASS_NAMES,
    "dtype": np.uint8,
}
