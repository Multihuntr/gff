import numpy as np

KUROSIWO_CLASS_NAMES = ["background", "permanent water", "flood"]
KUROSIWO_BG_CLASS = KUROSIWO_CLASS_NAMES.index("background")
KUROSIWO_PW_CLASS = KUROSIWO_CLASS_NAMES.index("permanent water")
KUROSIWO_FLOOD_CLASS = KUROSIWO_CLASS_NAMES.index("flood")
KUROSIWO_S1_NAMES = ["pre2", "pre1", "post"]
FLOODMAP_BLOCK_SIZE = 224
FLOODMAP_PROFILE_DEFAULTS = {
    "COMPRESS": "DEFLATE",
    "ZLEVEL": 1,
    "PREDICTOR": 2,
    "count": 1,
    "dtype": np.uint8,
}
WORLDCOVER_PW_CLASS = 80
