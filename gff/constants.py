import numpy as np

KUROSIWO_CLASS_NAMES = ["background", "permanent water", "flood"]
KUROSIWO_BG_CLASS = KUROSIWO_CLASS_NAMES.index("background")
KUROSIWO_PW_CLASS = KUROSIWO_CLASS_NAMES.index("permanent water")
KUROSIWO_FLOOD_CLASS = KUROSIWO_CLASS_NAMES.index("flood")
KUROSIWO_S1_NAMES = ["pre2", "pre1", "post"]
FLOODMAP_BLOCK_SIZE = 224
# For some reason PACKBITS wasn't compressing very well.
# Disk size scaled by regions of nodata.
# Back to DEFLATE :shrug:
FLOODMAP_PROFILE_DEFAULTS = {
    "COMPRESS": "DEFLATE",
    "ZLEVEL": 1,
    "PREDICTOR": 2,
    "count": 1,
    "dtype": np.uint8,
}
WORLDCOVER_PW_CLASS = 80

HYDROATLAS_CONTINENT_NAMES = {
    1: "Africa",
    2: "Europe and middle east",
    3: "Russia",
    4: "Asia",
    5: "Oceania, pacific islands and south-east Asia",
    6: "South America",
    7: "North America 1",
    8: "North America 2",
    9: "Greenland",
}
HYDROATLAS_CLIMATE_ZONE_NAMES = {
    1: "Arctic 1",
    2: "Arctic 2",
    3: "Extremely cold and wet 1",
    4: "Extremely cold and wet 2",
    5: "Cold and wet",
    6: "Extremely cold and mesic",
    7: "Cold and mesic",
    8: "Cool temperate and dry",
    9: "Cool temperate and xeric",
    10: "Cool temperate and moist",
    11: "Warm temperate and mesic",
    12: "Warm temperate and xeric",
    13: "Hot and mesic",
    14: "Hot and dry",
    15: "Hot and arid",
    16: "Extremely hot and arid",
    17: "Extremely hot and xeric",
    18: "Extremely hot and moist",
}
DFO_INCLUDE_TYPES = [
    "Cyclone/storm",
    "Heavy Rain",
    "Heavy Rain AND Cyclone/storm",
    "Heavy Rain AND Tides/Surge",
    "Tides/Surge",
]
