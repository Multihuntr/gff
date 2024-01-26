"""
I initially wrote some untested code to do basin-averaging. But then we decided
that we might as well make functioning code.

The main point I'm not sure about is whether my understanding of "basin-averaged" is correct.

!!  The main lines to look at are L63-90 and L112-125  !!

If you look further into the code than just these and have questions:
I have used a demo weather file with only 13 days of ERA5-Land data, and constrained spatial
boundaries. In this demo code I rely on this fact in a few situations, but if I use global
ERA5-Land, this will need to be changed.

The main points that I will change for a real project is to:
1. Calculate the basin attributes once. These functions are written to showcase the logic
    as clearly as I could; but there's certainly optimisations to be made.
2. Feed selected basin and the hours from other parts of code.
3. Use global gridded_weathers instead of my demo ones.

If you see any important optimisations missing, please let me know.
"""

import argparse
import sys
from pathlib import Path

import geopandas
import rasterio
import shapely
import numpy as np
import tensorflow as tf

import util


def get_basin_weather_values(gridded_weathers, basin_geom, band_idxs):
    """
    Calculate basin-averaged weather values for a single basin.

    Args:
        gridded_weathers (list(rasterio.Dataset)): Weather as rasters, one dataset per variable.
        basin_geom (shapely.Geometry): Geometry (Polygon) of basin
        band_idxs (np.ndarray): shaped [len(gridded_weathers), N] where N=len(range([start, end)))

    Returns:
        np.ndarray: Weather data shaped [len(gridded_weathers)]
    """
    # Transform the basin geometry from lat/long to pixel-space of gridded_weathers
    assert all(
        w.transform == gridded_weathers[0].transform for w in gridded_weathers
    ), "All gridded_weathers must have the same geotransform"
    # Note: geotransforms transform from pixel-space to lat/long.
    # We want the opposite, so we invert the transform (rasterio.Affine uses '~' to invert)
    # Note2: This only works so easily because the gridded_weather is in the same CRS as the geoms.
    latlon_to_px = (~(gridded_weathers[0].transform)).to_shapely()
    basin_geometry_pixel = shapely.affinity.affine_transform(basin_geom, latlon_to_px)

    # Determine bounds within gridded weather pixel-space
    xlo, ylo, xhi, yhi = basin_bounds = util.rounded_bounds(basin_geometry_pixel)

    # Create a soft overlap mask (values 0-1; the overlap between pixels and basin_row geometry)
    overlap_mask = util.mk_pixel_overlap_mask(basin_geometry_pixel, basin_bounds)
    basin_area_in_pixels = overlap_mask.sum()
    # overlap_mask (via shapely) defaults to xy, rasterio is in yx.
    overlap_mask = overlap_mask.T

    # Create minimal bounding block of weather data
    window = util.shapely_bounds_to_rasterio_window(basin_bounds)
    minimal_weather_block = np.empty((len(gridded_weathers), yhi - ylo, xhi - xlo))
    for weather_idx, gridded_weather in enumerate(gridded_weathers):
        # Read bands
        bidxs = band_idxs[weather_idx]
        bands = gridded_weather.read(bidxs, window=window)

        # Apply offsets and scales for int16 -> float32 conversion
        # (rasterio does not do this automatically)
        scales = np.array([gridded_weather.scales[i - 1] for i in bidxs])[:, None, None]
        offsets = np.array([gridded_weather.offsets[i - 1] for i in bidxs])[:, None, None]
        bands = (bands * scales) + offsets

        # Aggregate across bands (hours). There are two types of aggregation: average and sum.
        if gridded_weather.agg_type == "avg":
            # 'avg' is used when the thing being measured is a state-based variable:
            #    e.g. there IS AN average temperature over an area.
            # Normalise the mean per pixel so that a weighted sum becomes a weighted average.
            minimal_weather_block[weather_idx] = bands.mean(axis=0) / basin_area_in_pixels
        elif gridded_weather.agg_type == "sum":
            # 'sum' is used when the thing being measured is an action-based variable:
            #    e.g. there HAS BEEN a total amount of precipitation over an area.
            minimal_weather_block[weather_idx] = bands.sum(axis=0)

    # Calculate the weighted average/sum of each gridded_weather
    averaged = (minimal_weather_block * overlap_mask[None]).sum(axis=(1, 2))
    return averaged


def global_basin_averages(gridded_weathers, basin_df, band_idxs):
    """
    Calculates:
    1. the basin-averaged weather variables in `gridded_weathers`, and
    2. the basin-averaged basin features in `basin_df`.

    Args:
        gridded_weathers (list(rasterio.Dataset)): Weather as rasters, one dataset per variable.
        basin_df (geopandas.DataFrame): meta_features + features + geometry
        band_idxs (np.ndarray): [len(gridded_weathers), N] where N=len(range([start, end)))

    Returns:
        numpy.ndarray: Vector of basin-averaged weather values
        numpy.ndarray: Vector of basin-averaged basin features
    """
    avg_weather_values = np.zeros(len(gridded_weathers))
    feature_columns = basin_df.columns[len(META_BASIN_KEYS) : -1]
    avg_basin_values = np.zeros(len(feature_columns))
    total_area = 0

    # Pre-calculate the total area of basin_df
    for _, basin_row in basin_df.iterrows():
        total_area += basin_row.geometry.area

    for _, basin_row in basin_df.iterrows():
        # Collect weather variables
        weather_values = get_basin_weather_values(gridded_weathers, basin_row.geometry, band_idxs)
        avg_weather_values += (basin_row.geometry.area / total_area) * weather_values

        # Collect basin state variables
        basin_values = basin_row[feature_columns].to_numpy().astype(np.float32)
        avg_basin_values += (basin_row.geometry.area / total_area) * basin_values

    return avg_weather_values, avg_basin_values


def get_band_idxs(gridded_weathers, hours):
    """
    Gets a parallel list of band indices for gridded_weathers to extract the specified hours.

    Note: ERA5-Land needs to show times since before unix time orign.
          So it uses hours since 1900-01-01 00:00:00 instead of unix time.

    Args:
        gridded_weathers (list(rasterio.Dataset)): Weather as rasters, one dataset per variable.
        hours (tuple(int, int)): [start, end) range of hours to use

    Returns:
        (np.ndarray): shaped [len(gridded_weathers), N] where N=len(range([start, end)))
    """
    band_idxs = []
    for gridded_weather in gridded_weathers:
        # TODO: Surely there's a better way to read hour of first band from a netcdf file.
        as_txt = gridded_weather.tags()["NETCDF_DIM_time_VALUES"]
        first_comma = as_txt.index(",")
        first_hour = int(as_txt[1:first_comma])
        band_idxs.append(tuple(np.arange(*hours) - first_hour + 1))
    return band_idxs


META_BASIN_KEYS = ["HYBAS_ID", "SORT", "NEXT_SINK", "NEXT_DOWN"]
META_RIVER_KEYS = ["HYBAS_L12", "MAIN_RIV"]
# fmt: off
BASIN_FEATURES = [
    'inu_pc_umn', 'inu_pc_umx', 'inu_pc_ult', 'lka_pc_use', 'lkv_mc_usu', 'rev_mc_usu',
    'ria_ha_usu', 'riv_tc_usu', 'ele_mt_uav', 'slp_dg_uav', 'tmp_dc_uyr', 'pre_mm_uyr',
    'pet_mm_uyr', 'aet_mm_uyr', 'ari_ix_uav', 'cmi_ix_uyr', 'snw_pc_uyr', 'glc_pc_u01',
    'glc_pc_u02', 'glc_pc_u03', 'glc_pc_u04', 'glc_pc_u05', 'glc_pc_u06', 'glc_pc_u07',
    'glc_pc_u08', 'glc_pc_u09', 'glc_pc_u10', 'glc_pc_u11', 'glc_pc_u12', 'glc_pc_u13',
    'glc_pc_u14', 'glc_pc_u15', 'glc_pc_u16', 'glc_pc_u17', 'glc_pc_u18', 'glc_pc_u19',
    'glc_pc_u20', 'glc_pc_u21', 'glc_pc_u22', 'pnv_pc_u01', 'pnv_pc_u02', 'pnv_pc_u03',
    'pnv_pc_u04', 'pnv_pc_u05', 'pnv_pc_u06', 'pnv_pc_u07', 'pnv_pc_u08', 'pnv_pc_u09',
    'pnv_pc_u10', 'pnv_pc_u11', 'pnv_pc_u12', 'pnv_pc_u13', 'pnv_pc_u14', 'pnv_pc_u15',
    'wet_pc_ug1', 'wet_pc_ug2', 'wet_pc_u01', 'wet_pc_u02', 'wet_pc_u03', 'wet_pc_u04',
    'wet_pc_u05', 'wet_pc_u06', 'wet_pc_u07', 'wet_pc_u08', 'wet_pc_u09', 'for_pc_use',
    'crp_pc_use', 'pst_pc_use', 'ire_pc_use', 'gla_pc_use', 'prm_pc_use', 'pac_pc_use',
    'cly_pc_uav', 'slt_pc_uav', 'snd_pc_uav', 'soc_th_uav', 'swc_pc_uyr', 'kar_pc_use',
    'ero_kh_uav', 'pop_ct_usu', 'ppd_pk_uav', 'urb_pc_use', 'nli_ix_uav', 'rdd_mk_uav',
    'hft_ix_u93', 'hft_ix_u09', 'gdp_ud_usu'
]
# fmt: on
# WEATHER_VARIABLES is: shortname->(longname, aggregation_type)
WEATHER_VARIABLES = {
    "tp": ("total precipitation", "sum"),
    "t2m": ("2-meter temperature", "avg"),
    "ssr": ("surface net solar radiation", "sum"),
    "str": ("surface net thermal radiation", "sum"),
    "sf": ("snowfall", "sum"),
    "sp": ("surface pressure", "avg"),
}


def parse_args(argv):
    parser = argparse.ArgumentParser("Demo upstream basin-averaging")
    parser.add_argument("weather_path", type=Path, help="File path of weather data (.nc)")
    parser.add_argument("hydroatlas_path", type=Path, help="Folder with Basin and River ATLAS")
    parser.add_argument("model_path", type=Path, help="Directory of model to use")
    parser.add_argument("basin_id", type=int, help="ID of basin to calculate upstream")
    parser.add_argument(
        "hours",
        type=int,
        nargs=2,
        help="[start, end) of hours since 1900-01-01 to include in calculation",
    )
    return parser.parse_args(argv)


def main(args):
    # Get gridded weather (ERA5-Land)
    print("Opening weather file...")
    gridded_weathers = []
    for shortname, (_, aggregation_type) in WEATHER_VARIABLES.items():
        dataset = rasterio.open(f"netcdf:{str(args.weather_path)}:{shortname}", crs="EPSG:4326")
        # TODO: Better way to make this association?
        dataset.agg_type = aggregation_type
        gridded_weathers.append(dataset)
    xlo, ylo, xhi, yhi = gridded_weathers[0].bounds
    weather_bbox = xlo, ylo, xhi, yhi

    # Get HydroATLAS - have to load a subset, otherwise it takes ages.
    basin_dir = args.hydroatlas_path / "BasinATLAS" / "BasinATLAS_v10_shp"
    print("Loading basins...")
    basins = geopandas.read_file(
        basin_dir / f"BasinATLAS_v10_lev12.shp",
        bbox=weather_bbox,
        include_fields=META_BASIN_KEYS + BASIN_FEATURES,
        engine="pyogrio",
    )

    # Get all upstream basins
    print("Calculating all upstream basins...")
    upstream_basins = util.get_upstream_basins(basins, args.basin_id)

    # Calculate (weighted) average of variables over the basins
    print("Calculating global basin_averages...")
    weather_avgs = []
    for start_hour in range(*args.hours, 24):
        print(f" Processing day starting {start_hour}...")
        band_idxs = get_band_idxs(gridded_weathers, (start_hour, start_hour + 24))
        weather_avg, feature_avg = global_basin_averages(
            gridded_weathers, upstream_basins, band_idxs
        )
        weather_avgs.append(weather_avg)
    weather_avgs = np.array(weather_avgs)

    # ========================================================================
    # ==== Beyond this line is hacky code to try and get the model to run ====
    # ==== Not necessarily run well, mind you, just run wihout error.     ====
    # ==== i.e. just match data shapes and vaguely correct distributions. ====

    # Use the hoisted feature_avg from the loop, since it is the same across iterations.
    # Then copy attributes to make up the missing attributes the model asks for. Why 98 attributes?
    global_attributes_statics = np.concatenate([feature_avg, feature_avg[:11]])[None, :].astype(
        np.float32
    )

    # I didn't realise it would need 365 days, so I only downloaded 13 days of interest of ERA5-Land
    weather_avgs = np.concatenate([weather_avgs] * 40)[-(365 + 7) :][None, :].astype(np.float32)
    weather_hindcast = weather_avgs[:, :365]
    weather_forecast = weather_avgs[:, 365:]

    # Feed these values into a model
    model = tf.keras.models.load_model(args.model_path)

    # fmt: off
    inp = {
        'GLOBAL_ATTRIBUTES/STATICS': global_attributes_statics,
        'INCLUDE_MISSING_BASINS/CPC/PRECIP/hindcast': weather_hindcast[..., 0:1],
        'INCLUDE_MISSING_BASINS/IMERG/PRECIP_EARLY/hindcast': weather_hindcast[..., 0:1],
        'INPUT_MASKING/INCLUDE_MISSING_BASINS/CPC/PRECIP/hindcast': weather_hindcast[..., 0:1],
        'INPUT_MASKING/INCLUDE_MISSING_BASINS/IMERG/PRECIP_EARLY/hindcast': weather_hindcast[..., 0:1],
        'STATIC/BASIN_ID': tf.convert_to_tensor([[str(args.basin_id)]], dtype=tf.string),
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/TP//INCLUDE_MISSING_BASINS/ERA5/TP_FORECAST/forecast': weather_forecast[..., 0:1],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/TP_NOWCAST//INCLUDE_MISSING_BASINS/ERA5/TP/hindcast': weather_hindcast[..., 0:1],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/T2M//INCLUDE_MISSING_BASINS/ERA5/2T_FORECAST/forecast': weather_forecast[..., 1:2],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/T2M_NOWCAST//INCLUDE_MISSING_BASINS/ERA5/2T/hindcast': weather_hindcast[..., 1:2],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/SSR//INCLUDE_MISSING_BASINS/ERA5/SSR_FORECAST/forecast': weather_forecast[..., 2:3],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/SSR_NOWCAST//INCLUDE_MISSING_BASINS/ERA5/SSR/hindcast': weather_hindcast[..., 2:3],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/STR//INCLUDE_MISSING_BASINS/ERA5/STR_FORECAST/forecast': weather_forecast[..., 3:4],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/STR_NOWCAST//INCLUDE_MISSING_BASINS/ERA5/STR/hindcast': weather_hindcast[..., 3:4],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/SF//INCLUDE_MISSING_BASINS/ERA5/SF_FORECAST/forecast': weather_forecast[..., 4:5],
        'UNION//INCLUDE_MISSING_BASINS/ECMWF/SF_NOWCAST//INCLUDE_MISSING_BASINS/ERA5/SF/hindcast': weather_hindcast[..., 4:5],
        # No surface pressure?
    }
    # fmt: on
    out = model.predict(inp)

    print(out)  # What's the streamflow here?


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
