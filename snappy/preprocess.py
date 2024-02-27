import argparse
import itertools
from pathlib import Path
import sys

# From a S1 preprocessing tutorial
# https://step.esa.int/docs/tutorials/Performing%20SAR%20processing%20in%20Python%20using%20snappy.pdf
# And then modified by me

sys.path.append("/opt/conda/envs/snappy/snap/.snap/snap-python")
import snappy

OPTS = {
    "orbit": {"polyDegree": 2},
    "speckle": {
        "filter": "Lee Sigma",
        "filterSizeX": "5",
        "filterSizeY": "5",
        "sigmaStr": "0.9",
    },
}
# The closest I can find is Frost 7x7 with damping "2"
OPT_RANGES = {
    "orbit": {"polyDegree": [2]},
    "speckle": {
        "filter": ["Lee Sigma"],  # ["Boxcar", "Median", "Lee", "Refined Lee", "Frost"],
        "filterSizeX": ["7"],  # [f"{n}" for n in range(3, 16) if n % 2 == 1],
        "filterSizeY": "filterSizeX",
        "dampingFactor": ["1"],  # [f"{n}" for n in range(1, 5)], # Only for Frost
        "estimateENL": ["true"],
        "enl": ["1.0"],  # No change for Frost
        "numLooksStr": ["1"],  # No change for Frost
        "targetWindowSizeStr": ["3x3"],  # ["3x3", "5x5"],  # No change for Frost
        "sigmaStr": ["0.9"],  # Only for Lee Sigma
        "anSize": ["50"],  # No change for Frost
    },
}


def _str_list_to_java(str_list):
    return snappy.jpy.array("java.lang.String", str_list)


def _dict_to_java(d):
    out_dict = snappy.HashMap()
    for k, v in d.items():
        out_dict.put(k, v)
    return out_dict


def apply_orbit(product, opts=OPTS["orbit"]):
    parameters = snappy.HashMap()
    parameters.put("orbitType", "Sentinel Precise (Auto Download)")
    parameters.put("polyDegree", opts["polyDegree"])
    parameters.put("continueOnFail", "false")
    return snappy.GPF.createProduct("Apply-Orbit-File", parameters, product)


def remove_thermal_noise(product, bands=["VV", "VH"]):
    parameters = snappy.HashMap()
    parameters.put("selectedPolarisations", _str_list_to_java(bands))
    parameters.put("removeThermalNoise", "true")
    return snappy.GPF.createProduct("ThermalNoiseRemoval", parameters, product)


def remove_border_noise(product, bands=["VV", "VH"]):
    parameters = snappy.HashMap()
    parameters.put("selectedPolarisations", _str_list_to_java(bands))
    # parameters.put("borderLimit", ??)
    # parameters.put("trimThreshold", ??)
    return snappy.GPF.createProduct("Remove-GRD-Border-Noise", parameters, product)


def subset(product, wkt):
    parameters = snappy.HashMap()
    SubsetOp = snappy.jpy.get_type("org.esa.snap.core.gpf.common.SubsetOp")
    geometry = snappy.WKTReader().read(wkt)
    parameters.put("copyMetadata", True)
    parameters.put("geoRegion", geometry)
    return snappy.GPF.createProduct("Subset", parameters, product)


def calibration(product, bands=["VV", "VH"]):
    parameters = snappy.HashMap()
    parameters.put("outputSigmaBand", True)
    parameters.put("sourceBands", _str_list_to_java([f"Intensity_{band}" for band in bands]))
    parameters.put("selectedPolarisations", _str_list_to_java(bands))
    parameters.put("outputImageScaleInDb", False)
    return snappy.GPF.createProduct("Calibration", parameters, product)


def speckleFilter(product, bands=["VV", "VH"], opts=OPTS["speckle"]):
    parameters = _dict_to_java(opts)

    parameters.put("sourceBands", _str_list_to_java([f"Sigma0_{band}" for band in bands]))

    return snappy.GPF.createProduct("Speckle-Filter", parameters, product)


def terrainCorrection(product, bands=["VV", "VH"]):
    parameters = snappy.HashMap()
    parameters.put("demName", "SRTM 1Sec HGT")
    parameters.put("pixelSpacingInMeter", 10.0)
    parameters.put("sourceBands", _str_list_to_java([f"Sigma0_{band}" for band in bands]))
    parameters.put("mapProjection", "EPSG:3857")
    return snappy.GPF.createProduct("Terrain-Correction", parameters, product)


def add_dem_band(product, dem_name="SRTM 1Sec HGT"):
    parameters = snappy.HashMap()
    parameters.put("demName", dem_name)
    return snappy.GPF.createProduct("AddElevation", parameters, product)


def list_op_params(operator_name):
    snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    op_spi = snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi(operator_name)
    print("Op name:", op_spi.getOperatorDescriptor().getName())
    print("Op alias:", op_spi.getOperatorDescriptor().getAlias())
    param_Desc = op_spi.getOperatorDescriptor().getParameterDescriptors()
    for param in param_Desc:
        print(param.getName(), "or", param.getAlias())


def parse_args(argv):
    parser = argparse.ArgumentParser("Default preprocessing for a Sentinel-1 image.")

    parser.add_argument("zip_path", type=str, help="Path to Sentinel-1 zip download")
    parser.add_argument("aoi_wkt", type=str, help="Area of Interest (AOI) in WKT format")
    parser.add_argument("out_path", type=str, help="Where to save output")
    parser.add_argument("--sweep", action="store_true", help="Sweep various configs")
    parser.add_argument("--explain", type=str, default=None, help="Describe params for op")
    parser.add_argument("--incl_dem", action="store_true", help="Also ouput the equivalent DEM")
    parser.add_argument(
        "--dem_name", type=str, default="SRTM 1Sec HGT", help="Name of included dem"
    )

    return parser.parse_args(argv)


def do_preprocess(args, opts, expand_filename=False):
    file_path = args.out_path
    if expand_filename:
        out_path = Path(args.out_path)
        descs = [f"{k_in[0]}-{v}" for k_out in opts for k_in, v in opts[k_out].items()]
        fname = f'{out_path.stem}_{"_".join(descs)}.tif'
        file_path = out_path.parent / fname
        if file_path.exists():
            print("Already created")
            return
        else:
            print("Creating", file_path.name)
        file_path = str(file_path)

    snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    # Read only the area of interest from the image
    product_in = snappy.ProductIO.readProduct(args.zip_path)
    # snappy.showProductInformation(product_in)
    product_orbitfile = apply_orbit(product_in, opts=opts["orbit"])
    product_subset = subset(product_orbitfile, args.aoi_wkt)

    # Apply processing steps
    product_out = product_subset
    product_out = remove_border_noise(product_out)
    product_out = remove_thermal_noise(product_out)
    product_out = calibration(product_out)
    product_out = speckleFilter(product_out, opts=opts["speckle"])
    product_out = terrainCorrection(product_out)

    if args.incl_dem:
        product_out = add_dem_band(product_out, args.dem_name)

    snappy.ProductIO.writeProduct(product_out, file_path, "GeoTIFF")


def split_copied_iterable(opts):
    copied = {}
    iterables = {}
    for k, v in opts.items():
        if isinstance(v, str):
            copied[k] = v
        elif isinstance(v, list):
            iterables[k] = v
    return copied, iterables


def main(args):
    if args.explain is not None:
        list_op_params(args.explain)
    elif args.sweep:
        # Opts is nested like: { 'operation': { 'parameter': value } }
        # At least one parameter is copied. This is denoted by a range of just a string.
        # Start by separating them based on type
        copied = {}
        iterables = []
        for k in OPT_RANGES:
            inner_copied, inner_iterables = split_copied_iterable(OPT_RANGES[k])
            copied[k] = inner_copied
            # Use dot notation for nested properties. There is guaranteed to be precisely one level
            iterables.extend(
                [[(f"{k}.{j}", v) for v in v_out] for j, v_out in inner_iterables.items()]
            )
        # Then create the cross-product of all sweep-able parameters
        for kwargs_list in itertools.product(*iterables):
            kwargs = {k: {} for k in OPT_RANGES}
            for k, v in kwargs_list:
                k_out, k_in = k.split(".")
                kwargs[k_out][k_in] = v
            # Copy parameters where specified
            for k_out in copied:
                for k_in, copy_from_k in copied[k_out].items():
                    kwargs[k_out][k_in] = kwargs[k_out][copy_from_k]
            print(kwargs)
            do_preprocess(args, kwargs, expand_filename=True)
    else:
        do_preprocess(args, OPTS)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
