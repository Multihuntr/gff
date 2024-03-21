# Global compound floods

A globally distributed dataset for learning flood forecasting. *

<sub>* = Initially, it a project to identify and describe compound flood events globally. In particular, Fluvial (riverine) + Storm surge (coastal) flood contributions in areas with no in-situ measurements.</sub>

# Environment setup

We use both conda and docker because SNAP needs a different version of python.

```python
conda env create -p envs/flood --file environment.yml
pushd preprocessing ; docker build -t esa-snappy . ; popd
```

# Data

Download:

1. HydroATLAS (HydroBASIN and HydroRIVER)
2. Caravan + GRDC extension

# Authentication

To download S1 images programmatically, we use `asf_search`, which requires login details.

To set this up, there are some manual steps.
1. Ensure you have an [EarthData account](https://urs.earthdata.nasa.gov/).
2. Agree to the ASF EULA. To do this:
    1. Click the above link.
    2. Sign in if you have not already.
    3. Go to "EULAs"->"Accept New EULAs"
    4. Accept the "Alaska Satellite Facility Data Access" EULA
3. Create a file `.asf_auth` in this directory. It should be a JSON with your plaintext credentials. Like this:

```json
{"user": "<username>", "pass": "<password>"}
```

# Data sources

This project builds on many amazing existing datasets and models to achieve its various purposes.

Purposes
* P1. Find list of plausible compound flood events (river + storm surge)
* P2. Run google river gauge estimation model
* P3. Post-event flood maps
* P4. Forecast flood maps
* P5. Measure accuracy of water level predictions

All data is publicly available, but some need an account to download. Here is the full list of data sources and where to download them.

## Data

Population Map (UNUSED)
* WorldPop - Top-down unconstrained Global mosaic 1km
* Source: https://hub.worldpop.org/geodata/summary?id=24777 (Accessed 17/01/2024)
* More detail: https://www.worldpop.org/methods/top_down_constrained_vs_unconstrained/
* Size: 1GB
* Purpose: P1

River/Basin geometries and parameters
* HydroATLAS - RiverATLAS and BasinATLAS
* Source: https://www.hydrosheds.org/hydroatlas (Accessed 12/01/2024)
* Size: 22GB
* Purpose: P1, P2

Global past flood events
* Dartmouth Flood Observatory
* Source: https://floodobservatory.colorado.edu/temp/ (Accessed 12/01/2024)
* Size: 0.005GB
* Purpose: P1

River gauge/discharge data
* Caravan v1.3
* Source: https://zenodo.org/records/7944025 (Accessed ??/01/2024)
* Size: 15GB
* Purpose: P2, P3, P5
```bash
# IMPORTANT: Also download grdc
# (This snippet assumes you are one folder above the root of the Caravan dataset,
#  and downloads/adds the GRDC extension into it)
wget -O grdc.tar.gz "https://zenodo.org/records/10074416/files/caravan-grdc-extension-nc.tar.gz?download=1"
tar -xf grdc.tar.gz
mv GRDC-Caravan-extension-nc/timeseries/netcdf/grdc Caravan/timeseries/netcdf/GRDC
mv GRDC-Caravan-extension-nc/attributes/grdc Caravan/attributes/GRDC
mv GRDC-Caravan-extension-nc/shapefiles/grdc Caravan/shapefiles/GRDC
```

Coastal gauge data (UNUSED)
* GESLA-3
* Source: https://gesla787883612.wordpress.com/
* Size: 38GB
* Purpose: P5

Global storm surge predictions (UNUSED)
* Source:
    - GTSM (https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries-cmip6?tab=overview)
    - See `scripts/dl-gtsm.py`
* Size: 840MB
* Purpose: P4

Global DEM
* Source:
    - SRTM 3s DEM https://download.esa.int/step/auxdata/dem/SRTM90/tiff
    - CopDEM 30m https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload
* Size: ~ as needed
* Purpose: P4

Global weather parameters (ERA5-Land)
* Source:
    - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
    - See `scripts/dl-era5-land.py`
* Size:
* Purpose: P2, P3

Hand-labelled flood maps (WONTDO: Kuro Siwo casts doubt on applicability, and isn't global)
* Copernicus Emergency Mapping Service
* Source: [Archives][5] and API
* Size:
* Purpose: P3

Local Satellite Images (TODO: After specific test sites determine)
* Sentinel-1/2 images
* Source: Alaska Satellite Facility (through `asf_search`)
* Size:
* Purpose: P3


## Models
Flood maps from Sentinel-1
* Kuro Siwo - FloodViT
* Source: https://github.com/Orion-AI-Lab/KuroSiwo (Accessed 15/01/2024)
* Purpose: P3

River gauge estimation
* Google LSTM (https://g.co/floodhub and https://hess.copernicus.org/articles/26/4013/2022/)
* Source: https://zenodo.org/records/10397664
* Purpose: P2


# Sentinel-1 preprocessing

Sentinel-1 preprocessing uses SNAP in a dockerfile. Perhaps it should use a locally installed SNAP instead.

[1]: https://g.co/floodhub
[2]: https://hess.copernicus.org/articles/26/4013/2022/
[3]: https://journals.ametsoc.org/view/journals/clim/34/20/JCLI-D-21-0050.1.xml
[4]: https://github.com/Orion-AI-Lab/KuroSiwo
[5]: https://emergency.copernicus.eu/mapping/list-of-activations-rapid
