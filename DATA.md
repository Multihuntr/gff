# Data sources

This project builds on many amazing existing datasets and models to achieve its various purposes.

All data is publicly available, but some need an account to download. Here is the full list of data sources and where to download them.

## Data

River/Basin geometries and parameters
* HydroATLAS - RiverATLAS and BasinATLAS
* Source: https://www.hydrosheds.org/hydroatlas (Accessed 12/01/2024)
* Size: 22GB
* Purpose: P1, P2

Global past flood events
* Dartmouth Flood Observatory
* Source: https://floodobservatory.colorado.edu/temp/ (Accessed 12/01/2024)
* Size: 0.005GB

Global DEM
* Source:
    - SRTM 3s DEM https://download.esa.int/step/auxdata/dem/SRTM90/tiff
    - CopDEM 30m https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload
* Size: 70GB
* See: `gff/data_sources.py`

Global weather parameters (ERA5/ERA5-Land)
* Source:
    - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
    - https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY
    - https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
* Size: 200GB
* See `scripts/dl-era5-land.py`

Flood maps ground truth (Hand-labelled)
* Kuro Siwo
* Source: [GitHub][1]
* Size: 400GB download, <10GB after compression
* See: `scripts/dl-ks-labels.sh`

Local Satellite Images
* Sentinel-1 images
* Source: Alaska Satellite Facility (through `asf_search`)
* Size: Multiple TB
* See: `gff/generate/search.py`

Height Above Nearest Drainage (HAND)
* Source:
    - https://storymaps.arcgis.com/stories/fcd5cba6104645349f59e504042bacd6
    - https://glo-30-hand.s3.amazonaws.com/v1/2021/
* Size: 34GB
* See: `gff/data_sources.py`

## Models
Flood maps from Sentinel-1
* Kuro Siwo - FloodViT and SNUNet
* Source: https://github.com/Orion-AI-Lab/KuroSiwo (Accessed 15/01/2024)


[1]: https://github.com/Orion-AI-Lab/KuroSiwo
