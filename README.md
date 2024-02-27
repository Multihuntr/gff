# Global compound floods

A project to identify and describe compound flood events globally. In particular, Fluvial (riverine) + Storm surge (coastal) flood contributions in areas with no in-situ measurements.

Steps:
1. Identify plausible locations of compound flood events by:
    1. Existence of river
    2. Existence of ocean
    3. Existence of population
2. Analyse contributions
    1. River flood level ([Google floodhub][1] and [Nevo 2022][2])
    2. Storm surge level (TODO:)
3. Estimate compound flood dectection accuracy.
    1. Find coinciding extreme water levels at coastal and river gauges using true data ([Lai 2021][3])
    2. F1 between these and detected events.
4. Create difference flood maps
    1. Post-event flood maps
        1. Sentinel-1 - [Kuro Siwo][4]
        2. Sentinel-2 - NDWI > threshold
    2. Forecast flood maps
        1. Google Inundation Map from river ([Nevo 2022][2])
        2. Storm surge level > DEM level

# External sources

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
mv GRDC-Caravan-extension-nc/timeseries/netcdf/grdc Caravan/timeseries/netcdf/
mv GRDC-Caravan-extension-nc/attributes/grdc Caravan/attributes/
mv GRDC-Caravan-extension-nc/shapefiles/grdc Caravan/shapefiles/
```

Coastal gauge data
* GESLA-3
* Source: https://gesla787883612.wordpress.com/
* Size: 38GB
* Purpose: P5

Global storm surge predictions
* Source:
    - GTSM (https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-water-level-change-timeseries-cmip6?tab=overview)
    - See `scripts/dl-gtsm.py`
* Size: 840MB
* Purpose: P4

Global 1s DEM
* Source:
    - Rehosted by Australia for some reason? https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759
    - https://spacedata.copernicus.eu/documents/20123/121286/Copernicus+DEM+Open+HTTPS+Access.pdf/36c9adad-8488-f463-af43-573e68b7f481?t=1669283200177
    - https://prism-dem-open.copernicus.eu/pd-desk-open-access/publicDemURLs
* Size:
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
* Source:
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


# ERA5-Land from Google Earth Engine

I need daily-aggregated ERA5-Land for the NeuralHydrology models which is not provided natively from CDS. Theoretically, we could download the ERA5-Land archive and reprocess to daily values. But the whole archive is terrabytes large, the api has strong rate limits and I don't need hourly (at least, not everywhere all the time). Instead - presumably because of prior research - a daily reprocessed version is on Google Earth Engine.

I have some preferences:
1. I'd rather not be locked into using earth engine
2. I want to do arbitrary training on my machine
3. I have limited disk space.
4. I want to use a Google-trained model using the daily aggregates, anyway.

Thus, I decided to use `earthengine-api` to export the ERA5-Land files to my Drive and then download that file using `google-api-python-client` (which comes with `earthengine-api`). See `dl-era5-land.py` for more details of this process.


# Sentinel-1 preprocessing

Worked from this dockerfile to figure out how to install locally: https://github.com/snap-contrib/docker-snap
Used this tutorial to figure out how to use snappy: https://step.esa.int/docs/tutorials/Performing%20SAR%20processing%20in%20Python%20using%20snappy.pdf
Used `asf_search` to download S1 images.
TODO: Follow riverways?
TODO: Generate tiles that cover the same locations and run KuroSiwo model on both their data and the reprocessed version, ensuring roughly equivalent predictions.

[1]: https://g.co/floodhub
[2]: https://hess.copernicus.org/articles/26/4013/2022/
[3]: https://journals.ametsoc.org/view/journals/clim/34/20/JCLI-D-21-0050.1.xml
[4]: https://github.com/Orion-AI-Lab/KuroSiwo
[5]: https://emergency.copernicus.eu/mapping/list-of-activations-rapid
