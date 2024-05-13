# Generating your own floodmaps

So, you want to peer behind the veil, understand how the dataset was generated and modify it for your own purposes. I regret to inform you that it is not fully automated. You will have to do a number of manual steps. Consider yourself warned.

This is how we generated the floodmaps:

1. Download HydroATLAS and DFO
2. Download tropical storm archive: 'titleyetal2021_280storms.csv'
3. Create a raw data folder (e.g. `gff-raw-data`); for the default settings, you will need around 3TB of free space; mostly for S1 files.
4. Set up Sentinel-1 searching and preprocessing:
    1. Sign up for an ASF account and put auth file in cwd. [see instructions](#asf-instructions)
    2. Get SNAP ready for preprocessing:
        1. Install docker
        2. Build `preprocessing/dockerfile` with tag 'esa-snappy'.
        3. Run commands found in `preprocessing/README.md` to get Orbit files (approx 8GB)
5. Download/Merge Kuro Siwo data: Run `scripts/dl-ks-labels.sh`
    1. *Note*: Downloads whole 400GB of files, one part at a time. This might take some time.
    2. *Note*: Written to preserve disk space: only needs 150GB of storage free to run.
    3. *Note*: This will create two folders: `merged-labels` and `merged-s1`.
    1. *Note*: It will attempt to find an S1 that is closer to the flood date than was used in Kuro Siwo, and put it in the raw data folder (e.g. `gff-raw-data/s1`).
    1. *Note*: For ROIs that don't have such an S1 image available, `merged-s1` contains S1 data, taken from Kuro Siwo's S1 data.
6. Generate floodmaps: Run `scripts/gen-whole-dataset.py`, passing HydroATLAS, DFO, tropical storm archive, merged kurosiwo labels folder and the raw data folder.
    1. *Note*: This may take **weeks** of CPU/GPU time to run to completion.
    2. *Note*: If you want to split up the work over multiple workers, it is **only** safe to use the same raw data folder over an NFS if you pass a different value to `--continent` for each instance.
    3. *Note*: The code is partially safe to stop and restart and not lose progress. If you ever need to stop the code and then restart it, you should delete the latest set of floodmaps and S1 images (by modification date). In general, though, it will automatically pick up where it left off somewhat quickly.
    4. *Note*: You can export only one set of floodmaps if you want, but the big time cost is downloading and preprocessing Sentinel-1 images, so it won't be much faster.
    5. *Note*: Running SNUNet spams the console because of a C library within richdem. I'm sorry.
    6. *Note*: This will automatically download COPDEM for SNUNet.
7. Collect ERA5/ERA5-Land daily from GEE:
    1. See comments at the bottom of: `scripts/dl-era5-land.py`
    2. Run with and without `--not_land` to get both ERA5 and ERA5-LAND
        1. *Note*: This may take several days.
        2. *Note*: This needs something like 50GB of RAM.
8. Rasterize HydroATLAS: Run `scripts/rasterize-hydroatlas.py`
    1. *Note*: This requires 300GB of RAM. Here's a link to download a worldwide rasterised HydroATLAS: TODO
8. Finishing / exporting into DL-ready form:
    1. Copy the floodmaps of your choice (vit, vit+snunet, snunet) into a new directory at `gff-raw-data/floodmaps/final`. This is hard-coded in a number of places.
    2. Copy the exported kurosiwo floodmaps into the same directory.
    3. Run scripts/partition-the-world.py
    4. Run scripts/export-context.py
    5. Run scripts/export-local.py
    6. Run scripts/calc-norm-param.py *warning: has hard-coded outputs; maybe modify as needed*
    6. The `gff-raw-data/floodmaps/final` folder now contains your dataset, ready for deep learning. The `gff-raw-data/partitions` contains the partition information.


# Changing sampling method

In the special case that you have already obtained a list of locations/times that you want to search, you can simply replace step 6 above with many calls to `scripts/gen-floodmap.py` (e.g. a bash script). It takes a DFO flood ID and a HydroATLAS HYBAS_ID, and performs the whole generation process.

If your locations/times do not align with DFO/HydroATLAS, you will need to look at the code and figure it out yourself.


# ASF Instructions

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
