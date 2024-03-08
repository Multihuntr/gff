# Matching KuroSiwo's preprocessing

Method to compare:
1. I opened the catalogue, and chose a patch that was in the 01.tar.gz archive (didn't want to download all of KuroSiwo). Specifically, I chose a patch that had a recorded flood percentage a little more than 50%, but less than 70% (using info.json's "pflood" property).
2. I selected a tile somewhere in spain taken in 2014-11
    ```
    ID (from info.json's "grid_id")
    08ed4c4d-6530-59c3-9a55-f6766341f728

    Folder
    118/01/08ed4c4d653059c39a55f6766341f728/

    Tile
    SL1_IVH_118_01_20141111.tif

    DEM
    MK0_DEM_118_01_20141111.tif
    ```
3. Use `asf_search` to download S1 images.
    1. The `info.json` suggests a "s1_id", but I couldn't search by ID. Searching by time gave some results, so I just used those.
    2. I asked for `GRD`, but the results had a few different types. I found a `GRD_HD`, so I used that
    3. Put your credentials in `.asf_search` separated by `';'`.
    ```python
    # Not necessarily working code:
    from pathlib import Path
    import asf_search as asf
    start_date = datetime.datetime.fromisoformat(basin_shp["BEGAN"])
    end_date = datetime.datetime.fromisoformat(basin_shp["ENDED"])

    buffer = datetime.timedelta(days=10)
    search_results = asf.geo_search(
        intersectsWith=basin_shp.geometry.wkt,
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.AMPLITUDE_GRD,
        start=start_date - buffer,
        end=end_date,
    )
    result = [res for res in search_results if res.properties['processingLevel'] == 'GRD_HD'][0]
    with open(".asf_auth") as f:
        username, password = f.read().strip().split(";")
    session = asf.ASFSession().auth_with_creds(username, password)
    p = Path.home() / 'data' / 'kurosiwo'
    p.mkdir(exist_ok=True, parents=True)
    result.download(path=p, session=session)
    ```
4. Produce preprocessed tile
    ```bash
    docker build -t esa-snappy . && docker run --rm -it -v $PWD/dem:/root/.snap/auxdata/dem -v $PWD/Orbits:/root/.snap/auxdata/Orbits -v $HOME/data/kurosiwo:/data esa-snappy "/data/S1A_IW_GRDH_1SDV_20141111T060859_20141111T060924_003230_003BAC_3091.zip" "POLYGON ((-1.2382827033945543 41.84155753549106, -1.2382827033945543 41.856546711119194, -1.218160441030277 41.856546711119194, -1.2382827033945543 41.84155753549106))" "/data/outtile.tif"
    ```
5. Load `outtile.tif` and `SL1_IVH_118_01_20141111.tif` into QGIS and find the difference
6. Update preprocessing and repeat from 4 until they match.


The chosen settings in `preprocess.py` are the result of some guesswork. The resulting tile looks almost the same, just resampling and some extra spots in their data. Otherwise effectively identical.

TODO: DEM is definitely the correct DEM, but it is offset by a weird amount. Figure out why.
