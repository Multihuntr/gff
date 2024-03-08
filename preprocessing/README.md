# SNAPpy

SNAP, using Python! Great? Hmmm... it's not quite ready yet. SNAP 10 promises a better setup, but that's not out until June.

The only `conda install` available is through `terradue` channel, but that's only version 8.0.

Recently they changed the URL of precise orbit files, but it was hard-coded into the application, so the advice is to update. I created a dockerfile with SNAP v9.0.0, but it turns I need v9.0.4, and there doesn't seem to be a way to update the commandline version from v9.0.0. So...

I had to download the entire archive of precise orbit (PO) files and patch that in to the docker container at the caching folder. I installed `lftp`, told it to ignore SSL certificates (only do this if you know what this means) and downloaded the whole thing. It's 8GB large and took like 2 hours.

```bash
mkdir Orbits
cd Orbits
mkdir ~/.lftp
echo "set ssl:verify-certificate no" > ~/.lftp/rc
lftp -c mirror https://step.esa.int/auxdata/orbits/Sentinel-1/
```

See:
- https://stackoverflow.com/q/5043239
- https://serverfault.com/a/727003