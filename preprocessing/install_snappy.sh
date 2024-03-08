cd /tmp
wget -q "https://download.esa.int/step/snap/9.0/installers/esa-snap_sentinel_unix_${SNAP_VERSION}.sh"
chmod +x esa-snap_sentinel_unix_${SNAP_VERSION}.sh
./esa-snap_sentinel_unix_${SNAP_VERSION}.sh -q

# Configure snappy (don't know why they made this another step; just check for python in PATH?)
# Loop from post-link.sh in terradue's conda package:
# (the loop necessary because the command hangs once successfully complete)
/usr/local/snap/bin/snappy-conf /usr/local/bin/python | while read -r line; do
    echo "$line"
    [ "$line" = "or copy the 'snappy' module into your Python's 'site-packages' directory." ] && sleep 2 && pkill -TERM -f "nbexec"
done

# Install snappy globally
cd /root/.snap/snap-python/snappy
pip install -e .
