# !/bin/bash

set -e

export TQDM_DISABLE="True"
export DATA_FOLDER="~/data/gff-export"

function run_eval {
    python evaluate-extended.py paperruns/$1 ~/data/gff-export/worldcover ~/NFS_era5/HydroATLAS --device cuda:1 --blockout_worldcover_water --blockout_ks_pw --coast_buffer 0.1 -o data_folder=$DATA_FOLDER &
    echo "Running $1..."
}

declare -a BASE_MODELS=(
    "metnet"
    "utae"
    "recunet"
    "3dunet"
)

declare -a ABLATION_MODELS=(
    "recunet_no_hand"
    "recunet_and_no_dem"
    "recunet_and_no_hydroatlas"
    "recunet_and_no_era5l"
    "recunet_no_s1"
)

for model in "${BASE_MODELS[@]}"; do
    for i in $(seq 0 4); do run_eval "${model}_$i"; done
    wait
done

for model in "${ABLATION_MODELS[@]}"; do
    for i in $(seq 0 4); do run_eval "${model}_$i"; done
    wait
done
