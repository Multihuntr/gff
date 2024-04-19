# ~/bin/bash

set -e

batch_1_url="https://www.dropbox.com/scl/fi/vcmzontrqlywkb6827xr4/01.tar.gz?rlkey=p08coyujc8jtqc9omp8pee4u9&dl=0"
batch_2_url="https://www.dropbox.com/scl/fi/9eiskq6bvwvtkcgar7fau/02.tar.gz?rlkey=jjd8uitqa31wmivcirg84q83o&dl=0"
batch_3_url="https://www.dropbox.com/scl/fi/wal95mj2lgrhn5vfv82ru/03.tar.gz?rlkey=fxgfccf16v8bmawhkrlivat3s&dl=0"
batch_4_url="https://www.dropbox.com/scl/fi/r6zfyfkauctuvdgondul0/04.tar.gz?rlkey=zq4rnjljvyph7udidfntgn5p1&dl=0"
batch_5_url="https://www.dropbox.com/scl/fi/rwvumfstxp27k1akv0hv5/05.tar.gz?rlkey=ukysm10tljygxourdkbwzkr2k&dl=0"
batch_6_url="https://www.dropbox.com/scl/fi/uqobj3knewqm5y6a9cax3/06.tar.gz?rlkey=oub1c8905gwuthjlvz6lydib1&dl=0"
batch_7_url="https://www.dropbox.com/scl/fi/7uamc6g165so3mfv9n411/07.tar.gz?rlkey=53x6tmmmh7iqbbzizu64vonco&dl=0"
batch_8_url="https://www.dropbox.com/scl/fi/ejztwj7ib545qsk7kw8ou/08.tar.gz?rlkey=tbl0bk3kzqe77ugs5fucdpkd4&dl=0"

PYTHONPATH='.'

function get_extract_remove {
    wget -O $1 "$2"
    python scripts/ks-extract-just-labelled.py $1
    rm $1
}

get_extract_remove "$1/01.tar.gz" $batch_1_url
get_extract_remove "$1/02.tar.gz" $batch_2_url
get_extract_remove "$1/03.tar.gz" $batch_3_url
get_extract_remove "$1/04.tar.gz" $batch_4_url
get_extract_remove "$1/05.tar.gz" $batch_5_url
get_extract_remove "$1/06.tar.gz" $batch_6_url
get_extract_remove "$1/07.tar.gz" $batch_7_url
get_extract_remove "$1/08.tar.gz" $batch_8_url

python scripts/ks-merge-labels.py $1 $2
