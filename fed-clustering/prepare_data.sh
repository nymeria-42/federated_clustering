#!/bin/bash

DATASET_PATH="/tmp/nvflare/dataset/des.csv"
script_dir="$( dirname -- "$0"; )";


python3 "${script_dir}"/utils/prepare_data.py \
    --input_csv /home/nymeria/federated_clustering/fed-clustering/des.csv \
    --randomize 1 \
    --out_path ${DATASET_PATH}
echo "Data loaded and saved in ${DATASET_PATH}"
