#!/bin/bash

COLUMNS_TO_USE="coadd_object_id,mag_auto_g_dered,mag_auto_r_dered,mag_auto_i_dered,mag_auto_z_dered,mag_auto_y_dered,gmr,rmi,imz,zmy" # adjust as needed
N_CLIENTS=10

for i in $(seq 0 $((N_CLIENTS-1)))
do
    python utils/prepare_data.py \
        --input_csv client_${i}.csv \
        --columns "$COLUMNS_TO_USE" \
        --randomize 0 \
        --out_path processed/client_${i}_processed.csv
done

echo "All client files processed."
