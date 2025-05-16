#!/usr/bin/env bash
# ---------------------------------------------------------------
# Batch-convert a folder of EDF recordings into BIOT-ready *.npz
# (16-channel bipolar montage, 200 Hz, full-length recording)
# ---------------------------------------------------------------

RAW_DATA_PATH="../../data/raw_data/test_edf"
OUT_FOLDER="../../data/biot_test"

python biot_preprocessing.py \
    --raw_data_path   "$RAW_DATA_PATH" \
    --out_folder   "$OUT_FOLDER" \
    --num_biot_channels 18

# RAW_DATA_PATH="../../data/raw_data/edf"
# OUT_FOLDER="../../data/scalp_eeg_data_200HZ_np_format_biot"
# python biot_preprocessing.py \
#     --raw_data_path   "$RAW_DATA_PATH" \
#     --out_folder   "$OUT_FOLDER" \
#     --num_biot_channels 18