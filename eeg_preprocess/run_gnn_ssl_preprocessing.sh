#!/usr/bin/env bash
# ---------------------------------------------------------------
# Batch-convert a folder of EDF recordings into BIOT-ready *.npz
# (16-channel bipolar montage, 200 Hz, full-length recording)
# ---------------------------------------------------------------

# RAW_DATA_PATH="../../data/raw_data/edf"
# OUT_FOLDER="../../data/scalp_eeg_data_200HZ_np_format_gnn_ssl"
RAW_DATA_PATH="../../data/raw_data/case_control_30min_test"
OUT_FOLDER="../../data/scalp_eeg_data_200HZ_np_format_case_control_30min_test_gnn_ssl"

python gnn_ssl_preprocessing.py \
    --raw_data_path   "$RAW_DATA_PATH" \
    --out_folder   "$OUT_FOLDER" 
