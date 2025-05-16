#!/usr/bin/env bash

# Example usage of preprocessing.py with custom parameters

RAW_DATA_PATH="../../data/raw_data/edf"
OUT_FOLDER="../../data/scalp_eeg_data_200HZ_np_format"
mkdir -p "$OUT_FOLDER"

LOW_FREQ=0.5
HIGH_FREQ=50.0

# If we want to skip notch, remove --apply_notch
# Example: --apply_notch is just a flag (no arguments)
# python preprocessing.py \
#   --raw_data_path "$RAW_DATA_PATH" \
#   --out_folder "$OUT_FOLDER" \
#   --low_freq $LOW_FREQ \
#   --high_freq $HIGH_FREQ \
#   --reference_scheme "bipolar" \
#   --n_jobs 8

RAW_DATA_PATH="../../data/raw_data/test_edf"
OUT_FOLDER="../../data/baseline_test"
mkdir -p "$OUT_FOLDER"

python preprocessing.py \
  --raw_data_path "$RAW_DATA_PATH" \
  --out_folder "$OUT_FOLDER" \
  --low_freq $LOW_FREQ \
  --high_freq $HIGH_FREQ \
  --reference_scheme "bipolar" \
  --n_jobs 8
