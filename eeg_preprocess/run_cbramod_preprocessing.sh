#!/usr/bin/env bash

RAW_DATA_PATH="../../data/raw_data/edf"
OUT_FOLDER="../../data/scalp_eeg_data_200HZ_np_format_cbramod"

mkdir -p "$OUT_FOLDER"

# 2) Filter cutoffs
LOW_FREQ=0.3
HIGH_FREQ=75.0

# 3) Notch
# If you want to skip notch, comment out "--apply_notch"
NOTCH_FREQ=60.0

# 4) Trimming, chunk size, amplitude threshold
TRIM_SEC=60

# 6) Parallel jobs
N_JOBS=8

# 7) Run the script
# python cbramod_preprocessing.py \
#   --raw_data_path "$RAW_DATA_PATH" \
#   --out_folder "$OUT_FOLDER" \
#   --low_freq $LOW_FREQ \
#   --high_freq $HIGH_FREQ \
#   --apply_notch \
#   --notch_freq $NOTCH_FREQ \
#   --n_jobs $N_JOBS

RAW_DATA_PATH="../../data/raw_data/test_edf"
OUT_FOLDER="../../data/cbramod_test"
mkdir -p "$OUT_FOLDER"

python cbramod_preprocessing.py \
  --raw_data_path "$RAW_DATA_PATH" \
  --out_folder "$OUT_FOLDER" \
  --low_freq $LOW_FREQ \
  --high_freq $HIGH_FREQ \
  --apply_notch \
  --notch_freq $NOTCH_FREQ \
  --trim_data \
  --n_jobs $N_JOBS