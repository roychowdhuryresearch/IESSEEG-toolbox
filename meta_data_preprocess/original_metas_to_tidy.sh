ORI_META_FOLDER="../../data/raw_data"
DATA_ROOT="../../data/scalp_eeg_data_200HZ_np_format"

python original_metas_to_tidy.py \
  --meta_eeg "${ORI_META_FOLDER}/MetaEEG.csv" \
  --meta_ming "${ORI_META_FOLDER}/MetaDataForMingjian.csv" \
  --long_map "${ORI_META_FOLDER}/long_short_mappings.csv" \
  --npz_dir "${DATA_ROOT}" \
  --out_csv "${ORI_META_FOLDER}/../final_short_merged.csv"
