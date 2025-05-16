ORI_META_FOLDER="../../data"

python generate_test_meta.py \
    --final_short_merged_csv "${ORI_META_FOLDER}/final_short_merged.csv" \
    --test_long_mapping_csv "${ORI_META_FOLDER}/raw_data/case_control_30min_test_sample_mapping.csv" \
    --out_csv "${ORI_META_FOLDER}/final_test.csv" 