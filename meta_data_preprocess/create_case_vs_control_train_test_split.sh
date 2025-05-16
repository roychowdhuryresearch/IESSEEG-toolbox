DATA_DIR="../../data"

mkdir -p "${DATA_DIR}/case_control_train_test_split"

python create_case_vs_control_train_test_split.py \
    --merged_csv "${DATA_DIR}/final_short_merged.csv" \
    --test_csv "${DATA_DIR}/final_test.csv" \
    --out_dir "${DATA_DIR}/case_control_train_test_split" \
    --random_state 42