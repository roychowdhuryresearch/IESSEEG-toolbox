DATA_DIR="../../data"

mkdir -p "${DATA_DIR}/immediate_treatment_response_train_test_split"
mkdir -p "${DATA_DIR}/meaningful_treatment_response_train_test_split"

python create_treatment_response_train_test_split.py \
    --merged_csv "${DATA_DIR}/final_short_merged.csv" \
    --test_csv "${DATA_DIR}/final_test.csv" \
    --out_dir "${DATA_DIR}/immediate_treatment_response_train_test_split" \
    --random_state 42 \

python create_treatment_response_train_test_split.py \
    --merged_csv "${DATA_DIR}/final_short_merged.csv" \
    --test_csv "${DATA_DIR}/final_test.csv" \
    --out_dir "${DATA_DIR}/meaningful_treatment_response_train_test_split" \
    --random_state 42 \
    --meaningful_responder