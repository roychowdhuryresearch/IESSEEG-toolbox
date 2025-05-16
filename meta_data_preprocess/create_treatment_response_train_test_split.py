#!/usr/bin/env python
"""
split_train_test.py

Implements a two-step train–test splitting logic:

1) Read final_short_merged.csv
   - 50% random stratified by case/control => subsetA
   - The other 50% => subsetB

2) For subsetA:
   - This is trainA
   - Exclude any patient_id in trainA from final_case_control_test.csv => testA
   - Output trainA.csv, testA.csv

3) For subsetB:
   - This is trainB
   - Exclude any patient_id in trainB from final_case_control_test.csv => testB
   - Output trainB.csv, testB.csv

Columns remain in the same format:
  short_recording_id,long_recording_id,case_control_label,...
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", type=str, required=True,
                        help="Path to final_short_merged.csv (the training dataset).")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to final_case_control_test.csv (the test dataset).")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory for trainA.csv, testA.csv, trainB.csv, testB.csv.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--meaningful_responder", action="store_true", default=False,
                        help="If set, will use meaningful responder labels, otherwise use immediate responder labels.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.random_state)

    # 1) Read final_short_merged.csv
    df_merged = pd.read_csv(args.merged_csv)
    df_merged = df_merged[df_merged["pre_post_treatment_label"] == "PRE"].copy()
    # Must have columns => case_control_label, patient_id, plus the rest
    # We'll do a stratify by case_control_label => "CASE" vs. "CONTROL"

    if args.meaningful_responder:
        label_column = "meaningful_responder"
    else:
        label_column = "immediate_responder"

    if label_column not in df_merged.columns:
        raise ValueError(f"The merged_csv must contain {label_column} column.")
    if "patient_id" not in df_merged.columns:
        raise ValueError("The merged_csv must contain 'patient_id' column.")

    # 2) Stratified 50% split
    # We'll produce subsetA (50%) + subsetB (50%)
    df_merged = df_merged[df_merged[label_column].isin(["Responder", "Non-responder"])].copy()
    # dedup by patient_id
    all_patient_ids_label_df = df_merged[["patient_id", label_column]].drop_duplicates()
    all_patient_ids = all_patient_ids_label_df["patient_id"].values
    all_labels = all_patient_ids_label_df[label_column].values

    idx_full = np.arange(len(all_patient_ids))
    idx_A, idx_B = train_test_split(
        idx_full,
        test_size=0.5,   # 50% each
        stratify=all_labels, 
        random_state=args.random_state
    )

    patient_ids_A = all_patient_ids[idx_A]
    patient_ids_B = all_patient_ids[idx_B]

    dfA = df_merged[df_merged["patient_id"].isin(patient_ids_A)].copy().reset_index(drop=True)  # subsetA
    dfB = df_merged[df_merged["patient_id"].isin(patient_ids_B)].copy().reset_index(drop=True)  # subsetB

    # 3) For subsetA => trainA
    #    Then from final_case_control_test.csv => exclude patient_id in trainA => testA
    df_test = pd.read_csv(args.test_csv)
    if "patient_id" not in df_test.columns:
        raise ValueError("The test_csv must contain 'patient_id' column as well.")

    pidA = set(dfA["patient_id"].tolist())
    pidB = set(dfB["patient_id"].tolist())
    testA = df_test[df_test["patient_id"].isin(pidB)].copy()
    testB = df_test[df_test["patient_id"].isin(pidA)].copy()

    out_trainA = os.path.join(args.out_dir, f"train_A_{label_column}.csv")
    out_testA  = os.path.join(args.out_dir, f"test_A_{label_column}.csv")
    out_trainB = os.path.join(args.out_dir, f"train_B_{label_column}.csv")
    out_testB  = os.path.join(args.out_dir, f"test_B_{label_column}.csv")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dfA.to_csv(out_trainA, index=False)
    testA.to_csv(out_testA, index=False)
    dfB.to_csv(out_trainB, index=False)
    testB.to_csv(out_testB, index=False)
    print(f"trainA => {out_trainA}, shape={dfA.shape}")
    print(f"testA  => {out_testA}, shape={testA.shape}")
    print(f"trainB => {out_trainB}, shape={dfB.shape}")
    print(f"testB  => {out_testB}, shape={testB.shape}")

if __name__ == "__main__":
    main()
