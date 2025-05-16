#!/usr/bin/env python
"""
gen_inference_meta.py

Generates a metadata CSV for inference, merging:
  - case_control_30min_test_sample_mapping.csv: (file_id, original_name)
       => "file_id" is an integer ID
       => "original_name" is the EDF path, e.g. "/mnt/.../423JGC_window_1_start_127_end_157.edf"

  - long_eeg_meta.csv: (patient_id, recording_id, label)
       => "recording_id" might be "423JGC", with a label (0=control, 1=case)

We parse out the "recording_id" from the EDF path in "original_name", 
then look it up in "long_eeg_meta.csv" to get an optional label. 
Finally, we write out "test_inference_metadata.csv" with columns:
  [file_id, edf_path, recording_id, label]

Example usage:
  python gen_inference_meta.py \
    --mapping_csv "/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/case_control_30min_test_sample_mapping.csv" \
    --meta_csv "/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/long_eeg_meta.csv" \
    --out_csv "/mnt/SSD1/mingjian/Neurips 2025/data/case_control_test.csv"
"""

import os
import argparse
import pandas as pd
import numpy as np
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
pd.set_option('mode.chained_assignment', None)  # Suppress pandas warnings

def get_ad_sk_annotation(ad_path="/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/sample_mapping_AD_test.csv", sk_path="/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/sample_mapping_SK.csv"):
    mapping = {"CASE": 1, "CONTROL": 0, "ARTIFACT": -1, "?": -1, "CONTROL?": 0}
    ad_df = pd.read_csv(ad_path)
    ad_df = ad_df.sort_values(by=["file_id"]).reset_index(drop=True)

    ad_df["seg_name"] = ad_df["original_name"].apply(lambda x: x.split("/")[-1].split(".")[0])
    ad_df["rec_id"]   = ad_df["seg_name"].apply(lambda x: x.split("_")[0])
    ad_df["start_ind"]= ad_df["seg_name"].apply(lambda x: int(x.split("_")[4]) * 60 * 200)
    ad_df["end_ind"]  = ad_df["seg_name"].apply(lambda x: int(x.split("_")[6]) * 60 * 200)
    ad_df["judgement"]= ad_df["CaseorControl"].apply(lambda x: mapping.get(str(x).upper(), -1))

    # Convert AD columns to int
    ad_df["start_ind"] = pd.to_numeric(ad_df["start_ind"], errors="coerce").astype(int)
    ad_df["end_ind"]   = pd.to_numeric(ad_df["end_ind"], errors="coerce").astype(int)
    ad_df["file_id"]   = pd.to_numeric(ad_df["file_id"], errors="coerce").astype(int)

    # Load SK
    sk_df = pd.read_csv(sk_path)
    sk_df = sk_df.sort_values(by=["file_id"]).reset_index(drop=True)
    sk_df["seg_name"] = sk_df["original_name"].apply(lambda x: x.split("/")[-1].split(".")[0])
    sk_df["rec_id"]   = sk_df["seg_name"].apply(lambda x: x.split("_")[0])
    sk_df["start_ind"]= sk_df["seg_name"].apply(lambda x: int(x.split("_")[4]) * 60 * 200)
    sk_df["end_ind"]  = sk_df["seg_name"].apply(lambda x: int(x.split("_")[6]) * 60 * 200)
    sk_df["judgement"]= sk_df["case or control"].apply(lambda x: mapping.get(str(x).upper(), -1))

    # Convert SK columns to int
    sk_df["start_ind"] = pd.to_numeric(sk_df["start_ind"], errors="coerce").astype(int)
    sk_df["end_ind"]   = pd.to_numeric(sk_df["end_ind"], errors="coerce").astype(int)
    sk_df["file_id"]   = pd.to_numeric(sk_df["file_id"], errors="coerce").astype(int)

    # Also get ground truth from a big CSV
    long_eeg_labels_df = pd.read_csv("/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/long_eeg_meta.csv")
    recid_to_label = dict(zip(long_eeg_labels_df["recording_id"], long_eeg_labels_df["label"]))
    ad_df["label"] = ad_df["rec_id"].apply(lambda x: recid_to_label.get(x, 0))
    sk_df["label"] = sk_df["rec_id"].apply(lambda x: recid_to_label.get(x, 0))

    merged_df = pd.merge(ad_df, sk_df, on=["file_id", "original_name", "seg_name", "rec_id", "start_ind", "end_ind", "label"], how="inner", suffixes=("_AD", "_SK"))
    merged_df = merged_df.drop(columns=["CaseorControl", "Unnamed: 3_AD", "case or control", "Unnamed: 3_SK"])
    return merged_df

def get_shaun_patch_annotation(shaun_path="/mnt/SSD1/mingjian/Neurips 2025/data/raw_data/test_annotation_patch.csv"):
    shaun_df = pd.read_csv(shaun_path)
    shaun_df["short_recording_id"] = shaun_df["file_id"].apply(lambda x: int(x) + 200)
    mapping = {"CASE": 1, "CONTROL": 0, "ARTIFACT": -1, "?": -1}
    shaun_df["judgement_Shaun"] = shaun_df["Shaun Annotation"].apply(lambda x: mapping.get(str(x).upper(), -1))
    shaun_df["long_recording_id"] = shaun_df["original_name"].apply(lambda x: x.split("/")[-1].split("_RESAMPLED")[0])
    shaun_df = shaun_df[["short_recording_id", "long_recording_id", "judgement_Shaun"]].copy()
    return shaun_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_short_merged_csv", type=str, required=True,
                        help="final_short_merged.csv with short_recording_id, long_recording_id, case_control_label, pre_post_treatment_label, sleep_awake_label, patient_id, Interval, AgeAtEEG1y, AOOmo, LeadtimeD, LeadtimeUKISS")
    parser.add_argument("--test_long_mapping_csv", type=str, required=True,
                        help="case_control_30min_test_sample_mapping.csv with file_id, original_name")
    parser.add_argument("--out_csv", type=str, default="test_inference_metadata.csv",
                        help="Output CSV file with columns [file_id, edf_path, recording_id, label]")
    args = parser.parse_args()

    final_short_merged_df = pd.read_csv(args.final_short_merged_csv)
    final_short_merged_df_columns = final_short_merged_df.columns.tolist()
    test_long_mapping_df = pd.read_csv(args.test_long_mapping_csv)

    test_long_mapping_df["long_recording_id"] = test_long_mapping_df["original_name"].apply(lambda x: os.path.basename(x).split("_")[0])
    final_short_merged_df = final_short_merged_df[["long_recording_id", "case_control_label", "pre_post_treatment_label", "patient_id", "Interval", "AgeAtEEG1y", "AOOmo", "LeadtimeD", "LeadtimeUKISS", "immediate_responder", "meaningful_responder"]].copy()
    # dedup
    final_short_merged_df = final_short_merged_df.drop_duplicates(subset=["long_recording_id"])

    merged_df = pd.merge(test_long_mapping_df, final_short_merged_df, on="long_recording_id", how="left")
    merged_df["short_recording_id"] = merged_df["file_id"].copy()
    merged_df["sleep_awake_label"] = ["UNKNOWN"] * len(merged_df)

    # sort column in the same order as final_short_merged_df
    merged_df = merged_df[final_short_merged_df_columns]
    merged_gt_df = merged_df.copy()
    merged_doctor_df = get_ad_sk_annotation()
    merged_doctor_df = merged_doctor_df[["file_id", "judgement_AD", "judgement_SK"]].copy()
    merged_doctor_df = merged_doctor_df.rename(columns={"file_id": "short_recording_id"})
    merged_doctor_df = merged_doctor_df.merge(merged_gt_df, on="short_recording_id", how="inner").sort_values(by=["short_recording_id"]).reset_index(drop=True)

    shaun_df = get_shaun_patch_annotation()
    shaun_with_meta_df = pd.merge(shaun_df, final_short_merged_df, on="long_recording_id", how="inner").sort_values(by=["short_recording_id"]).reset_index(drop=True)
    shaun_with_meta_df["sleep_awake_label"] = ["UNKNOWN"] * len(shaun_with_meta_df)

    # find judgement_AD != -1
    merged_doctor_df["any_noise"] = merged_doctor_df.apply(lambda x: 1 if x["judgement_AD"] == -1 or x["judgement_SK"] == -1 else 0, axis=1)
    merged_doctor_1st_path_no_noise_df = merged_doctor_df[merged_doctor_df["any_noise"] == 0].copy().reset_index(drop=True)
    merged_doctor_1st_path_noise_df = merged_doctor_df[merged_doctor_df["any_noise"] == 1].copy().reset_index(drop=True)
  
    # for each with noise, we can find a replacement from shaun that is not noise
    replacement_count = merged_doctor_1st_path_noise_df["long_recording_id"].value_counts()
    clean_shaun_df = shaun_with_meta_df[shaun_with_meta_df["judgement_Shaun"] != -1].copy().reset_index(drop=True)
    # shuffle order of clean_shaun_df
    clean_shaun_df = clean_shaun_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Find replacements for noisy samples
    replacements = []
    for long_recording_id, count in replacement_count.items():
        # Get available replacements for this recording_id
        available = clean_shaun_df[clean_shaun_df["long_recording_id"] == long_recording_id]
        if len(available) >= count:
            # Take first 'count' rows and remove them from clean_shaun_df
            selected = available.head(count)
            clean_shaun_df = clean_shaun_df.drop(selected.index)
            replacements.append(selected)
        else:
            print(f"Warning: Not enough replacements for {long_recording_id} (needed {count}, found {len(available)})")
    
    row_to_add_df = pd.concat(replacements, ignore_index=True)
    row_to_add_df["human_label"] = row_to_add_df["judgement_Shaun"]
    merged_doctor_1st_path_no_noise_df["human_label"] = merged_doctor_1st_path_no_noise_df["judgement_AD"]

    # keep short_recording_id, long_recording_id,case_control_label,pre_post_treatment_label,sleep_awake_label,patient_id,Interval,AgeAtEEG1y,AOOmo,LeadtimeD,LeadtimeUKISS,immediate_responder,meaningful_responder, human_label
    full_resulted_clean_df = pd.concat([merged_doctor_1st_path_no_noise_df, row_to_add_df], ignore_index=True).sort_values(by=["short_recording_id"]).reset_index(drop=True)
    full_resulted_clean_df = full_resulted_clean_df[["short_recording_id", "long_recording_id", "case_control_label", "pre_post_treatment_label", "sleep_awake_label", "patient_id", "Interval", "AgeAtEEG1y", "AOOmo", "LeadtimeD", "LeadtimeUKISS", "immediate_responder", "meaningful_responder", "human_label"]].copy()
    full_resulted_clean_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()