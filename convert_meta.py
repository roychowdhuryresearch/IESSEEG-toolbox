#!/usr/bin/env python
"""
build_final_metadata.py

This script produces a single 'final_open_source_metadata.csv' that you can
publish alongside your dataset. It merges:
  - MetaEEG.csv
  - MetaDataForMingjian.csv
  - long_eeg_meta.csv
builds the dictionaries for awake/sleep recordings, case vs. control, etc.,
and finally creates one table with columns:
  [scenario, filename, label, patient_id, AgeAtEEG1y, LeadtimeD, ...]

Users can load this CSV in baseline scripts to know:
  - Which file belongs to which scenario
  - Which patient each file belongs to
  - The case/control label, and optional demographic info.
"""

import os
import numpy as np
import pandas as pd

def find_file_location_by_recording_id(data_dir, eeg_recording_id):
    """
    Determines the correct file name (without .npz) for a given recording ID,
    handling appended 'A' or uppercase logic. If neither form exists, raises.
    
    Returns just the 'base filename' that you'd use in your CSV's 'filename' column.
    For example:
       EEGCODE -> "EEGCODE"
       EEGCODE with appended A -> "EEGCODEA"
    """
    # Potential actual file paths (with .npz extension)
    data_path1 = os.path.join(data_dir, f"{eeg_recording_id.upper()}.npz")
    data_path2 = os.path.join(data_dir, f"{eeg_recording_id.upper()}A.npz")
    
    # Check if there's a file named EEGCODE.npz
    if os.path.exists(data_path1):
        # So the final base filename is EEGCODE (uppercase)
        return eeg_recording_id.upper()
    
    # If there's a file named EEGCODEA.npz
    elif os.path.exists(data_path2):
        return eeg_recording_id.upper() + "A"
    
    # If the input ends with 'A' but the file is actually EEGCODE (no A)
    # e.g. input was "XYZ12A", actual file is "XYZ12.npz"
    elif eeg_recording_id.upper().endswith("A"):
        candidate = eeg_recording_id.upper()[:-1]  # remove 'A'
        alt_path = os.path.join(data_dir, f"{candidate}.npz")
        if os.path.exists(alt_path):
            return candidate
    
    # If none of the above matched, raise or handle as you prefer
    raise FileNotFoundError(
        f"Recording ID {eeg_recording_id} not found in either {data_path1} or {data_path2}"
    )

def main():
    # ------------------------------------------------------------------------
    # 1) Load original CSV files
    #    Make sure these paths point to your actual CSV locations
    # ------------------------------------------------------------------------
    data_path = os.path.abspath("../data/raw_data/")

    meta_df_path = os.path.join(data_path, "MetaEEG.csv")
    meta_addition_df_path = os.path.join(data_path, "MetaDataForMingjian.csv")
    long_eeg_meta_path = os.path.join(data_path, "long_eeg_meta.csv")

    meta_df = pd.read_csv(meta_df_path)
    meta_addition_df = pd.read_csv(meta_addition_df_path)
    long_eeg_meta_df = pd.read_csv(long_eeg_meta_path)

    # ------------------------------------------------------------------------
    # 2) Build dictionaries exactly as in your original code snippet
    #    (Slightly refactored for clarity)
    # ------------------------------------------------------------------------
    total_recordings_ids_dict = {}
    total_labels_dict = {}
    total_patient_id_dict = {}

    case_df = meta_df[meta_df["CASE"] == 1]
    control_df = meta_df[meta_df["CASE"] == 0]

    sleep_case_recordings_ids = case_df["CodePreSleep1"].values.tolist() + case_df["CodePreSleep2"].values.tolist()
    sleep_control_recordings_ids = control_df["CodePreSleep1"].values.tolist() + control_df["CodePreSleep2"].values.tolist()

    awake_case_recordings_ids = case_df["CodePreWake1"].values.tolist() + case_df["CodePreWake2"].values.tolist()
    awake_control_recordings_ids = control_df["CodePreWake1"].values.tolist() + control_df["CodePreWake2"].values.tolist()

    total_recordings_ids_dict = {"sleep": sleep_case_recordings_ids + sleep_control_recordings_ids, 
                                "awake": [id for id in (awake_case_recordings_ids + awake_control_recordings_ids)], 
                                "gentle_cleaned_awake": ["gentle_cleaned_" + id for id in (awake_case_recordings_ids + awake_control_recordings_ids)], 
                                "aggressive_cleaned_awake": ["aggressive_cleaned_" + id for id in (awake_case_recordings_ids + awake_control_recordings_ids)]
                                }
    # total_recordings_ids_dict = {"sleep": sleep_case_recordings_ids + sleep_control_recordings_ids, "awake": [id for id in (awake_case_recordings_ids + awake_control_recordings_ids)]}
    total_labels_dict = {"sleep": [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids),
                    "awake": [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids), 
                    "gentle_cleaned_awake": [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids), 
                    "aggressive_cleaned_awake": [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids)
                    }

    total_patient_id_dict = {"sleep": case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2,
                        "awake": case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2, 
                        "gentle_cleaned_awake": case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2, 
                        "aggressive_cleaned_awake": case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2
                        }

    # Responder Sleep 
    meta_addition_df = pd.read_csv('{}/MetaDataForMingjian.csv'.format(data_path))
    meta_addition_df = meta_addition_df[~meta_addition_df["ResponderAtEEG01"].isnull()].reset_index(drop=True)

    case_df = meta_addition_df[meta_addition_df["ResponderAtEEG01"] == 1]
    control_df = meta_addition_df[meta_addition_df["ResponderAtEEG01"] == 0]

    sleep_case_recordings_ids = case_df["CodePreSleep1a"].values.tolist() + case_df["CodePreSleep2a"].values.tolist()
    sleep_control_recordings_ids = control_df["CodePreSleep1a"].values.tolist() + control_df["CodePreSleep2a"].values.tolist()

    awake_case_recordings_ids = case_df["CodePreWake1a"].values.tolist() + case_df["CodePreWake2a"].values.tolist()
    awake_control_recordings_ids = control_df["CodePreWake1a"].values.tolist() + control_df["CodePreWake2a"].values.tolist()

    total_recordings_ids_dict["pre_sleep_responder"] = sleep_case_recordings_ids + sleep_control_recordings_ids
    total_labels_dict["pre_sleep_responder"] = [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids)
    total_patient_id_dict["pre_sleep_responder"] = case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2

    sleep_case_recordings_ids = case_df["CodePostSleep1a"].values.tolist() + case_df["CodePostSleep2a"].values.tolist()
    sleep_control_recordings_ids = control_df["CodePostSleep1a"].values.tolist() + control_df["CodePostSleep2a"].values.tolist()

    total_recordings_ids_dict["post_sleep_responder"] = sleep_case_recordings_ids + sleep_control_recordings_ids
    total_labels_dict["post_sleep_responder"] = [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids)
    total_patient_id_dict["post_sleep_responder"] = case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2

    # Responder Awake
    awake_case_recordings_ids = case_df["CodePreWake1a"].values.tolist() + case_df["CodePreWake2a"].values.tolist()
    awake_control_recordings_ids = control_df["CodePreWake1a"].values.tolist() + control_df["CodePreWake2a"].values.tolist()

    total_recordings_ids_dict["pre_awake_responder"] = awake_case_recordings_ids + awake_control_recordings_ids
    total_labels_dict["pre_awake_responder"] = [1] * len(awake_case_recordings_ids) + [0] * len(awake_control_recordings_ids)
    total_patient_id_dict["pre_awake_responder"] = case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2

    awake_case_recordings_ids = case_df["CodePostWake1a"].values.tolist() + case_df["CodePostWake2a"].values.tolist()
    awake_control_recordings_ids = control_df["CodePostWake1a"].values.tolist() + control_df["CodePostWake2a"].values.tolist()

    total_recordings_ids_dict["post_awake_responder"] = awake_case_recordings_ids + awake_control_recordings_ids
    total_labels_dict["post_awake_responder"] = [1] * len(awake_case_recordings_ids) + [0] * len(awake_control_recordings_ids)
    total_patient_id_dict["post_awake_responder"] = case_df.index.values.tolist() * 2 + control_df.index.values.tolist() * 2

    # Responder Sleep (pre and post)
    sleep_case_recordings_ids = case_df["CodePreSleep1a"].values.tolist() + case_df["CodePreSleep2a"].values.tolist() + case_df["CodePostSleep1a"].values.tolist() + case_df["CodePostSleep2a"].values.tolist()
    sleep_control_recordings_ids = control_df["CodePreSleep1a"].values.tolist() + control_df["CodePreSleep2a"].values.tolist() + control_df["CodePostSleep1a"].values.tolist() + control_df["CodePostSleep2a"].values.tolist()

    total_recordings_ids_dict["pre_post_sleep_responder"] = sleep_case_recordings_ids + sleep_control_recordings_ids
    total_labels_dict["pre_post_sleep_responder"] = [1] * len(sleep_case_recordings_ids) + [0] * len(sleep_control_recordings_ids)
    total_patient_id_dict["pre_post_sleep_responder"] = case_df.index.values.tolist() * 4 + control_df.index.values.tolist() * 4

    # Responder Awake (pre and post)
    awake_case_recordings_ids = case_df["CodePreWake1a"].values.tolist() + case_df["CodePreWake2a"].values.tolist() + case_df["CodePostWake1a"].values.tolist() + case_df["CodePostWake2a"].values.tolist()
    awake_control_recordings_ids = control_df["CodePreWake1a"].values.tolist() + control_df["CodePreWake2a"].values.tolist() + control_df["CodePostWake1a"].values.tolist() + control_df["CodePostWake2a"].values.tolist()

    total_recordings_ids_dict["pre_post_awake_responder"] = awake_case_recordings_ids + awake_control_recordings_ids
    total_labels_dict["pre_post_awake_responder"] = [1] * len(awake_case_recordings_ids) + [0] * len(awake_control_recordings_ids)
    total_patient_id_dict["pre_post_awake_responder"] = case_df.index.values.tolist() * 4 + control_df.index.values.tolist() * 4

    # sensitivity analysis
    meta_addition_df = pd.read_csv('{}/MetaDataForMingjian.csv'.format(data_path))
    control20 = meta_addition_df[meta_addition_df["CONTROL20"] == 1].reset_index(drop=True)
    case40 = meta_addition_df[meta_addition_df["CASE40"] == 1].reset_index(drop=True)

    sensitivity_sleep_case_recordings_ids = case40["CodePreSleep1a"].values.tolist() + case40["CodePreSleep2a"].values.tolist()
    sensitivity_sleep_control_recordings_ids = control20["CodePreSleep1a"].values.tolist() + control20["CodePreSleep2a"].values.tolist()

    sensitivity_awake_case_recordings_ids = case40["CodePreWake1a"].values.tolist() + case40["CodePreWake2a"].values.tolist()
    sensitivity_awake_control_recordings_ids = control20["CodePreWake1a"].values.tolist() + control20["CodePreWake2a"].values.tolist()

    total_recordings_ids_dict["sensitivity_sleep"] = sensitivity_sleep_case_recordings_ids + sensitivity_sleep_control_recordings_ids
    total_labels_dict["sensitivity_sleep"] = [1] * len(sensitivity_sleep_case_recordings_ids) + [0] * len(sensitivity_sleep_control_recordings_ids)

    total_recordings_ids_dict["sensitivity_awake"] = sensitivity_awake_case_recordings_ids + sensitivity_awake_control_recordings_ids
    total_labels_dict["sensitivity_awake"] = [1] * len(sensitivity_awake_case_recordings_ids) + [0] * len(sensitivity_awake_control_recordings_ids)

    original_recording_id_to_patient_id_map = {recording_id: patient_id for patient_id, recording_id in zip(total_patient_id_dict["sleep"], total_recordings_ids_dict["sleep"])}
    original_recording_id_to_patient_id_map.update({recording_id: patient_id for patient_id, recording_id in zip(total_patient_id_dict["awake"], total_recordings_ids_dict["awake"])})

    total_patient_id_dict["sensitivity_sleep"] = [original_recording_id_to_patient_id_map[recording_id] for recording_id in total_recordings_ids_dict["sensitivity_sleep"]]
    total_patient_id_dict["sensitivity_awake"] = [original_recording_id_to_patient_id_map[recording_id] for recording_id in total_recordings_ids_dict["sensitivity_awake"]]

    # sleep vs awake classification
    sleep_recording_ids = meta_df["CodePreSleep1"].values.tolist() + meta_df["CodePreSleep2"].values.tolist()
    awake_recording_ids = meta_df["CodePreWake1"].values.tolist() + meta_df["CodePreWake2"].values.tolist()

    total_recordings_ids_dict["sleep_awake"] = sleep_recording_ids + awake_recording_ids
    total_labels_dict["sleep_awake"] = [1] * len(sleep_recording_ids) + [0] * len(awake_recording_ids)
    total_patient_id_dict["sleep_awake"] = meta_df.index.values.tolist() * 2 + meta_df.index.values.tolist() * 2

    # Combined sleep + awake case/control classification
    total_recordings_ids_dict["combined_sleep_awake_case_control"] = total_recordings_ids_dict["sleep"] + total_recordings_ids_dict["awake"]
    total_labels_dict["combined_sleep_awake_case_control"] = total_labels_dict["sleep"] + total_labels_dict["awake"]
    total_patient_id_dict["combined_sleep_awake_case_control"] = total_patient_id_dict["sleep"] + total_patient_id_dict["awake"]

    # Long EEG
    long_eeg_meta_df = pd.read_csv('{}/long_eeg_meta.csv'.format(data_path))
    long_eeg_patient_id = np.array(long_eeg_meta_df["patient_id"].values.tolist())
    long_eeg_recording_id = np.array(long_eeg_meta_df["recording_id"].values.tolist())
    long_eeg_labels = np.array(long_eeg_meta_df["label"].values.tolist())
    total_patient_id_dict["long_eeg"] = long_eeg_patient_id
    total_recordings_ids_dict["long_eeg"] = long_eeg_recording_id
    total_labels_dict["long_eeg"] = long_eeg_labels

    # Long EEG p5
    index = (long_eeg_recording_id == "564SCY")
    total_patient_id_dict["long_eeg_p5"] = long_eeg_patient_id[index]
    total_recordings_ids_dict["long_eeg_p5"] = long_eeg_recording_id[index]
    total_labels_dict["long_eeg_p5"] = long_eeg_labels[index]

    # Convert everything to numpy array
    total_recordings_ids_dict = {key: np.array(value) for key, value in total_recordings_ids_dict.items()}
    total_labels_dict = {key: np.array(value) for key, value in total_labels_dict.items()}
    total_patient_id_dict = {key: np.array(value) for key, value in total_patient_id_dict.items()}

    # Purified EEG sleep + awake
    purification_set = [2, 3, 5, 6, 14, 15, 16, 18, 20, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 38, 43, 45, 47, 48, 49]
    discard_index = np.isin(total_patient_id_dict["combined_sleep_awake_case_control"], purification_set)

    total_patient_id_dict["purified_sleep_awake_case_control"] = total_patient_id_dict["combined_sleep_awake_case_control"][~discard_index]
    total_recordings_ids_dict["purified_sleep_awake_case_control"] = total_recordings_ids_dict["combined_sleep_awake_case_control"][~discard_index]
    total_labels_dict["purified_sleep_awake_case_control"] = total_labels_dict["combined_sleep_awake_case_control"][~discard_index]

    # Purification exclusion
    total_patient_id_dict["purification_exlusion_sleep_awake_case_control"] = total_patient_id_dict["combined_sleep_awake_case_control"][discard_index]
    total_recordings_ids_dict["purification_exlusion_sleep_awake_case_control"] = total_recordings_ids_dict["combined_sleep_awake_case_control"][discard_index]
    total_labels_dict["purification_exlusion_sleep_awake_case_control"] = total_labels_dict["combined_sleep_awake_case_control"][discard_index]

    # 30min VAE purification work EDFs

    total_patient_id_dict["VAE_30min_purification"] = np.arange(0, 200).astype(int)
    total_recordings_ids_dict["VAE_30min_purification"] = np.arange(0, 200).astype(str)
    total_labels_dict["VAE_30min_purification"] = np.ones(200).astype(int)

    # ------------------------------------------------------------------------
    # 4) Build a single DataFrame with columns:
    #    [scenario, filename, label, patient_id]
    # ------------------------------------------------------------------------
    master_rows = []

    # Path to the directory containing your .npz data (for checking existence).
    # You might have a different location for these .npz files, so adjust as needed.
    npz_dir = os.path.abspath("../data/scalp_eeg_data_200HZ_np_format/")

    for scenario in ["sleep", "awake", "combined_sleep_awake_case_control"]:
        rec_ids = total_recordings_ids_dict[scenario]
        labs = total_labels_dict[scenario]
        pids = total_patient_id_dict[scenario]

        for recording_id, lbl, pid in zip(rec_ids, labs, pids):
            # We now use find_file_location_by_recording_id to standardize
            # the naming in 'filename' so that it matches the actual .npz
            # naming conventions (handling appended 'A', uppercase, etc.)
            #
            # If this ID includes extra prefixes like "gentle_cleaned_",
            # you'll need to parse that out or handle it similarly.

            # Example approach: if your "gentle_cleaned_" prefix is not
            # in the actual .npz name, strip it first:
            cleaned_id = recording_id
            # if cleaned_id.startswith("gentle_cleaned_"):
            #     cleaned_id = cleaned_id.replace("gentle_cleaned_", "")
            # elif cleaned_id.startswith("aggressive_cleaned_"):
            #     cleaned_id = cleaned_id.replace("aggressive_cleaned_", "")

            # Now attempt to find the real base .npz name:
            try:
                standardized_base = find_file_location_by_recording_id(npz_dir, cleaned_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # If we can't find it, you might skip or store some placeholder.
                # For now, let's skip.
                continue

            row = {
                "scenario": scenario,
                "filename": standardized_base,  # e.g. "XYZ12" or "XYZ12A"
                "label": lbl,
                "patient_id": pid
            }
            master_rows.append(row)

    master_df = pd.DataFrame(master_rows)

    # ------------------------------------------------------------------------
    # 5) (Optional) Add columns from meta_df (like AgeAtEEG1y, Interval, LeadtimeD)
    #    if they exist for each 'patient_id'
    # ------------------------------------------------------------------------
    meta_df = meta_df.reset_index(drop=False).rename(columns={"index": "pid_index"})
    meta_df.set_index("pid_index", inplace=True)

    def get_meta_value(pid, colname):
        if pid in meta_df.index and colname in meta_df.columns:
            return meta_df.at[pid, colname]
        return np.nan

    master_df["AgeAtEEG1y"] = master_df["patient_id"].apply(lambda pid: get_meta_value(pid, "AgeAtEEG1y"))
    master_df["LeadtimeD"]  = master_df["patient_id"].apply(lambda pid: get_meta_value(pid, "LeadtimeD"))

    # ------------------------------------------------------------------------
    # 6) Save the final CSV
    # ------------------------------------------------------------------------
    out_csv_path = os.path.join(os.path.dirname(data_path), "final_open_source_metadata.csv")
    master_df.to_csv(out_csv_path, index=False)
    print(f"Saved final CSV to: {out_csv_path}")
    print(f"Shape = {master_df.shape}")
    print(master_df.head(10))

if __name__ == "__main__":
    main()
