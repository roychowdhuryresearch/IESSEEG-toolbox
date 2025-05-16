#!/usr/bin/env python
"""
make_final_csv.py

Merges data from:
  - MetaEEG.csv
  - MetaDataForMingjian.csv
  - long_short_mappings.csv

Produces a single CSV with columns:
  [short_recording_id, long_recording_id, case_control_label,
   pre_post_treatment_label, sleep_awake_label,
   patient_id, AgeAtEEG1y, LeadtimeD]

Example usage:
  python make_final_csv.py \
    --meta_eeg "MetaEEG.csv" \
    --meta_ming "MetaDataForMingjian.csv" \
    --long_map "long_short_mappings.csv" \
    --out_csv "final_merged.csv"
"""

import os
import argparse
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

def determine_case_control(row):
    """
    Given a row (namedtuple) from 'long_short_mappings.csv' with columns like:
      CASE, CONTROL, ...
    Return "CASE" if row.CASE == 1, "CONTROL" if row.CONTROL == 1,
    or "UNKNOWN" if neither. Adjust logic as needed.
    """
    # Use getattr(...) to get column values safely with a default of 0
    case_val = getattr(row, "CASE", 0)
    control_val = getattr(row, "CONTROL", 0)

    if case_val == 1:
        return "CASE"
    elif control_val == 1:
        return "CONTROL"
    return "UNKNOWN"

def determine_pre_post(short_id):
    """
    If the short recording ID is e.g. CodePreWake1a => label "PRE".
    If it's CodePostSleep2a => label "POST".
    Adjust logic if you have separate columns "Code2_PRE" or "Code2_POST" to check.
    """
    s_id = str(short_id).lower()
    if "pre" in s_id:
        return "PRE"
    elif "post" in s_id:
        return "POST"
    return "UNKNOWN"

def determine_sleep_awake(short_id):
    """
    If "Wake" in short_id => "AWAKE", if "Sleep" => "SLEEP".
    Adjust as needed (some might say "wake" vs. "awake" in the column).
    """
    s_id = str(short_id).lower()
    if "wake" in s_id:
        return "AWAKE"
    elif "sleep" in s_id:
        return "SLEEP"
    return "UNKNOWN"

def determine_treatment_response(numeric_label):
    """
    Given a numeric label, determine if it's "Responder" or "Non-responder".
    Adjust logic as needed.
    """
    if numeric_label == 1:
        return "Responder"
    elif numeric_label == 0:
        return "Non-responder"
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_eeg", type=str, required=True,
                        help="Path to MetaEEG.csv")
    parser.add_argument("--meta_ming", type=str, required=True,
                        help="Path to MetaDataForMingjian.csv")
    parser.add_argument("--long_map", type=str, required=True,
                        help="Path to long_short_mappings.csv")
    parser.add_argument("--out_csv", type=str, default="final_merged.csv",
                        help="Output CSV file with final merged data.")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Directory where your .npz files for 30min samples are stored.")
    args = parser.parse_args()

    # 1) Load the CSVs
    meta_eeg_df = pd.read_csv(args.meta_eeg)
    meta_ming_df = pd.read_csv(args.meta_ming)
    long_map_df  = pd.read_csv(args.long_map)

    rename_map = {
        # key = original col in meta_ming_df, value = name you want in the result
        "CodePreWake1a":   "CodePreWake1",
        "CodePreSleep1a":  "CodePreSleep1",
        "CodePreWake2a":   "CodePreWake2",
        "CodePreSleep2a":  "CodePreSleep2",
        "CodePostWake1a":  "CodePostWake1",
        "CodePostSleep1a": "CodePostSleep1",
        "CodePostWake2a":  "CodePostWake2",
        "CodePostSleep2a": "CodePostSleep2",
    }

    keep_extra = [
        "PatientID",
        "ResponderAtEEG01",
        "RESPONDER"
    ]

    meta_ming_aligned = (
        meta_ming_df
        .rename(columns=rename_map)
        .loc[:, keep_extra + list(rename_map.values())]
    )

    meta_eeg_df = meta_eeg_df.merge(
        meta_ming_aligned,
        on=["CodePreWake1", "CodePreSleep1", "CodePreWake2", "CodePreSleep2", "CodePostWake1", "CodePostSleep1", "CodePostWake2", "CodePostSleep2"],
        how="left",
        validate="m:1"
    )

    # 2) Build a dictionary: patient_id => (AgeAtEEG1y, LeadtimeD)
    #    If 'MetaEEG.csv' has these columns. Adapt if not.
    meta_eeg_df = meta_eeg_df.rename(columns={"CASE40":"CASE40_OLD"})  # avoid conflicts
    meta_eeg_map = {}
    for row in meta_eeg_df.to_dict(orient="records"):
        Interval, AgeAtEEG1y, AOOmo, LeadtimeD, LeadtimeUKISS, immediate_responder, meaningful_responder = row["Interval"], row["AgeAtEEG1y"], row["AOOmo"], row["LeadtimeD"], row["LeadtimeUKISS"], row["ResponderAtEEG01"], row["RESPONDER"]
        for col in ["CodePreWake1", "CodePreSleep1", "CodePreWake2", "CodePreSleep2", "CodePostWake1", "CodePostSleep1", "CodePostWake2", "CodePostSleep2"]:
            short_id = row[col]
            if short_id:
                meta_eeg_map[short_id] = (Interval, AgeAtEEG1y, AOOmo, LeadtimeD, LeadtimeUKISS, immediate_responder, meaningful_responder)
    final_rows = []

    # 3) For each row in long_short_mappings.csv => produce multiple output rows
    for pid, row in enumerate(long_map_df.itertuples()):
        official_patient_id = getattr(row, "PatientID", None)
        case_control_label = determine_case_control(row)

        pre_long_id  = getattr(row, "Code2_PRE", None)
        post_long_id = getattr(row, "Code2_POST", None)

        short_id_cols = [
            "CodePreWake1a", "CodePreSleep1a", "CodePreWake2a", "CodePreSleep2a",
            "CodePostWake1a","CodePostSleep1a","CodePostWake2a","CodePostSleep2a"
        ]

        for colname in short_id_cols:
            short_id = getattr(row, colname, None)
            if not short_id or pd.isnull(short_id):
                continue

            try:
                standardized_base = find_file_location_by_recording_id(args.npz_dir, short_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # If we can't find it, you might skip or store some placeholder.
                # For now, let's skip.
                continue

            sleep_awake_label = determine_sleep_awake(colname)
            # Optionally you can do `pre_post_label = determine_pre_post(short_id)`
            # but here we do simpler => "PRE" if colname has "Pre", else "POST"
            pre_post_label = "PRE" if "Pre" in colname else "POST"
            long_recording_id = pre_long_id if "Pre" in colname else post_long_id

            if short_id in meta_eeg_map:
                Interval, AgeAtEEG1y, AOOmo, LeadtimeD, LeadtimeUKISS, immediate_responder, meaningful_responder = meta_eeg_map[short_id]
                immediate_responder = determine_treatment_response(immediate_responder)
                meaningful_responder = determine_treatment_response(meaningful_responder)
                # if LeadtimeD is not a number or cannot convert to int or float, set to NaN
                try:
                    LeadtimeD = int(LeadtimeD)
                except (ValueError, TypeError):
                    LeadtimeD = np.nan
            else:
                Interval, AgeAtEEG1y, AOOmo, LeadtimeD, LeadtimeUKISS, immediate_responder, meaningful_responder = (np.nan, np.nan, np.nan, np.nan, np.nan)

            final_rows.append({
                "short_recording_id": standardized_base,
                "long_recording_id": long_recording_id,
                "case_control_label": case_control_label,
                "pre_post_treatment_label": pre_post_label,
                "sleep_awake_label": sleep_awake_label,
                "patient_id": pid,
                "Interval": Interval,
                "AgeAtEEG1y": AgeAtEEG1y,
                "AOOmo": AOOmo,
                "LeadtimeD": LeadtimeD,
                "LeadtimeUKISS": LeadtimeUKISS,
                "immediate_responder": immediate_responder,
                "meaningful_responder": meaningful_responder
            })

    # 4) Build final DataFrame
    final_df = pd.DataFrame(final_rows)

    # 5) Save to CSV
    final_df.to_csv(args.out_csv, index=False)
    print(f"Created final CSV => {args.out_csv}")
    print(final_df.head(10))
    print(f"Shape => {final_df.shape}")

if __name__ == "__main__":
    main()
