#!/usr/bin/env python
import os
import argparse
import numpy as np
import mne
from mne.io import read_raw_edf
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Channel names are not unique.*",
    category=RuntimeWarning
)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Omitted.*annotation.*outside data range')

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

###############################################################################
# TUEG 19-channel subset (exact order) for CBraMod
###############################################################################
TUEG_19_ORDER = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ'
]
REMOVE_PREFIX_SUFFIX = ["-Ref", "EEG ", "POL", " "]

###############################################################################
# 1) Parallel processing helper
###############################################################################
def parallel_process(file_list, function, n_jobs=16, front_num=3):
    """
    A parallel version of map with a progress bar.

    Args:
        file_list (list): List of items (e.g. filenames) to process.
        function (callable): A function that processes a single item from file_list.
        n_jobs (int): Number of parallel workers.
        front_num (int): Number of iterations to run serially to catch bugs early.

    Returns:
        List of results from 'function'.
    """
    # Run a few in serial first to catch potential errors
    front_results = []
    if front_num > 0:
        front_results = [function(file_list[i]) for i in range(min(front_num, len(file_list)))]

    # If only one job, do it in a straightforward loop
    if n_jobs == 1:
        return front_results + [function(f) for f in tqdm(file_list[front_num:])]

    # Otherwise, use concurrent futures
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(function, f) for f in file_list[front_num:]]
        # progress bar over the futures
        for _ in tqdm(as_completed(futures), total=len(futures), unit='it', unit_scale=True, leave=True):
            pass

        # collect results
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(e)

    return front_results + results

###############################################################################
# 2) Argparse
###############################################################################
def get_args():
    parser = argparse.ArgumentParser(description="TUEG 19-ch Average-Ref Preprocessing for CBraMod")

    parser.add_argument("--raw_data_path", type=str, required=True,
                        help="Directory containing .edf files.")
    parser.add_argument("--out_folder", type=str, required=True,
                        help="Directory for saving .npz files.")

    parser.add_argument("--low_freq", type=float, default=0.3,
                        help="High-pass cutoff (default=0.3 Hz).")
    parser.add_argument("--high_freq", type=float, default=75.0,
                        help="Low-pass cutoff (default=75.0 Hz).")
    parser.add_argument("--apply_notch", action="store_true",
                        help="Apply 60 Hz notch filter if set.")
    parser.add_argument("--notch_freq", type=float, default=60.0,
                        help="Notch frequency (default=60 Hz).")

    parser.add_argument("--trim_data", action="store_true", default=False,
                        help="Trim first/last N seconds if set.")
    parser.add_argument("--trim_sec", type=int, default=60,
                        help="Trim first/last N seconds (default=60).")

    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Number of parallel workers (default=8).")

    return parser.parse_args()

###############################################################################
# 3) Helper for channel renaming
###############################################################################
def clean_ch_name(ch_name):
    """
    Remove known prefixes/suffixes and convert to uppercase.
    e.g. "EEG Fp1-Ref" -> "FP1"
    """
    upper_ch = ch_name.upper()
    for fix in REMOVE_PREFIX_SUFFIX:
        upper_ch = upper_ch.replace(fix.upper(), "")
    return upper_ch

# Pre-build a lookup dict so we know how to map cleaned uppercase names
# back to the correct TUEG label (with original case).
TUEG_19_MAP = {ch.upper(): ch for ch in TUEG_19_ORDER}

###############################################################################
# 4) The main function that processes a single .edf
###############################################################################
def process_one_file(
    fname,
    raw_data_path,
    out_folder,
    low_freq,
    high_freq,
    apply_notch,
    notch_freq,
    trim_data,
    trim_sec
):
    """Process a single .edf file to a .npz with TUEG 19-ch data."""

    # Skip non-EDF
    if not fname.lower().endswith(".edf"):
        return 0

    filepath = os.path.join(raw_data_path, fname)
    try:
        raw = read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"Skipping {fname} due to error: {e}")
        return 0

    # 1) Filter (band-pass) and optional notch
    raw.load_data(verbose=False)
    raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
    if apply_notch:
        raw.notch_filter(freqs=[notch_freq], verbose=False)

    # 2) Resample to 200 Hz if needed
    if raw.info["sfreq"] != 200:
        raw.resample(200, verbose=False)

    # 3) Rename channels to TUEG labels
    rename_dict = {
        ch: TUEG_19_MAP[clean_ch_name(ch)]
        for ch in raw.ch_names
        if clean_ch_name(ch) in TUEG_19_MAP
    }
    raw.rename_channels(rename_dict)

    # 4) Pick exactly the TUEG_19_ORDER channels (drop the rest)
    raw.pick(TUEG_19_ORDER)

    # Ensure we truly have all 19
    picked_chs = set(raw.ch_names)
    required_chs = set(TUEG_19_ORDER)
    missing = list(required_chs - picked_chs)
    if missing:
        print(f"Skipping {fname}: missing channels {missing}")
        return 0

    # Reorder them to the canonical order
    raw.reorder_channels(TUEG_19_ORDER)

    # 5) Average referencing (only the 19 channels remain)
    raw.set_eeg_reference(ref_channels='average', verbose=False)

    # 6) Convert data to microvolts
    final_data = raw.get_data(units='uV')  # shape = (19, n_times)

    # 7) Trim first/last N seconds if requested
    if trim_data:
        fs = 200
        trim_pts = trim_sec * fs
        if final_data.shape[1] <= 2 * trim_pts:
            print(f"Skipping {fname}: not enough data after trim.")
            return 0
        final_data = final_data[:, trim_pts:-trim_pts]

    # 8) Save
    out_base = os.path.splitext(fname)[0].upper()
    out_file = os.path.join(out_folder, f"{out_base}.npz")
    referenced_channel_names = [f"{ch}-REF" for ch in TUEG_19_ORDER]
    np.savez_compressed(
        out_file,
        data=final_data,
        channel=np.array(referenced_channel_names)
    )

    return 1

###############################################################################
# 5) Main
###############################################################################
def main():
    # Parse command-line arguments
    ARGS = get_args()
    os.makedirs(ARGS.out_folder, exist_ok=True)

    # Gather all .edf files
    files = sorted([
        f for f in os.listdir(ARGS.raw_data_path) 
        if f.lower().endswith(".edf")
    ])
    print(f"Found {len(files)} .edf files in {ARGS.raw_data_path}")

    # Build a partial to freeze the function parameters, except 'fname'
    process_fn = partial(
        process_one_file,
        raw_data_path=ARGS.raw_data_path,
        out_folder=ARGS.out_folder,
        low_freq=ARGS.low_freq,
        high_freq=ARGS.high_freq,
        apply_notch=ARGS.apply_notch,
        notch_freq=ARGS.notch_freq,
        trim_data=ARGS.trim_data,
        trim_sec=ARGS.trim_sec
    )

    # Run in parallel
    results = parallel_process(files, process_fn, n_jobs=ARGS.n_jobs, front_num=3)
    total_saved = sum(r for r in results if isinstance(r, int))
    print(f"Done! Total .npz files saved: {total_saved}.")

if __name__ == "__main__":
    main()
