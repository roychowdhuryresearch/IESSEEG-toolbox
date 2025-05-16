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
    message=".*Channel names are not unique.*",  # a regex pattern
    category=RuntimeWarning
)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Omitted.*annotation.*outside data range')

# ------------------------------------------------------------------------
# 1) Parallel processing with progress bar
# ------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]

    # If we set n_jobs=1, just run a list comprehension
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]

    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass

    out = []
    # Get the results from the futures
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

# ------------------------------------------------------------------------
# 2) Defaults for allowed channels and bipolar pairs
# ------------------------------------------------------------------------
DEFAULT_ALLOWED_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ', 'A1', 'A2'
]
DEFAULT_ALLOWED_CHANNEL_NAMES = [ch_name.upper() for ch_name in DEFAULT_ALLOWED_CHANNEL_NAMES]

# Example from your snippet; adjust to your exact bipolar setup
DEFAULT_BIPOLAR_CHANNELS = [
    [1, 11], [11, 13], [13, 15], [15, 9],
    [0, 10], [10, 12], [12, 14], [14, 8],
    [20, 13], [13, 5], [5, 17], [17, 4],
    [4, 12], [12, 19], [1, 3], [3, 5],
    [5, 7], [7, 9], [0, 2], [2, 4],
    [4, 6], [6, 8]
]

REMOVE_PREFIX_SUFFIX = ["-Ref","EEG ","POL"," "]

# ------------------------------------------------------------------------
# 3) Argparse
# ------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="EDF to NPZ Preprocessing Script")

    parser.add_argument("--raw_data_path", type=str, required=True,
                        help="Directory containing .edf files.")
    parser.add_argument("--out_folder", type=str, required=True,
                        help="Directory for saving .npz outputs.")

    # Filter cutoffs
    parser.add_argument("--low_freq", type=float, default=1.0,
                        help="High-pass cutoff frequency (default=1.0 Hz).")
    parser.add_argument("--high_freq", type=float, default=50.0,
                        help="Low-pass cutoff frequency (default=50.0 Hz).")

    # Notch params
    parser.add_argument("--apply_notch", action="store_true",
                        help="Whether to apply a notch filter.")
    parser.add_argument("--notch_freq", type=float, default=50.0,
                        help="Notch frequency (e.g., 50 or 60). Only used if --apply_notch is provided.")

    # Referencing
    parser.add_argument("--reference_scheme", type=str, choices=["monopolar","bipolar"],
                        default="monopolar", help="Reference scheme: 'monopolar' or 'bipolar'. Default=monopolar")
    parser.add_argument("--monopolar_reference", type=str, default="average",
                        help="If reference_scheme=monopolar, choose 'average', 'A1', etc. Default='average'.")

    # Parallel
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Number of parallel workers (default=8).")

    # Channels
    parser.add_argument("--allowed_channel_names", nargs="+", default=None,
                        help="List of channel names to keep. If not provided, uses a default set of scalp channels.")
    parser.add_argument("--bipolar_pairs", nargs="+", type=int, default=None,
                        help="Flattened list of bipolar index pairs: e.g. --bipolar_pairs 1 11 11 13 => [(1,11),(11,13)].")

    return parser.parse_args()

# ------------------------------------------------------------------------
# 4) Filtering & referencing helpers
# ------------------------------------------------------------------------
def remove_prefix_suffix_from_ch_name(ch_name):
    for fix in REMOVE_PREFIX_SUFFIX:
        ch_name = ch_name.replace(fix, "")
    return ch_name

def apply_filters(raw, l_freq, h_freq, apply_notch, notch_freq):
    """
    Band-pass raw between l_freq and h_freq, optionally apply a separate notch at notch_freq.
    """
    raw.load_data()
    # Band-pass
    raw.filter(
        l_freq=l_freq, 
        h_freq=h_freq, 
        picks='eeg', 
        method='fir',
        fir_design='firwin',
        verbose=False
    )
    # Optional notch
    if apply_notch:
        raw.notch_filter(freqs=[notch_freq], picks='eeg', verbose=False)
    return raw

def apply_monopolar_reference(raw, ref):
    """
    Use MNE's built-in referencing:
      - 'average' => common average ref
      - 'A1' => single-channel reference (if channel A1 is present)
    """
    raw.set_eeg_reference(ref_channels=ref, projection=False, verbose=False)
    return raw

def apply_bipolar_reference(data_array, channel_names, bipolar_pairs):
    """
    data_array shape: (n_channels, n_times)
    bipolar_pairs: list of [ch1_idx, ch2_idx]
    returns (bipolar_data, new_channel_names)
    """
    n_times = data_array.shape[1]
    out_data = np.zeros((len(bipolar_pairs), n_times))
    out_names = []
    for i, (idx1, idx2) in enumerate(bipolar_pairs):
        out_data[i, :] = data_array[idx1, :] - data_array[idx2, :]
        ch1 = channel_names[idx1]
        ch2 = channel_names[idx2]
        out_names.append(f"{ch1}-{ch2}")
    return out_data, out_names

# ------------------------------------------------------------------------
# 5) The function for a single file
# ------------------------------------------------------------------------
def process_one_file(edf_filename):
    """
    This function will be called by parallel_process().
    We'll rely on global 'ARGS', 'ALLOWED_CHANNELS', 'BIPOLAR_PAIRS' or
    store them in a closure for convenience.
    """
    # 1) Build full path
    file_path = os.path.join(ARGS.raw_data_path, edf_filename)
    if not edf_filename.lower().endswith(".edf"):
        return 0  # skip non-EDF

    # 2) Load data
    raw = read_raw_edf(file_path, preload=True, verbose=False)

    # 3) Filter (band-pass + optional notch)
    raw = apply_filters(raw,
                        l_freq=ARGS.low_freq,
                        h_freq=ARGS.high_freq,
                        apply_notch=ARGS.apply_notch,
                        notch_freq=ARGS.notch_freq)

    # 4) Downsample to 200 Hz if needed
    if raw.info["sfreq"] != 200:
        raw.resample(200, npad="auto", verbose=False)

    # 5) Monopolar or bipolar referencing
    if ARGS.reference_scheme == "monopolar":
        pick_list = []
        for ch_name in raw.ch_names:
            cleaned = remove_prefix_suffix_from_ch_name(ch_name).upper()
            if cleaned in ALLOWED_CHANNELS:
                pick_list.append(ch_name)
        # 2) Pick those channels in MNE
        raw.pick(pick_list)

        raw = apply_monopolar_reference(raw, ARGS.monopolar_reference)

        data = raw.get_data(units="uV")  # shape: (n_ch, n_times)
        ch_names = raw.ch_names

        # Keep only allowed channels
        ch_data_pairs = []
        for i, ch_name in enumerate(ch_names):
            clean_name = remove_prefix_suffix_from_ch_name(ch_name).upper()
            if clean_name in ALLOWED_CHANNELS:
                ch_data_pairs.append((clean_name, data[i, :]))

        # Sort by known channel order
        ch_data_pairs.sort(key=lambda x: ALLOWED_CHANNELS.index(x[0]))
        final_channels = [p[0] for p in ch_data_pairs]
        final_data = np.vstack([p[1] for p in ch_data_pairs])

    else:  # ARGS.reference_scheme == "bipolar"
        data = raw.get_data(units="uV")
        ch_names = raw.ch_names

        # Pick + sort allowed channels
        ch_data_pairs = []
        for i, ch_name in enumerate(ch_names):
            clean_name = remove_prefix_suffix_from_ch_name(ch_name).upper()
            if clean_name in ALLOWED_CHANNELS:
                ch_data_pairs.append((clean_name, data[i, :]))

        ch_data_pairs.sort(key=lambda x: ALLOWED_CHANNELS.index(x[0]))
        sorted_names = [p[0] for p in ch_data_pairs]
        sorted_data = np.vstack([p[1] for p in ch_data_pairs])

        # Apply bipolar referencing
        final_data, final_channels = apply_bipolar_reference(
            sorted_data, 
            sorted_names, 
            BIPOLAR_PAIRS
        )

    # 6) Save
    out_base = os.path.splitext(edf_filename)[0].upper()
    out_file = os.path.join(ARGS.out_folder, f"{out_base}.npz")
    np.savez_compressed(out_file,
                        data=final_data,
                        channel=np.array(final_channels))

    return 1  # success indicator

# ------------------------------------------------------------------------
# 6) MAIN
# ------------------------------------------------------------------------
def main():
    global ARGS, ALLOWED_CHANNELS, BIPOLAR_PAIRS

    # 1) Parse command line
    ARGS = get_args()
    os.makedirs(ARGS.out_folder, exist_ok=True)

    # 2) Set up allowed channels
    if not ARGS.allowed_channel_names:
        ALLOWED_CHANNELS = DEFAULT_ALLOWED_CHANNEL_NAMES
    else:
        ALLOWED_CHANNELS = ARGS.allowed_channel_names

    # 3) Set up bipolar pairs
    if ARGS.bipolar_pairs and len(ARGS.bipolar_pairs) % 2 == 0:
        # e.g.: --bipolar_pairs 1 2 3 4 => pairs = [[1,2],[3,4]]
        BIPOLAR_PAIRS = [
            ARGS.bipolar_pairs[i:i+2] for i in range(0, len(ARGS.bipolar_pairs), 2)
        ]
    else:
        BIPOLAR_PAIRS = DEFAULT_BIPOLAR_CHANNELS

    # 4) Gather EDF files
    all_files = [f for f in os.listdir(ARGS.raw_data_path) if f.lower().endswith(".edf")]
    print(f"Found {len(all_files)} EDF files in {ARGS.raw_data_path}")
    print(f"Output will be saved in {ARGS.out_folder}")

    # 5) Run in parallel (using the unmodified parallel_process)
    results = parallel_process(
        array=all_files,
        function=process_one_file,
        n_jobs=ARGS.n_jobs,
        use_kwargs=False,
        front_num=3
    )

    # Summarize
    success_count = sum(r == 1 for r in results)
    print(f"Completed {success_count}/{len(all_files)} files successfully.")

if __name__ == "__main__":
    main()
