#!/usr/bin/env python
"""
edf_folder_to_biot_npz.py

Convert every EDF in a folder to BIOT-ready *.npz, in parallel.
Can handle either 16-channel or 18-channel BIOT style, depending on --num_biot_channels.

Example usage:
  python edf_folder_to_biot_npz.py \
      --raw_data_path  /path/to/edf_files \
      --out_folder     /path/to/biot_npz \
      --num_biot_channels 16 \
      --n_jobs 8
"""

import os, warnings, argparse, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mne.io import read_raw_edf
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Channel names are not unique.*",
    category=RuntimeWarning
)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Omitted.*annotation.*outside data range')

# ------------------------------------------------------------------ #
# 1) 16-channel BIOT montage definition                              #
# ------------------------------------------------------------------ #
BIOT16_BIPOLAR_PAIRS = [
    ("EEG FP1-REF","EEG F7-REF"), ("EEG F7-REF","EEG T3-REF"),
    ("EEG T3-REF","EEG T5-REF"), ("EEG T5-REF","EEG O1-REF"),
    ("EEG FP2-REF","EEG F8-REF"), ("EEG F8-REF","EEG T4-REF"),
    ("EEG T4-REF","EEG T6-REF"), ("EEG T6-REF","EEG O2-REF"),
    ("EEG FP1-REF","EEG F3-REF"), ("EEG F3-REF","EEG C3-REF"),
    ("EEG C3-REF","EEG P3-REF"), ("EEG P3-REF","EEG O1-REF"),
    ("EEG FP2-REF","EEG F4-REF"), ("EEG F4-REF","EEG C4-REF"),
    ("EEG C4-REF","EEG P4-REF"), ("EEG P4-REF","EEG O2-REF"),
]
BIOT16_CHANNEL_NAMES = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
]

# ------------------------------------------------------------------ #
# 2) 18-channel BIOT montage definition  (example)                    #
#    If you want a standard for 18-ch, update these pairs.            #
#    The idea is that the user wants 2 extra bipolars.                #
# ------------------------------------------------------------------ #
BIOT18_BIPOLAR_PAIRS = [
    ("EEG FP1-REF","EEG F7-REF"), ("EEG F7-REF","EEG T3-REF"),
    ("EEG T3-REF","EEG T5-REF"), ("EEG T5-REF","EEG O1-REF"),
    ("EEG FP2-REF","EEG F8-REF"), ("EEG F8-REF","EEG T4-REF"),
    ("EEG T4-REF","EEG T6-REF"), ("EEG T6-REF","EEG O2-REF"),
    ("EEG FP1-REF","EEG F3-REF"), ("EEG F3-REF","EEG C3-REF"),
    ("EEG C3-REF","EEG P3-REF"), ("EEG P3-REF","EEG O1-REF"),
    ("EEG FP2-REF","EEG F4-REF"), ("EEG F4-REF","EEG C4-REF"),
    ("EEG C4-REF","EEG P4-REF"), ("EEG P4-REF","EEG O2-REF"),
    ("EEG FZ-REF", "EEG CZ-REF"), 
    ("EEG CZ-REF", "EEG PZ-REF"), 
]
BIOT18_CHANNEL_NAMES = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FZ-CZ","CZ-PZ"
]

# ------------------------------------------------------------------ #
# 3) Canonicalize channel names and build montage.                   #
# ------------------------------------------------------------------ #
def _canon(name: str) -> str:
    """Remove spaces, 'EEG', 'REF', etc. Return uppercase for easy matching."""
    n = name.upper()
    n = n.replace("EEG","").replace(" ","").replace("POL","")
    n = n.replace("-REF","").replace("REF","")
    return n

def build_biot_montage(raw_data, raw_names, bipolars):
    """
    raw_data: shape (n_raw_ch, n_samples)
    raw_names: list[str] channel names from MNE
    bipolars: list of (plus, minus) pairs (already canonicalized)
    """
    # map canonical -> index
    canon2idx = {_canon(n): i for i,n in enumerate(raw_names)}

    out = np.zeros((len(bipolars), raw_data.shape[1]), dtype=np.float32)
    for i, (p, m) in enumerate(bipolars):
        p_ = _canon(p)
        m_ = _canon(m)
        if p_ not in canon2idx or m_ not in canon2idx:
            raise RuntimeError(f"Missing electrodes {p} or {m} in raw data.")
        out[i] = raw_data[ canon2idx[p_] ] - raw_data[ canon2idx[m_] ]
    return out

# ------------------------------------------------------------------ #
# 4)   single-file worker                                            #
# ------------------------------------------------------------------ #
def process_one_file(args_tuple):
    edf_path, out_folder, num_ch = args_tuple

    FS = 200
    # read EDF
    try:
        raw = read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        warnings.warn(f"[Error] Could not read {edf_path}: {e}")
        return 0

    # resample
    if int(raw.info["sfreq"]) != FS:
        raw.resample(FS, verbose=False)

    # get data in microvolts
    data_uV = raw.get_data(units="uV")  # (n_channels, n_times)

    # build montage
    if num_ch == 16:
        bipolars = [
            (_canon(a), _canon(b)) for (a,b) in BIOT16_BIPOLAR_PAIRS
        ]
        try:
            data_biot = build_biot_montage(data_uV, raw.ch_names, bipolars)
        except RuntimeError as e:
            warnings.warn(f"[Skipping] {edf_path} => {e}")
            return 0
        chan_names = BIOT16_CHANNEL_NAMES
    else:
        # e.g. 18-ch
        bipolars = [
            (_canon(a), _canon(b)) for (a,b) in BIOT18_BIPOLAR_PAIRS
        ]
        try:
            data_biot = build_biot_montage(data_uV, raw.ch_names, bipolars)
        except RuntimeError as e:
            warnings.warn(f"[Skipping] {edf_path} => {e}")
            return 0
        chan_names = BIOT18_CHANNEL_NAMES

    # save to .npz
    base = os.path.splitext(os.path.basename(edf_path))[0].upper()
    out_fp = os.path.join(out_folder, f"{base}.npz")
    np.savez_compressed(
        out_fp,
        data=data_biot,
        channel=np.array(chan_names),
        fs=FS
    )
    return 1

# ------------------------------------------------------------------ #
# 5)  parallel helper function                                       #
# ------------------------------------------------------------------ #
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a)
                 for a in array[:front_num]]

    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a)
                        for a in tqdm(array[front_num:])]

    out = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        for _ in tqdm(as_completed(futures),
                      total=len(futures), unit="it", unit_scale=True, leave=True):
            pass

    for fut in futures:
        try:
            out.append(fut.result())
        except Exception as e:
            out.append(e)
    return front + out

# ------------------------------------------------------------------ #
# 6)  main driver                                                    #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser("Convert EDF to BIOT (16- or 18-ch).")
    parser.add_argument("--raw_data_path", required=True,
                        help="Folder with .edf files (not recursive).")
    parser.add_argument("--out_folder", required=True,
                        help="Where to save .npz.")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--front_num", type=int, default=3)
    parser.add_argument("--num_biot_channels", type=int, choices=[16,18], default=16,
                        help="Use 16-ch or 18-ch BIOT. Default=16")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    edfs = sorted([f for f in os.listdir(args.raw_data_path) if f.lower().endswith(".edf")])
    print(f"Found {len(edfs)} EDF files in {args.raw_data_path}. Using {args.num_biot_channels}-ch montage...")

    job_args = []
    for edf_name in edfs:
        full_edf_path = os.path.join(args.raw_data_path, edf_name)
        job_args.append((full_edf_path, args.out_folder, args.num_biot_channels))

    results = parallel_process(
        job_args, process_one_file,
        n_jobs=args.n_jobs,
        use_kwargs=False,
        front_num=args.front_num
    )

    done_count = sum(r==1 for r in results)
    print(f"[Done] Successfully wrote {done_count} npz from {len(edfs)} EDF files.")

if __name__=="__main__":
    main()
