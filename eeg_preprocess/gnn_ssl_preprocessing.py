#!/usr/bin/env python
"""
Convert EDF to 19-channel referential *.npz

Example (monopolar):
  python edf_folder_to_biot_npz.py \
      --raw_data_path /path/to/edf \
      --out_folder    /path/to/mono_npz \
      --out_format    mono_npz \
      --n_jobs 8
"""
import os, argparse, warnings, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mne.io import read_raw_edf
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Channel names are not unique.*",
    category=RuntimeWarning
)


# ------------------------------------------------------------------ #
NEUROGNN_CH = [
    'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'
]

def _canon(name: str) -> str:
    n = name.upper().replace(" ", "")
    for tag in ("EEG", "-REF", "REF", "POL"):
        n = n.replace(tag, "")
    return n

def pick_channels(raw):
    idx = {_canon(n): i for i, n in enumerate(raw.ch_names)}
    miss = [c for c in NEUROGNN_CH if _canon(c) not in idx]
    if miss:
        raise RuntimeError(f"Missing channels: {miss}")
    src = raw.get_data(units='uV')
    out = np.empty((19, src.shape[1]), np.float32)
    for i, ch in enumerate(NEUROGNN_CH):
        out[i] = src[idx[_canon(ch)]]
    return out            # shape (19, T)

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

def process_one_file(args_tuple):
    edf_path, out_folder = args_tuple
    FS = 200
    try:
        raw = read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        warnings.warn(f"[Error] {edf_path}: {e}")
        return 0

    if int(raw.info["sfreq"]) != FS:
        raw.resample(FS, verbose=False)

    base = os.path.splitext(os.path.basename(edf_path))[0].upper()

    try:
        sig = pick_channels(raw)
    except RuntimeError as e:
        warnings.warn(f"[Skip] {base}: {e}")
        return 0
    np.savez_compressed(
        os.path.join(out_folder, f"{base}.npz"),
        data=sig, channel=np.array(NEUROGNN_CH), fs=FS
    )
    return 1

# ------------------------------------------------------------------ #
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--raw_data_path", required=True)
    pa.add_argument("--out_folder", required=True)
    pa.add_argument("--n_jobs", type=int, default=4)
    pa.add_argument("--front_num", type=int, default=3)
    args = pa.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    edfs = sorted(f for f in os.listdir(args.raw_data_path) if f.lower().endswith(".edf"))
    print(f"Found {len(edfs)} EDFs")

    jobs = [(os.path.join(args.raw_data_path, f), args.out_folder) for f in edfs]

    res = parallel_process(jobs, process_one_file,
                           n_jobs=args.n_jobs, front_num=args.front_num)
    print(f"✓ Wrote {sum(r==1 for r in res)} files to {args.out_folder}")

if __name__ == "__main__":
    main()
