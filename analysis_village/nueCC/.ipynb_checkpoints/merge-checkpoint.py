#!/usr/bin/env python3
"""
merge.py
----------------
Merge multiple *_skimmed.df files (produced by skim_nue_v2.py) into a single
consolidated HDF5 file ready for the analysis notebook.

What gets merged
----------------
  evt_0         particle-level df (4-level index, preselected FM+FV)
  truth_info_0  per-interaction truth+preselection metadata (3-level index)
  histpotdf_0   POT table (concatenated then summed)

Index remapping
---------------
Each input file shares __ntuple = 0,1,2,... across files.  The merger
offsets the __ntuple level by i * NTUPLE_STRIDE (default 10_000) so
indices are globally unique in the merged file.

Usage
-----
    python merge_skimmed.py [OPTIONS]

    --input-glob   glob for *_skimmed.df files   (default: CONFIG)
    --output-file  path for merged output file    (default: CONFIG)
    --ntuple-stride  __ntuple offset per file     (default: 10000)
    --dry-run      list files, do not write

Dependencies: numpy, pandas, tqdm
"""

import argparse
import glob as _glob
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = dict(
    input_glob    = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/skimmed_df/*_skimmed.df",
    output_file   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/skimmed_df/merged.df",
    ntuple_stride = 10_000,
    n_files       = None,   # None = all; set to int to cap
    dry_run       = False,
    pot_key       = "histpotdf_0",
)

EVT_KEY_OUT    = "evt_0"
TRUTH_INFO_KEY = "truth_info_0"


# =============================================================================
# Per-file loader with index remapping
# =============================================================================

def load_file(path, file_index, ntuple_stride):
    """
    Load evt_0 and truth_info_0 from one skimmed file.
    Offsets __ntuple by file_index * ntuple_stride to ensure global uniqueness.

    Returns (df_evt, df_truth, pot_val) or raises on failure.
    """
    offset = file_index * ntuple_stride

    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()

        # ── evt_0 ──────────────────────────────────────────────────────
        evt_keys = [k for k in keys if k.lstrip("/").startswith("evt")]
        if not evt_keys:
            raise KeyError(f"No 'evt' key in {path}. Keys: {keys}")
        df_evt = store[evt_keys[0]]

        # ── truth_info_0 ────────────────────────────────────────────────
        if "/" + TRUTH_INFO_KEY in keys:
            df_truth = store[TRUTH_INFO_KEY]
        else:
            # Graceful fallback: synthesize an empty truth_info with correct index
            print(f"  WARNING: {TRUTH_INFO_KEY} missing in {path} — skipping truth_info.")
            df_truth = pd.DataFrame()

        # ── POT ─────────────────────────────────────────────────────────
        pot_keys = [k for k in keys if "histpotdf" in k]
        if pot_keys:
            pot_df = store[pot_keys[0]]
            pot_col = "pot" if "pot" in pot_df.columns else pot_df.select_dtypes("number").columns[0]
            pot_val = float(pot_df[pot_col].sum())
        else:
            print(f"  WARNING: no histpotdf in {path} — POT set to 0.")
            pot_val = 0.0

    # ── Remap __ntuple (level 0) ─────────────────────────────────────────
    def _remap(df):
        new_lvl0 = df.index.get_level_values(0) + offset
        if df.index.nlevels == 1:
            return df.set_index(new_lvl0)
        new_idx = pd.MultiIndex.from_arrays(
            [new_lvl0] + [df.index.get_level_values(i) for i in range(1, df.index.nlevels)],
            names=df.index.names,
        )
        return df.set_index(new_idx)

    df_evt   = _remap(df_evt)
    if not df_truth.empty:
        df_truth = _remap(df_truth)

    return df_evt, df_truth, pot_val


# =============================================================================
# Aggregate POT table
# =============================================================================

def _load_pot_tables(paths, pot_key):
    """Load and concatenate POT tables from all input files."""
    frames = []
    for p in paths:
        try:
            with pd.HDFStore(p, mode="r") as store:
                cands = [k for k in store.keys() if "histpotdf" in k]
                if cands:
                    key = ("/" + pot_key) if ("/" + pot_key) in store.keys() else cands[0]
                    frames.append(store[key])
        except Exception as e:
            print(f"  WARNING: POT read failed for {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# Main merge logic
# =============================================================================

def merge(files, output_file, ntuple_stride, dry_run, pot_key):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    if dry_run:
        print("DRY RUN — files that would be merged:")
        for f in files:
            print(f"  {f}")
        return

    evt_frames   = []
    truth_frames = []
    total_pot    = 0.0

    print(f"\nMerging {len(files)} file(s) into:\n  {output_file}\n")

    for i, path in enumerate(tqdm(files, desc="Loading", unit="file")):
        t0 = time.time()
        try:
            df_evt, df_truth, pot_val = load_file(path, i, ntuple_stride)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}  — skipping.")
            continue

        evt_frames.append(df_evt)
        if not df_truth.empty:
            truth_frames.append(df_truth)
        total_pot += pot_val

        tqdm.write(
            f"  [{i+1:>3d}/{len(files)}]  "
            f"evt={df_evt.shape}  "
            f"truth={df_truth.shape if not df_truth.empty else 'n/a'}  "
            f"pot={pot_val:.3e}  ({time.time()-t0:.1f}s)"
        )

    if not evt_frames:
        print("ERROR: no files loaded successfully.")
        sys.exit(1)

    # ── Concatenate ───────────────────────────────────────────────────────
    print("\nConcatenating evt frames ...", end=" ", flush=True)
    t1      = time.time()
    evtdf   = pd.concat(evt_frames, axis=0)
    print(f"done ({time.time()-t1:.1f}s)  shape={evtdf.shape}")

    print("Concatenating truth_info frames ...", end=" ", flush=True)
    t2 = time.time()
    if truth_frames:
        truth_info = pd.concat(truth_frames, axis=0)
        print(f"done ({time.time()-t2:.1f}s)  shape={truth_info.shape}")
    else:
        truth_info = pd.DataFrame()
        print("skipped (no truth_info tables found).")

    # ── Summary from truth_info ───────────────────────────────────────────
    if not truth_info.empty:
        n_true_signal = int(truth_info["is_true_signal"].sum())
        n_inter_total = len(truth_info)
        n_after_fm    = int(truth_info["passed_fm"].sum())
        n_presel      = int(truth_info["passed_presel"].sum())
        n_sig_fm      = int(truth_info.loc[truth_info["passed_fm"],    "is_true_signal"].sum())
        n_sig_presel  = int(truth_info.loc[truth_info["passed_presel"],"is_true_signal"].sum())

        CAT_LABELS = {
            0:"nueCC FV (signal)", 1:"nueCC out FV",   2:"numuCC + pi0",
            3:"NC pi0",            4:"other numuCC",    5:"other NC",
            6:"cosmic",            7:"other neutrino",
        }
        print("\n  -- Merged truth_info summary (pre-cut denominators) --------")
        print(f"  n_inter_total       : {n_inter_total:>15,}")
        print(f"  n_after_fm          : {n_after_fm:>15,}  ({n_after_fm/n_inter_total*100:.1f}%)")
        print(f"  n_presel (FM+FV)    : {n_presel:>15,}  ({n_presel/n_inter_total*100:.1f}%)")
        print(f"  n_true_signal (all) : {n_true_signal:>15,}  <- denominator")
        print(f"  n_sig after FM      : {n_sig_fm:>15,}  ({n_sig_fm/max(n_true_signal,1)*100:.1f}%)")
        print(f"  n_sig after FM+FV   : {n_sig_presel:>15,}  ({n_sig_presel/max(n_true_signal,1)*100:.1f}%)")
        print("  Truth categories:")
        cat_counts = truth_info["truth_cat"].value_counts().sort_index()
        for c, n in cat_counts.items():
            print(f"    cat {c}  {CAT_LABELS.get(c,'?'):<22s}: {n:>9,}")

    # ── Write ─────────────────────────────────────────────────────────────
    print(f"\nWriting merged file -> {output_file} ...", end=" ", flush=True)
    t3 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evtdf.to_hdf(output_file, key=EVT_KEY_OUT, mode="w",
                     complevel=1, complib="blosc")
        if not truth_info.empty:
            truth_info.to_hdf(output_file, key=TRUTH_INFO_KEY, mode="a",
                               complevel=1, complib="blosc")

        # Merged POT table
        pot_tables = _load_pot_tables(files, pot_key)
        if not pot_tables.empty:
            with pd.HDFStore(output_file, mode="a") as dst:
                dst[pot_key] = pot_tables
    print(f"done ({time.time()-t3:.1f}s)")

    size_mb = os.path.getsize(output_file) / 1024**2
    print(f"\n{'='*60}")
    print(f"  Output : {output_file}")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"  Total POT : {total_pot:.4e}")
    print(f"  HDF5 keys : {EVT_KEY_OUT}, {TRUTH_INFO_KEY}, {pot_key}")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge *_skimmed.df files into a single analysis-ready .df"
    )
    parser.add_argument("--input-glob",    default=CONFIG["input_glob"])
    parser.add_argument("--output-file",   default=CONFIG["output_file"])
    parser.add_argument("--ntuple-stride", type=int, default=CONFIG["ntuple_stride"])
    parser.add_argument("--n-files",       type=int, default=CONFIG["n_files"],
                        help="Merge only the first N files (default: all)")
    parser.add_argument("--dry-run",       action="store_true", default=CONFIG["dry_run"])
    args = parser.parse_args()

    files = sorted(_glob.glob(args.input_glob))
    if not files:
        print(f"ERROR: no files match {args.input_glob!r}")
        sys.exit(1)
    if args.n_files:
        files = files[: args.n_files]

    print(f"Found {len(files)} skimmed file(s).")
    print(f"Output     : {args.output_file}")
    print(f"__ntuple stride : {args.ntuple_stride:,}")

    merge(
        files         = files,
        output_file   = args.output_file,
        ntuple_stride = args.ntuple_stride,
        dry_run       = args.dry_run,
        pot_key       = CONFIG["pot_key"],
    )


if __name__ == "__main__":
    main()