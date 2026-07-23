#!/usr/bin/env python3
"""
merge_detvar_dfs.py  (Step 1b)
-------------------------------
Merge the per-job .df outputs from run_detvar_dfmaker.sh into a single
.df per variation — same logic as your existing merge.py but pointed at
the det-var output directories.

Usage:
    python scripts/merge_detvar_dfs.py
    python scripts/merge_detvar_dfs.py --var wiremodxtheta
    python scripts/merge_detvar_dfs.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
NUECC_DIR = SCRIPT_DIR.parent

DETVAR_DF_BASE = "/exp/sbnd/data/users/castalyf/nue_sel/detsys/gen1/detvar_dfs"
NTUPLE_STRIDE = 10_000  # same as your merge.py


def merge_df_files(
    df_files: list[str],
    output_file: str,
    *,
    ntuple_stride: int = NTUPLE_STRIDE,
    dry_run: bool = False,
) -> dict:
    """
    Merge multiple .df (HDF5) files into one, offsetting __ntuple for uniqueness.

    This mirrors your existing merge.py logic.
    """
    stats = {
        "n_files": len(df_files),
        "output": output_file,
    }

    if dry_run:
        print(f"  Would merge {len(df_files)} files -> {output_file}")
        return stats

    all_evtdf = []
    all_truth = []
    all_pot = []
    all_hdr = []

    for i, fpath in enumerate(tqdm(df_files, desc="  Merging")):
        try:
            store = pd.HDFStore(fpath, mode="r")
            keys = store.keys()

            # Load each table, offset __ntuple
            for key in keys:
                df = store[key]

                # Offset __ntuple index level for uniqueness
                if isinstance(df.index, pd.MultiIndex) and "__ntuple" in df.index.names:
                    ntuple_level = df.index.names.index("__ntuple")
                    new_levels = list(df.index.levels)
                    new_levels[ntuple_level] = new_levels[ntuple_level] + i * ntuple_stride
                    df.index = df.index.set_levels(new_levels)

                clean_key = key.strip("/")
                if "evt" in clean_key and "truth" not in clean_key:
                    all_evtdf.append(df)
                elif "truth" in clean_key:
                    all_truth.append(df)
                elif "pot" in clean_key:
                    all_pot.append(df)
                elif "hdr" in clean_key:
                    all_hdr.append(df)

            store.close()
        except Exception as e:
            print(f"  WARNING: Error reading {fpath}: {e}")

    # Concatenate and save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(output_file, mode="w") as store:
        if all_evtdf:
            merged_evt = pd.concat(all_evtdf, axis=0)
            store["evt_0"] = merged_evt
            stats["n_events"] = len(merged_evt)
            print(f"  evt_0: {len(merged_evt)} rows")
            del merged_evt

        if all_truth:
            merged_truth = pd.concat(all_truth, axis=0)
            store["truth_info_0"] = merged_truth
            stats["n_truth"] = len(merged_truth)
            print(f"  truth_info_0: {len(merged_truth)} rows")
            del merged_truth

        if all_pot:
            merged_pot = pd.concat(all_pot, axis=0)
            store["pot_0"] = merged_pot
            total_pot = merged_pot.sum().sum() if not merged_pot.empty else 0
            stats["total_pot"] = float(total_pot)
            print(f"  pot_0: total POT = {total_pot:.4e}")
            del merged_pot

        if all_hdr:
            merged_hdr = pd.concat(all_hdr, axis=0)
            store["hdr_0"] = merged_hdr
            del merged_hdr

    print(f"  Saved: {output_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-job det-var .df files."
    )
    parser.add_argument("--var", default=None,
                        help="Merge single variation (default: all found)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detvar-dir", default=DETVAR_DF_BASE)
    args = parser.parse_args()

    base = Path(args.detvar_dir)

    # Find variation subdirectories
    if args.var:
        var_dirs = [base / args.var]
    else:
        var_dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    if not var_dirs:
        print(f"No variation directories found in {base}")
        return

    print("═" * 60)
    print("  Merging det-var .df files")
    print("═" * 60)

    for var_dir in var_dirs:
        var = var_dir.name

        # Find all .df files (from grid jobs)
        df_files = sorted(glob.glob(str(var_dir / "*.df")))

        if not df_files:
            print(f"\n  {var}: no .df files found in {var_dir}")
            continue

        output_file = str(base / f"merged_{var}.df")

        print(f"\n── {var}: {len(df_files)} files ──")
        merge_df_files(df_files, output_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()