#!/usr/bin/env python3
"""
skim_nue.py
-----------
Pure concatenation skimmer for nueCC analysis.

Strategy
--------
1. Load one .df file at a time (never hold two full files in RAM).
2. Apply NO reco or truth cuts — the full DataFrame is written as-is.
3. Record pre-cut statistics into a JSON sidecar:
       n_true_signal : truth-nueCC-in-FV BEFORE any cut (efficiency denominator)
       n_inter_total : all reco interactions
   All cuts (flash-match, FV, topology, shower quality) are applied in the
   notebook via nue_selection.py, where both reco and truth indices are
   available together and the denominator is never biased by the skim.
4. Write the full DataFrame to a new compressed HDF5 file.

Why no cuts at skim time?
--------------------------
Any reco cut applied here would require re-running the skimmer to study a
different cut threshold.  Keeping the skimmed file cut-free means the
notebook has full flexibility, and the efficiency denominator (computed here
on the raw files) is never contaminated by a reco/truth mismatch.

Index anatomy reminder
----------------------
Full 4-level index: (__ntuple, entry, rec.dlp..index, rec.dlp.particles..index)
"Interaction level" = first 3 levels  →  drop level -1 to get it.

Running
-------
    python skim_nue.py [OPTIONS]

Options (all have defaults — edit the CONFIG block below or pass via CLI):
    --input-glob   glob pattern for input files  (default: see CONFIG)
    --output-dir   directory for skimmed files   (default: see CONFIG)
    --target-pot   target POT                    (default: 6.6e20)
    --n-files      max number of files to process (default: all)
    --dry-run      print what would be done, do not write files

Dependencies: numpy, pandas, tqdm, glob, json, argparse
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    input_glob  = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_*.df",
    output_dir  = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/skimmed",
    target_pot  = 6.6e20,
    n_files     = None,
    dry_run     = False,
    evt_key     = "evt_0",
    pot_key     = "histpotdf_0",
)

# ══════════════════════════════════════════════════════════════════════════════
# Index names
# ══════════════════════════════════════════════════════════════════════════════
IDX_NTUPLE   = "__ntuple"
IDX_ENTRY    = "entry"
IDX_INTER    = "rec.dlp..index"
IDX_PARTICLE = "rec.dlp.particles..index"
IL           = [IDX_NTUPLE, IDX_ENTRY, IDX_INTER]

# True-signal column suffixes — all resolved from the "dlp_true" branch
BRANCH_TRUE          = "dlp_true"
NU_ID_SUFFIX         = "nu_id"
CURRENT_TYPE_SUFFIX  = "current_type"
FV_TRUE_SUFFIX       = "is_fiducial"
PID_TRUE_SUFFIX      = "pid"
IS_PRIMARY_SUFFIX    = "is_primary"
KE_TRUE_SUFFIX       = "ke"

ELECTRON_THRESHOLD_MEV = 0.0
PID_ELECTRON           = 1


# ══════════════════════════════════════════════════════════════════════════════
# Column helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_col(df, suffix, branch_must_contain=None, branch_must_not_contain=None):
    """
    Find the first MultiIndex column whose last non-empty element equals `suffix`.

    Parameters
    ----------
    branch_must_contain     : str | None
        At least one element of the column tuple must contain this string.
    branch_must_not_contain : str | None
        No element of the column tuple may contain this string.
        Prevents picking up reco columns that share the same suffix as truth
        columns (e.g. "ke", "pid", "is_fiducial").
    """
    for col in df.columns:
        parts = [p for p in col if p != ""]
        if not parts or parts[-1] != suffix:
            continue
        if branch_must_contain and not any(branch_must_contain in p for p in col):
            continue
        if branch_must_not_contain and any(branch_must_not_contain in p for p in col):
            continue
        return col
    raise KeyError(
        f"Column suffix='{suffix}' branch_must_contain={branch_must_contain!r} "
        f"branch_must_not_contain={branch_must_not_contain!r} not found.\n"
        f"Available suffixes: {sorted({c[-1] for c in df.columns if c[-1]})}"
    )


def inter_index_from_df(df):
    """Return the unique 3-level interaction MultiIndex from the full 4-level df."""
    return df.index.droplevel(IDX_PARTICLE).unique()


# ══════════════════════════════════════════════════════════════════════════════
# True-signal counting (denominator only — no cuts applied to the data)
# ══════════════════════════════════════════════════════════════════════════════

def count_true_signal(df, label=""):
    """
    Count true nueCC-in-FV signal interactions.

    Definition (must exactly match true_signal_index in nue_selection.py):
        nu_id >= 0          (neutrino interaction)
        current_type == 0   (CC)
        is_fiducial == 1    (truth-FV)
        >= 1 primary true electron with KE > 75 MeV

    All columns resolved from BRANCH_TRUE ("dlp_true").
    branch_must_not_contain="rec.dlp." prevents picking up reco columns
    that share the same suffix.

    Returns int (0 if required columns are missing).
    """
    try:
        nu_col = _find_col(df, NU_ID_SUFFIX,       branch_must_contain=BRANCH_TRUE)
        cc_col = _find_col(df, CURRENT_TYPE_SUFFIX, branch_must_contain=BRANCH_TRUE)
        fv_col = _find_col(df, FV_TRUE_SUFFIX,
                           branch_must_contain=BRANCH_TRUE,
                           branch_must_not_contain="rec.dlp.")
    except KeyError as e:
        print(f"  WARNING (true signal{' '+label if label else ''}): {e}")
        return 0

    inter = df.groupby(level=IL).first()
    is_nu = inter[nu_col] >= 0
    is_cc = inter[cc_col] == 0
    is_fv = inter[fv_col] == 1

    try:
        pid_col = _find_col(df, PID_TRUE_SUFFIX,
                            branch_must_contain=BRANCH_TRUE,
                            branch_must_not_contain="rec.dlp.")
        pri_col = _find_col(df, IS_PRIMARY_SUFFIX,
                            branch_must_contain=BRANCH_TRUE,
                            branch_must_not_contain="rec.dlp.")
        ke_col  = _find_col(df, KE_TRUE_SUFFIX,
                            branch_must_contain=BRANCH_TRUE,
                            branch_must_not_contain="rec.dlp.")
    except KeyError as e:
        print(f"  WARNING (true electron{' '+label if label else ''}): {e}")
        return 0

    # KE scale guard — warn if values look like GeV instead of MeV
    raw_ke     = df[ke_col]
    nonzero_ke = raw_ke[raw_ke > 0]
    if len(nonzero_ke) > 0 and nonzero_ke.median() < 1.0:
        print(
            f"  WARNING: true KE median={nonzero_ke.median():.4f} — looks like GeV, "
            f"not MeV. Threshold {ELECTRON_THRESHOLD_MEV} MeV will reject everything."
        )

    elec_mask = (
        (df[pid_col] == PID_ELECTRON) &
        (df[pri_col] == 1) &
        (df[ke_col]  > ELECTRON_THRESHOLD_MEV)
    )
    elec_count = elec_mask.groupby(level=IL).sum()
    has_single_elec = elec_count == 1
    
    sig = is_nu & is_cc & is_fv & has_single_elec.reindex(inter.index, fill_value=False)
    return int(sig.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic: column resolution report (first file only)
# ══════════════════════════════════════════════════════════════════════════════

def _print_truth_column_report(df):
    """Print resolved column tuples for every truth suffix. Call once on first file."""
    suffixes = [NU_ID_SUFFIX, CURRENT_TYPE_SUFFIX, FV_TRUE_SUFFIX,
                PID_TRUE_SUFFIX, IS_PRIMARY_SUFFIX, KE_TRUE_SUFFIX]
    print("\n  ── Truth column resolution ──")
    for suf in suffixes:
        try:
            col = _find_col(df, suf, branch_must_contain=BRANCH_TRUE,
                            branch_must_not_contain="rec.dlp.")
            print(f"    {suf:<20s} → {col}")
        except KeyError:
            print(f"    {suf:<20s} → NOT FOUND")
    try:
        reco_fv = _find_col(df, FV_TRUE_SUFFIX, branch_must_contain="rec.dlp.",
                            branch_must_not_contain=BRANCH_TRUE)
        print(f"    {'is_fiducial (reco)':<20s} → {reco_fv}  [correctly excluded from truth search]")
    except KeyError:
        print("    is_fiducial (reco) → NOT FOUND")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# POT helper
# ══════════════════════════════════════════════════════════════════════════════

def read_pot(path, pot_key):
    """Read total POT from histpotdf table. Returns float or 0.0."""
    try:
        with pd.HDFStore(path, mode="r") as store:
            if "/" + pot_key in store.keys():
                pot_df = store[pot_key]
            else:
                candidates = [k for k in store.keys() if "histpotdf" in k]
                if not candidates:
                    return 0.0
                pot_df = store[candidates[0]]
        if "pot" in pot_df.columns:
            return float(pot_df["pot"].sum())
        return float(pot_df.select_dtypes(include="number").values.sum())
    except Exception as e:
        print(f"  WARNING: could not read POT from {path}: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Per-file processing
# ══════════════════════════════════════════════════════════════════════════════

def process_file(src_path, dst_path, pot_key, dry_run=False, print_col_report=False):
    """
    Copy one file with compression and record its denominator stats.
    No selection is applied — the full DataFrame is written.
    Returns a stats dict.
    """
    t0    = time.time()
    stats = dict(src=src_path, dst=dst_path)

    print(f"\n{'─'*70}")
    print(f"  SRC : {src_path}")
    print(f"  DST : {dst_path}")

    # ── Load ──────────────────────────────────────────────────────────────
    print("  Loading evt table …", end=" ", flush=True)
    try:
        with pd.HDFStore(src_path, mode="r") as store:
            keys           = store.keys()
            evt_candidates = [k for k in keys if k.lstrip("/").startswith("evt")]
            if not evt_candidates:
                raise KeyError(f"No 'evt' key found. Keys: {keys}")
            df = store[evt_candidates[0]]
    except Exception as e:
        print(f"\n  ERROR loading {src_path}: {e}")
        stats["error"] = str(e)
        return stats

    print(f"shape={df.shape}  ({time.time()-t0:.1f}s)")

    # ── Column resolution report (first file only) ─────────────────────────
    if print_col_report:
        _print_truth_column_report(df)

    # ── Count denominator on the full raw df ──────────────────────────────
    print("  Counting true signal …", end=" ", flush=True)
    t1 = time.time()
    n_true_sig = count_true_signal(df, label="pre-cut")
    n_inter    = inter_index_from_df(df).nunique()
    print(f"  n_inter={n_inter:,}  n_true_sig={n_true_sig:,}  ({time.time()-t1:.1f}s)")

    # ── POT ───────────────────────────────────────────────────────────────
    pot = read_pot(src_path, pot_key)
    print(f"  POT = {pot:.4e}")

    # ── Stats dict ────────────────────────────────────────────────────────
    stats.update(dict(
        n_rows          = len(df),
        n_inter_total   = n_inter,
        n_true_signal   = n_true_sig,   # ← the one true efficiency denominator
        pot             = pot,
        elapsed_s       = round(time.time() - t0, 1),
    ))

    # ── Write (compressed, no rows removed) ───────────────────────────────
    if dry_run:
        print("  DRY RUN — not writing.")
    else:
        print(f"  Writing → {dst_path} …", end=" ", flush=True)
        t2 = time.time()
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df.to_hdf(
                dst_path, key="evt_0", mode="w",
                complevel=1, complib="blosc",
            )
            # Copy histpotdf so downstream load_dfs still works
            try:
                with pd.HDFStore(src_path, mode="r") as src_store:
                    with pd.HDFStore(dst_path, mode="a") as dst_store:
                        for k in src_store.keys():
                            if k.lstrip("/").startswith("histpot"):
                                dst_store[k] = src_store[k]
            except Exception as e:
                print(f"\n  WARNING: could not copy histpotdf: {e}")
        print(f"done ({time.time()-t2:.1f}s)")

    del df

    stats["elapsed_s"] = round(time.time() - t0, 1)
    print(f"  File done in {stats['elapsed_s']:.1f}s")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="nueCC skimmer — compression only, no cuts")
    parser.add_argument("--input-glob",  default=CONFIG["input_glob"])
    parser.add_argument("--output-dir",  default=CONFIG["output_dir"])
    parser.add_argument("--target-pot",  type=float, default=CONFIG["target_pot"])
    parser.add_argument("--n-files",     type=int,   default=CONFIG["n_files"])
    parser.add_argument("--dry-run",     action="store_true", default=CONFIG["dry_run"])
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print(f"ERROR: no files found matching {args.input_glob!r}")
        sys.exit(1)
    if args.n_files:
        files = files[:args.n_files]

    print(f"Found {len(files)} file(s) to process.")
    print(f"Output directory: {args.output_dir}")
    print(f"Dry run: {args.dry_run}\n")
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for i, src in enumerate(tqdm(files, desc="Files", unit="file")):
        stem  = Path(src).stem
        dst   = os.path.join(args.output_dir, f"{stem}_skimmed.df")
        stats = process_file(
            src_path         = src,
            dst_path         = dst,
            pot_key          = CONFIG["pot_key"],
            dry_run          = args.dry_run,
            print_col_report = (i == 0),
        )
        all_stats.append(stats)

    # ── Summary JSON (one entry per file) ─────────────────────────────────
    summary_path = os.path.join(args.output_dir, "skim_summary.json")
    if not args.dry_run:
        with open(summary_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nSummary written → {summary_path}")

    # ── Aggregate printout ─────────────────────────────────────────────────
    ok = [s for s in all_stats if "error" not in s]
    print("\n" + "═"*70)
    print("AGGREGATE TOTALS  (no cuts applied — all interactions kept)")
    print("═"*70)
    print(f"  Files processed OK  : {len(ok)}/{len(all_stats)}")
    if ok:
        total_pot = sum(s["pot"]           for s in ok)
        n_sig     = sum(s["n_true_signal"] for s in ok)
        print(f"  Total rows          : {sum(s['n_rows']         for s in ok):>15,}")
        print(f"  Total interactions  : {sum(s['n_inter_total']  for s in ok):>15,}")
        print(f"  True signal (denom) : {n_sig:>15,}  ← use this in notebook")
        print(f"  Total POT           : {total_pot:.4e}")
        print(f"  POT scale factor    : {args.target_pot / total_pot:.4f}")
        print(f"  Total wall time     : {sum(s['elapsed_s'] for s in ok):.0f}s")
    print("═"*70)

    # ── Efficiency-denominator JSON ────────────────────────────────────────
    denom_path = os.path.join(args.output_dir, "efficiency_denominators.json")
    if not args.dry_run and ok:
        total_pot = sum(s["pot"]           for s in ok)
        n_sig     = sum(s["n_true_signal"] for s in ok)
        denom = {
            "n_inter_total"  : sum(s["n_inter_total"] for s in ok),
            # n_true_signal: truth-nueCC-in-FV before ANY cut.
            # This is the single efficiency denominator for all notebook cuts.
            # All per-cut signal counts (FM, FV, single_electron, ...) are
            # computed in the notebook via true_signal_index(evtdf) intersected
            # with each cut_flow stage.
            "n_true_signal"  : n_sig,
            "total_pot"      : total_pot,
            "target_pot"     : args.target_pot,
            "pot_scale"      : args.target_pot / total_pot,
            "files"          : [s["src"] for s in ok],
            "_note"          : (
                "No cuts applied at skim time. All selection cuts (flash-match, "
                "FV, single_electron, shower quality) are applied in the notebook "
                "via nue_selection.py. Use n_true_signal as the efficiency "
                "denominator and true_signal_index(evtdf) for per-stage signal counts."
            ),
        }
        with open(denom_path, "w") as f:
            json.dump(denom, f, indent=2)
        print(f"Efficiency denominators → {denom_path}")


if __name__ == "__main__":
    main()