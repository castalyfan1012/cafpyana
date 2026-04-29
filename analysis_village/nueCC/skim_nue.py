#!/usr/bin/env python3
"""
skim_nue.py  (v2 — FM+FV preselection + truth metadata in HDF5)
----------------------------------------------------------------
Strategy
--------
1.  Load one raw .df file at a time.
2.  Build per-interaction truth metadata covering ALL interactions (pre-cut).
    This table (truth_info_0) is the single source of truth for every
    efficiency/purity calculation — no JSON sidecar needed.
3.  Apply reco FM + FV preselection; write only passing rows to evt_0.
4.  truth_info_0 carries both truth flags AND reco-presel flags so the
    notebook can reconstruct every cut-stage count without re-applying cuts
    to the (now-smaller) particle-level df.

truth_info_0 columns (one row per interaction, 3-level index)
─────────────────────────────────────────────────────────────
  Truth flags (computed on raw df, before any cut):
    truth_cat          int8    0-7  (matches nue_selection.classify_truth)
    is_true_signal     bool    cat == 0
    is_nu              bool    nu_id >= 0
    is_cc              bool    current_type == 0
    is_fv_true         bool    truth is_fiducial == 1
    has_true_electron  bool    exactly 1 primary e- above threshold
    has_true_muon      bool    >= 1 primary mu above threshold
    has_pi0            bool    any particle with |parent_pdg| == 111

  Reco preselection flags (let notebook reconstruct FM / FV stage counts):
    passed_fm          bool    reco is_flash_matched == 1
    passed_fv_reco     bool    reco is_fiducial == 1
    passed_presel      bool    passed_fm & passed_fv_reco

Truth categories:
    0  nueCC FV          <- signal
    1  nueCC out FV
    2  numuCC + pi0
    3  NC pi0
    4  other numuCC
    5  other NC
    6  cosmic / non-neutrino
    7  neutrino catch-all

Workflow inside process_file():
    load raw df  (4-level index)
      |
      +-- build_truth_info()   -> truth_info (ALL interactions, truth+presel flags)
      +-- apply_preselection() -> filtered particle-level df (FM+FV only)
      |
      +-- write  evt_0          (preselected, compressed)
          write  truth_info_0   (all interactions, compressed)
          copy   histpotdf_0

Running
-------
    python skim_nue.py [OPTIONS]

    --input-glob   glob for input files   (default: CONFIG)
    --output-dir   output directory       (default: CONFIG)
    --target-pot   target POT             (default: 6.6e20)
    --n-files      max files to process   (default: all)
    --dry-run      print plan, no writes

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
    input_glob = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_*.df",
    output_dir = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/skimmed_df",
    target_pot = 6.6e20,
    n_files    = None,
    dry_run    = False,
    pot_key    = "histpotdf_0",
)


# =============================================================================
# Index / branch constants
# =============================================================================
IDX_NTUPLE   = "__ntuple"
IDX_ENTRY    = "entry"
IDX_INTER    = "rec.dlp..index"
IDX_PARTICLE = "rec.dlp.particles..index"
IL           = [IDX_NTUPLE, IDX_ENTRY, IDX_INTER]   # 3-level interaction key

BRANCH_TRUE  = "dlp_true"   # substring present in every truth-branch column tuple

# Truth column suffixes (resolved in the dlp_true branch)
_S_NU_ID      = "nu_id"
_S_CURR_TYPE  = "current_type"
_S_FV         = "is_fiducial"
_S_PID        = "pid"
_S_IS_PRIMARY = "is_primary"
_S_KE         = "ke"
_S_PARENT_PDG = "parent_pdg_code"

# Reco column suffixes (resolved in the dlp branch, not dlp_true)
_S_FLASH      = "is_flash_matched"
# FV reuses _S_FV

# Thresholds — must match nue_selection.py
ELECTRON_THRESHOLD_MEV = 75.0
MUON_THRESHOLD_MEV     = 50.0

# PID codes — must match nue_selection.py
PID_ELECTRON = 1
PID_MUON     = 2

# HDF5 output keys
EVT_KEY_OUT    = "evt_0"
TRUTH_INFO_KEY = "truth_info_0"


# =============================================================================
# Column helpers
# =============================================================================

def _find_col(df, suffix, branch_must_contain=None, branch_must_not_contain=None):
    """
    Return the first MultiIndex column whose last non-empty part equals suffix.

    branch_must_contain     : at least one tuple element contains this string.
    branch_must_not_contain : no tuple element may contain this string.
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
        f"suffix={suffix!r} branch_must_contain={branch_must_contain!r} "
        f"branch_must_not_contain={branch_must_not_contain!r} not found.\n"
        f"Last-part suffixes: {sorted({c[-1] for c in df.columns if c[-1]})}"
    )


def _truth(df, suffix):
    """Find a column in the dlp_true (truth) branch."""
    return _find_col(df, suffix, branch_must_contain=BRANCH_TRUE)


def _reco(df, suffix):
    """Find a column in the reco (dlp, NOT dlp_true) branch."""
    return _find_col(df, suffix, branch_must_not_contain=BRANCH_TRUE)


def _find_col_ends(df, *suffix_parts, branch_must_contain=None):
    """
    Find the first column whose last N non-empty parts equal suffix_parts.
    Useful for nested sub-fields like ('momentum', 'x') where the single-suffix
    search is ambiguous.
    """
    n      = len(suffix_parts)
    target = tuple(suffix_parts)
    for col in df.columns:
        parts = tuple(p for p in col if p != "")
        if len(parts) < n or parts[-n:] != target:
            continue
        if branch_must_contain and not any(branch_must_contain in p for p in col):
            continue
        return col
    raise KeyError(
        f"No column ending with {suffix_parts!r} "
        f"(branch_must_contain={branch_must_contain!r})"
    )


# =============================================================================
# Core: build truth metadata + reco preselection flags in one pass
# =============================================================================

def build_truth_info(df, label=""):
    """
    Build compact per-interaction truth + reco-preselection metadata.

    Operates on the raw 4-level df (before any cut) so:
      - truth flags are never biased by reco decisions
      - reco presel flags record which interactions passed FM / FV
      - the notebook reconstructs all cut-stage counts from this table alone

    Returns
    -------
    pd.DataFrame  3-level interaction MultiIndex, columns as documented above.
    Empty DataFrame on critical column-resolution failure.
    """
    # ── Resolve columns (non-fatal: missing reco columns degrade gracefully) ─
    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except KeyError:
            return None

    nu_col   = _safe(_truth, df, _S_NU_ID)
    cc_col   = _safe(_truth, df, _S_CURR_TYPE)
    fv_t_col = _safe(_truth, df, _S_FV)
    pid_col  = _safe(_truth, df, _S_PID)
    pri_col  = _safe(_truth, df, _S_IS_PRIMARY)
    ke_col   = _safe(_truth, df, _S_KE)
    ppd_col  = _safe(_truth, df, _S_PARENT_PDG)
    fm_col   = _safe(_reco,  df, _S_FLASH)
    fv_r_col = _safe(_reco,  df, _S_FV)

    required = [nu_col, cc_col, fv_t_col, pid_col, pri_col, ke_col, ppd_col]
    if any(c is None for c in required):
        missing = [n for n, c in zip(
            ["nu_id", "current_type", "is_fiducial(T)", "pid", "is_primary", "ke", "parent_pdg_code"],
            required,
        ) if c is None]
        print(f"  ERROR build_truth_info({label}): missing truth columns: {missing}")
        return pd.DataFrame()

    # ── KE scale sanity guard ─────────────────────────────────────────────
    nz = df[ke_col][df[ke_col] > 0]
    if len(nz) and nz.median() < 1.0:
        print(
            f"  WARNING: true KE median={nz.median():.4f} looks like GeV, not MeV. "
            f"Threshold {ELECTRON_THRESHOLD_MEV} MeV will reject everything."
        )

    # ── Interaction-level index ───────────────────────────────────────────
    inter = df.groupby(level=IL).first()
    idx   = inter.index

    # ── Truth flags (interaction level) ──────────────────────────────────
    is_nu      = (inter[nu_col]   >= 0).reindex(idx, fill_value=False)
    is_cc      = (inter[cc_col]   == 0).reindex(idx, fill_value=False)
    is_fv_true = (inter[fv_t_col] == 1).reindex(idx, fill_value=False)

    # ── Truth flags (particle-level → aggregated per interaction) ─────────
    _elec_mask = (
        (df[pid_col] == PID_ELECTRON) &
        (df[pri_col] == 1) &
        (df[ke_col]  > ELECTRON_THRESHOLD_MEV)
    )
    has_true_electron = _elec_mask.groupby(level=IL).sum().reindex(idx, fill_value=0) == 1

    has_true_muon = (
        (df[pid_col] == PID_MUON) &
        (df[pri_col] == 1) &
        (df[ke_col]  > MUON_THRESHOLD_MEV)
    ).groupby(level=IL).any().reindex(idx, fill_value=False)

    has_pi0 = (
        df[ppd_col].abs() == 111
    ).groupby(level=IL).any().reindex(idx, fill_value=False)

    # ── Truth category (exactly mirrors classify_truth in nue_selection.py) ─
    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                           = 6  # cosmic
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_true_electron]   = 1  # nueCC out FV
    cat[nu &  is_cc &  is_fv_true & has_true_electron]   = 0  # nueCC in FV <- SIGNAL
    cat[nu &  is_cc & has_true_muon & has_pi0]            = 2  # numuCC + pi0
    cat[nu & ~is_cc & has_pi0]                            = 3  # NC pi0
    cat[nu &  is_cc & has_true_muon & ~has_pi0]           = 4  # other numuCC
    cat[nu & ~is_cc & ~has_pi0]                           = 5  # other NC
    # cat 7 = neutrino catch-all (anything not matched above)

    # ── Reco preselection flags ───────────────────────────────────────────

    
    if fm_col is not None:
        passed_fm = (
            (df[fm_col] == 1).groupby(level=IL).any().reindex(idx, fill_value=False)
        )
    else:
        print("  WARNING: FM column missing — passed_fm = False everywhere.")
        passed_fm = pd.Series(False, index=idx)

    if fv_r_col is not None:
        passed_fv_reco = (
            (df[fv_r_col] == 1).groupby(level=IL).any().reindex(idx, fill_value=False)
        )
    else:
        print("  WARNING: reco FV column missing — passed_fv_reco = False everywhere.")
        passed_fv_reco = pd.Series(False, index=idx)

    passed_presel = passed_fm & passed_fv_reco

    # ── Leading true-electron kinematics (saved for confusion-matrix cells) ──
    # Stored for ALL interactions; NaN for interactions without a primary electron.
    # Enables the notebook to build efficiency matrices using the full pre-cut
    # population from truth_info alone (no need to access the particle-level df).
    true_leading_e_ke       = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_costheta = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_p        = pd.Series(np.nan, index=idx, dtype=np.float32)

    try:
        # Use 3-part suffix to get particle-level momentum, not interaction-level
        # neutrino momentum (rec->dlp_true->momentum->x vs
        # rec->dlp_true->particles->momentum->x — both end in ("momentum","x"))
        momx_col = _find_col_ends(df, "particles", "momentum", "x", branch_must_contain=BRANCH_TRUE)
        momy_col = _find_col_ends(df, "particles", "momentum", "y", branch_must_contain=BRANCH_TRUE)
        momz_col = _find_col_ends(df, "particles", "momentum", "z", branch_must_contain=BRANCH_TRUE)

        # KE of all primary electrons; -inf elsewhere so idxmax ignores non-electrons
        elec_ke_for_max = df[ke_col].where(_elec_mask, -np.inf)
        lead_4idx = elec_ke_for_max.groupby(level=IL).idxmax().dropna()

        if len(lead_4idx) > 0:
            # Select the leading-electron rows from the full df
            lead_mi      = pd.MultiIndex.from_tuples(
                lead_4idx.values, names=df.index.names
            )
            leading_rows = df.loc[lead_mi, [ke_col, momx_col, momy_col, momz_col]].copy()
            leading_rows.index = lead_4idx.index   # replace 4-level → 3-level (interaction)

            px   = leading_rows[momx_col]
            py   = leading_rows[momy_col]
            pz   = leading_rows[momz_col]
            pmag = np.sqrt(px**2 + py**2 + pz**2).replace(0, np.nan)

            true_leading_e_ke       = leading_rows[ke_col].reindex(idx).astype(np.float32)
            true_leading_e_costheta = (pz / pmag).reindex(idx).astype(np.float32)
            true_leading_e_p        = pmag.reindex(idx).astype(np.float32)
    except Exception as e:
        print(f"  WARNING: leading-electron kinematics skipped: {e}")

    # ── Assemble ──────────────────────────────────────────────────────────
    return pd.DataFrame(
        {
            "truth_cat"              : cat,
            "is_true_signal"         : (cat == 0),
            "is_nu"                  : is_nu,
            "is_cc"                  : is_cc,
            "is_fv_true"             : is_fv_true,
            "has_true_electron"      : has_true_electron,
            "has_true_muon"          : has_true_muon,
            "has_pi0"                : has_pi0,
            "passed_fm"              : passed_fm,
            "passed_fv_reco"         : passed_fv_reco,
            "passed_presel"          : passed_presel,
            # Leading true-electron kinematics (NaN for non-signal interactions)
            "true_leading_e_ke"      : true_leading_e_ke,
            "true_leading_e_costheta": true_leading_e_costheta,
            "true_leading_e_p"       : true_leading_e_p,
        },
        index=idx,
    ).astype({
        "truth_cat"         : np.int8,
        "is_true_signal"    : bool,
        "is_nu"             : bool,
        "is_cc"             : bool,
        "is_fv_true"        : bool,
        "has_true_electron" : bool,
        "has_true_muon"     : bool,
        "has_pi0"           : bool,
        "passed_fm"         : bool,
        "passed_fv_reco"    : bool,
        "passed_presel"     : bool,
    })


# =============================================================================
# Preselection: filter particle rows using truth_info flags (no recomputation)
# =============================================================================

def apply_preselection(df, truth_info):
    """
    Keep only particle rows whose interaction passed FM + FV (passed_presel).
    Reads flags from truth_info — no re-computation against df columns needed.

    Returns (df_presel, stats_dict).
    """
    presel_idx = truth_info.index[truth_info["passed_presel"]]
    row_mask   = df.index.droplevel(-1).isin(presel_idx)
    stats = dict(
        n_all      = len(truth_info),
        n_after_fm = int(truth_info["passed_fm"].sum()),
        n_presel   = int(truth_info["passed_presel"].sum()),
    )
    return df[row_mask], stats


# =============================================================================
# Diagnostic: column resolution report (first file only)
# =============================================================================

def _print_column_report(df):
    print("\n  -- Column resolution -----------------------------------------")
    for branch, suf, kw in [
        ("truth", _S_NU_ID,      {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_CURR_TYPE,  {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_FV,         {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_PID,        {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_IS_PRIMARY, {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_KE,         {"branch_must_contain": BRANCH_TRUE}),
        ("truth", _S_PARENT_PDG, {"branch_must_contain": BRANCH_TRUE}),
        ("reco",  _S_FLASH,      {"branch_must_not_contain": BRANCH_TRUE}),
        ("reco",  _S_FV,         {"branch_must_not_contain": BRANCH_TRUE}),
    ]:
        try:
            col = _find_col(df, suf, **kw)
            print(f"    [{branch}] {suf:<22s} -> {col}")
        except KeyError:
            print(f"    [{branch}] {suf:<22s} -> NOT FOUND")
    print()


# =============================================================================
# POT helper
# =============================================================================

def read_pot(path, pot_key):
    try:
        with pd.HDFStore(path, mode="r") as store:
            cands = [k for k in store.keys() if "histpotdf" in k]
            if not cands:
                return 0.0
            key    = ("/" + pot_key) if ("/" + pot_key) in store.keys() else cands[0]
            pot_df = store[key]
        col = "pot" if "pot" in pot_df.columns else pot_df.select_dtypes("number").columns[0]
        return float(pot_df[col].sum())
    except Exception as e:
        print(f"  WARNING: could not read POT: {e}")
        return 0.0


# =============================================================================
# Per-file processing
# =============================================================================

def process_file(src_path, dst_path, pot_key, dry_run=False, print_col_report=False):
    """
    1. Load raw df.
    2. Build truth_info (ALL interactions: truth flags + reco presel flags).
    3. Apply FM+FV preselection.
    4. Write evt_0 + truth_info_0 + histpotdf.
    """
    t0    = time.time()
    stats = dict(src=str(src_path), dst=str(dst_path))

    # ── Load ─────────────────────────────────────────────────────────────
    try:
        with pd.HDFStore(src_path, mode="r") as store:
            evt_keys = [k for k in store.keys() if k.lstrip("/").startswith("evt")]
            if not evt_keys:
                raise KeyError(f"No evt key. Keys: {list(store.keys())}")
            df = store[evt_keys[0]]
    except Exception as e:
        tqdm.write(f"  ERROR loading {src_path}: {e}")
        stats["error"] = str(e)
        return stats

    if print_col_report:
        _print_column_report(df)

    # ── Build truth_info (pre-cut, ALL interactions) ──────────────────────
    truth_info = build_truth_info(df, label=Path(src_path).stem)

    if truth_info.empty:
        stats["error"] = "truth_info empty"
        return stats

    n_inter_total = len(truth_info)
    n_true_signal = int(truth_info["is_true_signal"].sum())
    n_after_fm    = int(truth_info["passed_fm"].sum())
    n_presel      = int(truth_info["passed_presel"].sum())
    n_sig_fm      = int(truth_info.loc[truth_info["passed_fm"],    "is_true_signal"].sum())
    n_sig_presel  = int(truth_info.loc[truth_info["passed_presel"],"is_true_signal"].sum())
    cat_counts    = truth_info["truth_cat"].value_counts().sort_index().to_dict()

    # ── Preselection (uses truth_info flags — no recomputation needed) ────
    df_presel, ps = apply_preselection(df, truth_info)
    pct           = 100.0 * (1 - len(df_presel) / max(len(df), 1))

    pot = read_pot(src_path, pot_key)

    # Single-line summary per file (keeps tqdm bar readable)
    tqdm.write(
        f"  {Path(src_path).name:<40s}  "
        f"inter={n_inter_total:>7,}  sig={n_true_signal:>4,}  "
        f"presel={n_presel:>6,}  sig_presel={n_sig_presel:>4,}  "
        f"rows {len(df):,}->{len(df_presel):,} ({pct:.0f}%)  "
        f"POT={pot:.3e}  {time.time()-t0:.1f}s"
    )

    stats.update(dict(
        n_rows_raw          = len(df),
        n_rows_presel       = len(df_presel),
        n_inter_total       = n_inter_total,
        n_inter_after_fm    = n_after_fm,
        n_inter_presel      = n_presel,
        n_true_signal       = n_true_signal,
        n_sig_after_fm      = n_sig_fm,
        n_sig_after_presel  = n_sig_presel,
        truth_cat_counts    = {int(k): int(v) for k, v in cat_counts.items()},
        pot                 = pot,
    ))

    del df  # free memory before writing

    # ── Write ─────────────────────────────────────────────────────────────
    if dry_run:
        tqdm.write(f"  DRY RUN -- nothing written: {Path(dst_path).name}")
    else:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_presel.to_hdf(
                dst_path, key=EVT_KEY_OUT, mode="w",
                complevel=1, complib="blosc",
            )
            truth_info.to_hdf(
                dst_path, key=TRUTH_INFO_KEY, mode="a",
                complevel=1, complib="blosc",
            )
            try:
                with pd.HDFStore(src_path, mode="r") as src_store:
                    with pd.HDFStore(dst_path, mode="a") as dst_store:
                        for k in src_store.keys():
                            if k.lstrip("/").startswith("histpot"):
                                dst_store[k] = src_store[k]
            except Exception as e:
                tqdm.write(f"  WARNING: could not copy histpotdf: {e}")

    del df_presel

    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="nueCC skimmer v2 -- FM+FV preselection + truth_info_0 in HDF5"
    )
    parser.add_argument("--input-glob",  default=CONFIG["input_glob"])
    parser.add_argument("--output-dir",  default=CONFIG["output_dir"])
    parser.add_argument("--target-pot",  type=float, default=CONFIG["target_pot"])
    parser.add_argument("--nfiles",      type=int,   default=CONFIG["n_files"])
    parser.add_argument("--dry-run",     action="store_true", default=CONFIG["dry_run"])
    args = parser.parse_args()

    files = sorted(_glob.glob(args.input_glob))
    if not files:
        print(f"ERROR: no files match {args.input_glob!r}")
        sys.exit(1)
    if args.nfiles:
        files = files[: args.nfiles]

    print(f"Found {len(files)} file(s).")
    print(f"Output dir : {args.output_dir}")
    print(f"Dry run    : {args.dry_run}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for i, src in enumerate(tqdm(files, desc="Files", unit="file")):
        stem = Path(src).stem
        dst  = os.path.join(args.output_dir, f"{stem}_skimmed.df")
        s    = process_file(
            src_path         = src,
            dst_path         = dst,
            pot_key          = CONFIG["pot_key"],
            dry_run          = args.dry_run,
            print_col_report = (i == 0),
        )
        all_stats.append(s)

    ok = [s for s in all_stats if "error" not in s]
    print("\n" + "=" * 72)
    print("AGGREGATE TOTALS  (truth_info_0 = permanent denominator source)")
    print("=" * 72)
    print(f"  Files OK                : {len(ok)}/{len(all_stats)}")
    if ok:
        total_pot    = sum(s["pot"]               for s in ok)
        n_sig        = sum(s["n_true_signal"]      for s in ok)
        n_sig_fm     = sum(s["n_sig_after_fm"]     for s in ok)
        n_sig_pre    = sum(s["n_sig_after_presel"] for s in ok)
        n_inter_all  = sum(s["n_inter_total"]      for s in ok)
        n_inter_pre  = sum(s["n_inter_presel"]     for s in ok)
        print(f"  Rows raw -> presel      : "
              f"{sum(s['n_rows_raw'] for s in ok):,} -> "
              f"{sum(s['n_rows_presel'] for s in ok):,}")
        print(f"  Interactions (all)      : {n_inter_all:>15,}")
        print(f"  Interactions (FM+FV)    : {n_inter_pre:>15,}")
        print(f"  True signal (pre-cut)   : {n_sig:>15,}  <- denominator")
        print(f"  True signal after FM    : {n_sig_fm:>15,}  ({n_sig_fm/n_sig*100:.1f}%)")
        print(f"  True signal after FM+FV : {n_sig_pre:>15,}  ({n_sig_pre/n_sig*100:.1f}%)")
        print(f"  Total POT               : {total_pot:.4e}")
        print(f"  Scale to {args.target_pot:.2e}  : {args.target_pot/total_pot:.4f}")
        print(f"  Total wall time         : {sum(s['elapsed_s'] for s in ok):.0f}s")
    print("=" * 72)
    print(
        "\nNOTE: all efficiency denominators are in truth_info_0 of each .df file.\n"
        "      Run merge_skimmed.py to combine files, then open the notebook."
    )


if __name__ == "__main__":
    main()