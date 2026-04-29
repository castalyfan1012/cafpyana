#!/usr/bin/env python3
"""
make_nueCC_df.py
----------------
SPINE DLP-based nueCC inclusive MC DataFrame makers for cafpyana.
Called from nueCC_mc.py via run_df_maker.py.

Two public functions
--------------------
make_nuecc_evtdf(f)
    Particle-level SPINE df (reco + truth), merged with the MC neutrino
    record, filtered to interactions passing reco flash-match + FV.
    Index: (entry, rec.dlp..index, rec.dlp.particles..index)
    Drop-in replacement for the Pandora make_nueccdf_mc — same column
    access pattern (nue_helpers.reco_particles / true_particles etc.).

make_nuecc_statsdf(f)
    One-row DataFrame with pre-cut interaction and signal counts.
    These are the efficiency denominators for the notebook cut-flow
    table; they are computed BEFORE any reco cut is applied so no
    signal is lost.

Truth-signal definition (must match nue_selection.py / skim_nue_v2.py):
    nu_id >= 0  &  current_type == 0  &  is_fiducial (truth) == 1
    &  exactly 1 primary true electron with KE > ELECTRON_THRESHOLD_MEV
"""

import warnings

import numpy as np
import pandas as pd

from makedf.makedf import (
    make_all_spine_df,
    make_mcnudf,
)
from pyanalib.pandas_helpers import multicol_merge


# =============================================================================
# Constants  (must match nue_selection.py and skim_nue_v2.py)
# =============================================================================
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 75.0
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2


# =============================================================================
# Column helpers  (same logic as skim_nue_v2, duplicated for standalone use)
# =============================================================================

def _find_col(df, suffix, branch_must_contain=None, branch_must_not_contain=None):
    """Return the first MultiIndex column whose last non-empty part == suffix."""
    for col in df.columns:
        parts = [p for p in (col if isinstance(col, tuple) else (col,)) if p != ""]
        if not parts or parts[-1] != suffix:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        if branch_must_not_contain and any(branch_must_not_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(
        f"suffix={suffix!r} branch_must_contain={branch_must_contain!r} "
        f"branch_must_not_contain={branch_must_not_contain!r} not found"
    )


def _find_col_ends(df, *suffix_parts, branch_must_contain=None):
    """Find column whose last N non-empty parts equal suffix_parts."""
    n      = len(suffix_parts)
    target = tuple(suffix_parts)
    for col in df.columns:
        parts = tuple(p for p in (col if isinstance(col, tuple) else (col,)) if p != "")
        if len(parts) < n or parts[-n:] != target:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(f"No column ending with {suffix_parts!r}")


def _safe(fn, *args, **kwargs):
    """Call fn; return None on KeyError / StopIteration."""
    try:
        return fn(*args, **kwargs)
    except (KeyError, StopIteration):
        return None


# =============================================================================
# Truth counting
# =============================================================================

def _build_truth_counts(df):
    """
    Compute per-interaction truth flags and reco preselection flags.

    Adapted from skim_nue_v2.build_truth_info; works on the SPINE df
    produced by make_all_spine_df (3-level index, no __ntuple level).

    Returns a dict with:
        idx           : interaction-level MultiIndex
        cat           : int8 Series (0-7) per interaction
        is_sig        : bool Series  (cat == 0)
        passed_fm     : bool Series  (reco flash-match)
        passed_presel : bool Series  (reco FM & FV)
    """
    il    = list(range(df.index.nlevels - 1))   # e.g. [0, 1] for 3-level df
    inter = df.groupby(level=il).first()
    idx   = inter.index

    def _b(s):
        """Reindex to idx, fill False."""
        return s.reindex(idx, fill_value=False).astype(bool)

    # ── Truth interaction-level flags ─────────────────────────────────────────
    nu_col   = _safe(_find_col, df, "nu_id",        branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, df, "current_type", branch_must_contain=BRANCH_TRUE)
    fv_t_col = _safe(_find_col, df, "is_fiducial",  branch_must_contain=BRANCH_TRUE,
                                                     branch_must_not_contain=None)
    is_nu      = _b(inter[nu_col]   >= 0) if nu_col   else pd.Series(False, index=idx)
    is_cc      = _b(inter[cc_col]   == 0) if cc_col   else pd.Series(False, index=idx)
    is_fv_true = _b(inter[fv_t_col] == 1) if fv_t_col else pd.Series(False, index=idx)

    # ── Truth particle-level → interaction aggregation ────────────────────────
    pid_col = _safe(_find_col, df, "pid",          branch_must_contain=BRANCH_TRUE)
    pri_col = _safe(_find_col, df, "is_primary",   branch_must_contain=BRANCH_TRUE)
    ke_col  = _safe(_find_col, df, "ke",           branch_must_contain=BRANCH_TRUE)
    ppd_col = _safe(_find_col, df, "parent_pdg_code", branch_must_contain=BRANCH_TRUE)

    if pid_col and pri_col and ke_col:
        elec_mask = (
            (df[pid_col] == PID_ELECTRON) &
            (df[pri_col] == 1) &
            (df[ke_col]  > ELECTRON_THRESHOLD_MEV)
        )
        has_elec = elec_mask.groupby(level=il).sum().reindex(idx, fill_value=0) == 1
    else:
        has_elec = pd.Series(False, index=idx)

    has_muon = (
        ((df[pid_col] == PID_MUON) & (df[pri_col] == 1) & (df[ke_col] > MUON_THRESHOLD_MEV))
        .groupby(level=il).any().reindex(idx, fill_value=False)
    ) if (pid_col and pri_col and ke_col) else pd.Series(False, index=idx)

    has_pi0 = (
        (df[ppd_col].abs() == 111).groupby(level=il).any().reindex(idx, fill_value=False)
    ) if ppd_col else pd.Series(False, index=idx)

    # ── Truth categories (mirrors classify_truth / build_truth_info) ──────────
    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                          = 6
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_elec]            = 1  # nueCC out FV
    cat[nu &  is_cc &  is_fv_true & has_elec]            = 0  # nueCC in FV ← signal
    cat[nu &  is_cc & has_muon    & has_pi0]             = 2  # numuCC + pi0
    cat[nu & ~is_cc & has_pi0]                           = 3  # NC pi0
    cat[nu &  is_cc & has_muon    & ~has_pi0]            = 4  # other numuCC
    cat[nu & ~is_cc & ~has_pi0]                          = 5  # other NC

    # ── Reco preselection flags ───────────────────────────────────────────────
    fm_col   = _safe(_find_col, df, "is_flash_matched", branch_must_not_contain=BRANCH_TRUE)
    fv_r_col = _safe(_find_col, df, "is_fiducial",      branch_must_not_contain=BRANCH_TRUE)

    passed_fm = (
        (df[fm_col] == 1).groupby(level=il).any().reindex(idx, fill_value=False)
    ) if fm_col else pd.Series(False, index=idx)

    passed_fv = (
        (df[fv_r_col] == 1).groupby(level=il).any().reindex(idx, fill_value=False)
    ) if fv_r_col else pd.Series(False, index=idx)

    return dict(
        idx           = idx,
        cat           = cat,
        is_sig        = (cat == 0),
        passed_fm     = passed_fm,
        passed_presel = passed_fm & passed_fv,
    )


# =============================================================================
# Public makers
# =============================================================================

def make_nuecc_evtdf(f):
    """
    Main nueCC analysis DataFrame for one CAF file.

    Steps
    -----
    1. Load full SPINE DLP df (reco + truth, interactions + particles) via
       make_all_spine_df.
    2. Merge in the MC neutrino record (make_mcnudf_nuecc) via mct_index —
       adds generator-level variables (is_cc, bjorken_x, particle counts …).
    3. Apply reco flash-match + fiducial-volume preselection.

    The returned df is structurally identical to the output of skim_nue_v2
    (evt_0 key) and is consumed by nue_selection.py without modification.

    Index  : (entry, rec.dlp..index, rec.dlp.particles..index)
    """
    # ── 1. SPINE df ───────────────────────────────────────────────────────────
    spine_df = make_all_spine_df(f)
    il       = list(range(spine_df.index.nlevels - 1))

    # ── 2. MC truth merge ─────────────────────────────────────────────────────
    # mct_index in rec.dlp_true links each interaction to the MC neutrino table.
    # Pandas multicol_merge on (entry, mct_index) ↔ (entry, rec.mc.nu..index).
    # In make_nuecc_evtdf, replace the MC merge block with:
    try:
        mcdf = make_mcnudf(f)   # use make_mcnudf directly, not make_mcnudf_nuecc
        mcdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["mcnu"] + list(c)) for c in mcdf.columns]
        )
        # Merge key confirmed from make_spine_int_mcnu_df:
        #   left:  ("rec", "dlp_true", "mct_index", "")
        #   right: (entry, rec.mc.nu..index)
        mct_col = _find_col(spine_df, "mct_index", branch_must_contain=BRANCH_TRUE)
    
        spineint_mcnu_df = multicol_merge(
            lhs=spine_df.reset_index(level=list(range(1, spine_df.index.nlevels))),
            rhs=mcdf.reset_index(),
            left_on  = ["entry", mct_col],
            right_on = ["entry", mcdf.index.names[-1]],  # "rec.mc.nu..index"
            how      = "left",
            validate = "many_to_one",
        )
        spine_df = spineint_mcnu_df.set_index(list(spine_df.index.names))
    except Exception as e:
        warnings.warn(f"make_nuecc_evtdf: MC merge skipped — {e}")

    # ── 3. Reco FM + FV preselection ─────────────────────────────────────────
    try:
        fm_col = _find_col(spine_df, "is_flash_matched", branch_must_not_contain=BRANCH_TRUE)
        fv_col = _find_col(spine_df, "is_fiducial",      branch_must_not_contain=BRANCH_TRUE)

        fm_pass  = (spine_df[fm_col] == 1).groupby(level=il).any()
        fv_pass  = (spine_df[fv_col] == 1).groupby(level=il).any()
        presel   = fm_pass[fm_pass].index.intersection(fv_pass[fv_pass].index)

        row_mask = spine_df.index.droplevel(-1).isin(presel)
        spine_df = spine_df[row_mask]

    except Exception as e:
        warnings.warn(f"make_nuecc_evtdf: FM/FV preselection skipped — {e}")

    return spine_df


def make_nuecc_statsdf(f):
    """
    One-row DataFrame with pre-cut interaction and truth-signal counts.

    These numbers are the efficiency denominators for the notebook cut-flow
    table.  They MUST be computed on the full, unfiltered SPINE df so that
    signal interactions that fail the reco preselection are still counted.

    Columns
    -------
    n_inter_total      all reconstructed interactions (pre-cut)
    n_true_signal      nueCC-in-FV signal interactions before any reco cut
    n_after_fm         interactions passing reco flash-match
    n_after_presel     interactions passing reco FM + FV
    n_sig_after_fm     true signal passing FM
    n_sig_after_presel true signal passing FM + FV
    cat_0 … cat_7      truth-category counts (same as skim_nue_v2)
    """
    empty = pd.DataFrame([{
        "n_inter_total": 0, "n_true_signal": 0,
        "n_after_fm": 0,    "n_after_presel": 0,
        "n_sig_after_fm": 0,"n_sig_after_presel": 0,
        **{f"cat_{c}": 0 for c in range(8)},
    }])

    try:
        spine_df = make_all_spine_df(f)
    except Exception as e:
        warnings.warn(f"make_nuecc_statsdf: could not load spine df — {e}")
        return empty

    try:
        tc = _build_truth_counts(spine_df)
    except Exception as e:
        warnings.warn(f"make_nuecc_statsdf: truth counting failed — {e}")
        return empty

    is_sig        = tc["is_sig"]
    passed_fm     = tc["passed_fm"]
    passed_presel = tc["passed_presel"]
    cat           = tc["cat"]

    cat_counts = {f"cat_{c}": int((cat == c).sum()) for c in range(8)}

    return pd.DataFrame([{
        "n_inter_total"     : len(tc["idx"]),
        "n_true_signal"     : int(is_sig.sum()),
        "n_after_fm"        : int(passed_fm.sum()),
        "n_after_presel"    : int(passed_presel.sum()),
        "n_sig_after_fm"    : int((is_sig & passed_fm).sum()),
        "n_sig_after_presel": int((is_sig & passed_presel).sum()),
        **cat_counts,
    }])