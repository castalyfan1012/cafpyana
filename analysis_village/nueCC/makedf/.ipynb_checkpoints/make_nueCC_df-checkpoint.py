#!/usr/bin/env python3
"""
make_nueCC_df.py
----------------
SPINE DLP-based nueCC inclusive MC DataFrame makers for cafpyana.

Public functions
----------------
make_nuecc_evtdf(f)   — reco+truth particle df, FM+FV preselected
make_nuecc_statsdf(f) — one-row pre-cut denominator table
make_nuecc_wgtdf(f)   — BNB+GENIE universe weights (preselected only)

Key design decisions
--------------------
* _get_spine_df()     caches the SPINE df so evtdf and statsdf share one load
* make_nuecc_wgtdf()  prefilters to FM+FV mct_index pairs BEFORE weight computation
  → GENIE processes ~5k interactions instead of ~80k (inspired by numuCC approach)
* All truth category definitions match nue_selection.py / skim_nue_v2.py exactly
"""

import warnings
import numpy as np
import pandas as pd

from makedf.makedf import make_all_spine_df, make_mcnudf
from pyanalib.pandas_helpers import multicol_merge, multicol_concat

# =============================================================================
# Constants — must match nue_selection.py
# =============================================================================
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 75.0
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2

# =============================================================================
# SPINE df cache — one load shared by evtdf + statsdf + wgtdf per file
# =============================================================================
_spine_cache = {}

def _get_spine_df(f):
    fname = str(f)
    if fname not in _spine_cache:
        _spine_cache.clear()
        _spine_cache[fname] = make_all_spine_df(f)
    return _spine_cache[fname]

# =============================================================================
# Column helpers
# =============================================================================
def _find_col(df, suffix, branch_must_contain=None, branch_must_not_contain=None):
    for col in df.columns:
        parts = [p for p in (col if isinstance(col, tuple) else (col,)) if p != ""]
        if not parts or parts[-1] != suffix:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        if branch_must_not_contain and any(branch_must_not_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(f"suffix={suffix!r} bc={branch_must_contain!r} bnc={branch_must_not_contain!r}")

def _find_col_ends(df, *suffix_parts, branch_must_contain=None):
    n, target = len(suffix_parts), tuple(suffix_parts)
    for col in df.columns:
        parts = tuple(p for p in (col if isinstance(col, tuple) else (col,)) if p != "")
        if len(parts) < n or parts[-n:] != target:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(f"No column ending with {suffix_parts!r}")

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (KeyError, StopIteration):
        return None

# =============================================================================
# FM+FV preselection — returns (presel_index, fm_pass, fv_pass)
# =============================================================================
def _get_presel(spine_df):
    il = list(range(spine_df.index.nlevels - 1))
    try:
        fm_col  = _find_col(spine_df, "is_flash_matched", branch_must_not_contain=BRANCH_TRUE)
        fv_col  = _find_col(spine_df, "is_fiducial",      branch_must_not_contain=BRANCH_TRUE)
        fm_pass = (spine_df[fm_col] == 1).groupby(level=il).any()
        fv_pass = (spine_df[fv_col] == 1).groupby(level=il).any()
        presel  = fm_pass[fm_pass].index.intersection(fv_pass[fv_pass].index)
        return presel, fm_pass, fv_pass
    except Exception as e:
        warnings.warn(f"_get_presel: FM/FV columns not found — {e}")
        inter_idx = spine_df.groupby(level=il).first().index
        dummy     = pd.Series(False, index=inter_idx)
        return inter_idx, dummy, dummy

# =============================================================================
# Truth counting
# =============================================================================
def _build_truth_counts(spine_df):
    il    = list(range(spine_df.index.nlevels - 1))
    inter = spine_df.groupby(level=il).first()
    idx   = inter.index

    def _b(s): return s.reindex(idx, fill_value=False).astype(bool)

    nu_col   = _safe(_find_col, spine_df, "nu_id",           branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, spine_df, "current_type",    branch_must_contain=BRANCH_TRUE)
    fv_t_col = _safe(_find_col, spine_df, "is_fiducial",     branch_must_contain=BRANCH_TRUE)
    pid_col  = _safe(_find_col, spine_df, "pid",             branch_must_contain=BRANCH_TRUE)
    pri_col  = _safe(_find_col, spine_df, "is_primary",      branch_must_contain=BRANCH_TRUE)
    ke_col   = _safe(_find_col, spine_df, "ke",              branch_must_contain=BRANCH_TRUE)
    ppd_col  = _safe(_find_col, spine_df, "parent_pdg_code", branch_must_contain=BRANCH_TRUE)

    is_nu      = _b(inter[nu_col]   >= 0) if nu_col   else pd.Series(False, index=idx)
    is_cc      = _b(inter[cc_col]   == 0) if cc_col   else pd.Series(False, index=idx)
    is_fv_true = _b(inter[fv_t_col] == 1) if fv_t_col else pd.Series(False, index=idx)

    if pid_col and pri_col and ke_col:
        elec_mask = ((spine_df[pid_col] == PID_ELECTRON) &
                     (spine_df[pri_col] == 1) &
                     (spine_df[ke_col]  > ELECTRON_THRESHOLD_MEV))
        has_elec  = elec_mask.groupby(level=il).sum().reindex(idx, fill_value=0) == 1
        has_muon  = ((spine_df[pid_col] == PID_MUON) &
                     (spine_df[pri_col] == 1) &
                     (spine_df[ke_col]  > MUON_THRESHOLD_MEV)
                    ).groupby(level=il).any().reindex(idx, fill_value=False)
    else:
        has_elec = has_muon = pd.Series(False, index=idx)

    has_pi0 = ((spine_df[ppd_col].abs() == 111).groupby(level=il).any()
               .reindex(idx, fill_value=False)) if ppd_col else pd.Series(False, index=idx)

    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                        = 6
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_elec]         = 1
    cat[nu &  is_cc &  is_fv_true & has_elec]         = 0  # signal
    cat[nu &  is_cc & has_muon    & has_pi0]           = 2
    cat[nu & ~is_cc & has_pi0]                         = 3
    cat[nu &  is_cc & has_muon    & ~has_pi0]          = 4
    cat[nu & ~is_cc & ~has_pi0]                        = 5

    _, fm_pass, fv_pass = _get_presel(spine_df)
    passed_presel = fm_pass.reindex(idx, fill_value=False) & fv_pass.reindex(idx, fill_value=False)

    return dict(idx=idx, cat=cat, is_sig=(cat == 0),
                passed_fm=fm_pass.reindex(idx, fill_value=False),
                passed_presel=passed_presel)

# =============================================================================
# Public maker 1: evtdf
# =============================================================================
def make_nuecc_evtdf(f):
    """Reco+truth SPINE df, FM+FV preselected, merged with MC neutrino CV info."""
    spine_df = _get_spine_df(f)
    il       = list(range(spine_df.index.nlevels - 1))

    # Merge MC neutrino CV info (no weights — small, fast)
    try:
        mcdf    = make_mcnudf(f, include_weights=False)
        mcdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["mcnu"] + list(c)) for c in mcdf.columns])
        mct_col = _find_col(spine_df, "mct_index", branch_must_contain=BRANCH_TRUE)
        merged  = multicol_merge(
            lhs      = spine_df.reset_index(level=list(range(1, spine_df.index.nlevels))),
            rhs      = mcdf.reset_index(),
            left_on  = ["entry", mct_col],
            right_on = ["entry", mcdf.index.names[-1]],
            how      = "left",
            validate = "many_to_one",
        )
        spine_df = merged.set_index(list(spine_df.index.names))
    except Exception as e:
        warnings.warn(f"make_nuecc_evtdf: MC CV merge skipped — {e}")

    # FM+FV preselection
    try:
        presel, _, _ = _get_presel(spine_df)
        row_mask     = spine_df.index.droplevel(-1).isin(presel)
        spine_df     = spine_df[row_mask]
    except Exception as e:
        warnings.warn(f"make_nuecc_evtdf: preselection skipped — {e}")

    return spine_df

# =============================================================================
# Public maker 2: statsdf
# =============================================================================
def make_nuecc_statsdf(f):
    """One-row pre-cut denominator table."""
    empty = pd.DataFrame([{"n_inter_total":0,"n_true_signal":0,
                            "n_after_fm":0,"n_after_presel":0,
                            "n_sig_after_fm":0,"n_sig_after_presel":0,
                            **{f"cat_{c}":0 for c in range(8)}}])
    try:
        spine_df = _get_spine_df(f)
        tc       = _build_truth_counts(spine_df)
    except Exception as e:
        warnings.warn(f"make_nuecc_statsdf: failed — {e}")
        return empty

    cat = tc["cat"]
    return pd.DataFrame([{
        "n_inter_total"     : len(tc["idx"]),
        "n_true_signal"     : int(tc["is_sig"].sum()),
        "n_after_fm"        : int(tc["passed_fm"].sum()),
        "n_after_presel"    : int(tc["passed_presel"].sum()),
        "n_sig_after_fm"    : int((tc["is_sig"] & tc["passed_fm"]).sum()),
        "n_sig_after_presel": int((tc["is_sig"] & tc["passed_presel"]).sum()),
        **{f"cat_{c}": int((cat == c).sum()) for c in range(8)},
    }])

# =============================================================================
# Public maker 3: wgtdf — prefiltered weights
# =============================================================================
def make_nuecc_wgtdf(f):
    """
    BNB+GENIE universe weights for FM+FV preselected interactions only.

    Strategy (inspired by numuCC pandora approach):
      1. Get preselected (entry, mct_index) pairs from cached SPINE df
      2. Load FULL mc ind for bnbsyst/geniesyst (they need aligned ind)
      3. Filter output rows to preselected pairs → tiny output df
    Memory: BNB ~3 GB, GENIE ~8 GB peak during step 2, output ~50 MB
    """
    from makedf import bnbsyst, geniesyst

    # ── Step 1: preselected (entry, mct_index) pairs ─────────────────────
    spine_df = _get_spine_df(f)
    il       = list(range(spine_df.index.nlevels - 1))

    presel, _, _ = _get_presel(spine_df)
    mct_col      = _safe(_find_col, spine_df, "mct_index", branch_must_contain=BRANCH_TRUE)
    if mct_col is None:
        warnings.warn("make_nuecc_wgtdf: mct_index not found")
        return pd.DataFrame()

    inter_df     = spine_df.groupby(level=il).first()
    inter_presel = inter_df.loc[inter_df.index.isin(presel), mct_col].dropna()
    entry_vals   = inter_presel.index.get_level_values(0)
    presel_pairs = set(zip(entry_vals, inter_presel.astype(int).values))

    # ── Step 2: load full mcdf + compute weights ──────────────────────────
    mcdf        = make_mcnudf(f, include_weights=False)
    mcdf["ind"] = mcdf.index.get_level_values(-1)   # Series — required by bnbsyst
    full_ind    = mcdf["ind"]

    print(f"  wgtdf: {len(presel_pairs)} preselected pairs "
          f"(of {len(mcdf)} total MC nu)")

    wgtdf = mcdf.copy()

    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=100, slim=True)
        if not bnb_wgt.empty:
            wgtdf = multicol_concat(wgtdf, bnb_wgt)
            print(f"  BNB: {bnb_wgt.shape[1]} columns")
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: BNB failed — {e}")

    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=100, slim=True)
        if not genie_wgt.empty:
            wgtdf = multicol_concat(wgtdf, genie_wgt)
            print(f"  GENIE: {genie_wgt.shape[1]} columns")
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: GENIE failed — {e}")

    # ── Step 3: filter to preselected (entry, mct_index) pairs ───────────
    e_vals = wgtdf.index.get_level_values(0)
    m_vals = wgtdf.index.get_level_values(-1)
    mask   = np.array([(e, m) in presel_pairs for e, m in zip(e_vals, m_vals)])
    out    = wgtdf[mask].copy()

    print(f"  wgtdf output: {len(out)} rows (was {len(wgtdf)} before filter)")
    del wgtdf, mcdf
    return out