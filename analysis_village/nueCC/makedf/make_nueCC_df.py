#!/usr/bin/env python3
"""
make_nueCC_df.py
----------------
SPINE DLP-based nueCC inclusive MC DataFrame makers for cafpyana.

Public functions
----------------
make_nuecc_evtdf(f)          — reco+truth particle df, FM+FV preselected
make_nuecc_truth_info_df(f)  — per-interaction truth+presel metadata (ALL interactions)
make_nuecc_statsdf(f)        — alias for make_nuecc_truth_info_df
make_nuecc_wgtdf(f)          — BNB + GENIE universe weights for preselected
                               interactions, computed memory-efficiently.

Output HDF5 keys (via nueCC_mc.py config)
------------------------------------------
    evt_0         particle-level df, FM+FV preselected, MC-truth merged
    truth_info_0  per-interaction metadata, ALL interactions pre-FM/FV cut
                  columns: truth_cat, is_true_signal, is_nu, is_cc, is_fv_true,
                           has_true_electron, has_true_muon, has_pi0,
                           passed_fm, passed_fv_reco, passed_presel,
                           true_leading_e_ke, true_leading_e_costheta,
                           true_leading_e_p, mct_index
    hdr_0         run/subrun/event header
    pot_0         BNB POT per file

Truth categories (match nue_selection.py exactly)
--------------------------------------------------
    0  nueCC FV          <- signal
    1  nueCC out FV
    2  numuCC + pi0
    3  NC pi0
    4  other numuCC
    5  other NC
    6  cosmic / non-neutrino
    7  neutrino catch-all

Fiducial volume (fiducial_cut_tmp — matches C++ definition)
------------------------------------------------------------
    10 < |x| < 190  cm
    -190 < y < 190  cm   if  10 < z < 250 cm
    -190 < y < 100  cm   if 250 < z < 450 cm
    Corresponds to 57 m³ (out of 80 m³ total active volume).

valid_flashmatch proxy (matches C++ definition)
------------------------------------------------
    is_flash_matched == 1  AND  flash_total_pe > 0
"""

import gc
import warnings
import numpy as np
import pandas as pd

from makedf.makedf import make_all_spine_df, make_mcnudf
from pyanalib.pandas_helpers import multicol_merge, multicol_concat

# ── Constants — must match nue_selection.py exactly ──────────────────────────
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 75.0
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2

# ── Module-level caches (one SPINE load shared across all DFS makers per file)
_spine_cache      = {}
_truth_info_cache = {}


def _get_spine_df(f):
    fname = str(f)
    if fname not in _spine_cache:
        _spine_cache.clear()
        _truth_info_cache.clear()
        gc.collect()
        _spine_cache[fname] = make_all_spine_df(f)
    return _spine_cache[fname]


# ── Fiducial volume ───────────────────────────────────────────────────────────

def _fiducial_cut_tmp(x, y, z):
    """
    SBND FV: fiducial_cut_tmp from SPINE analysis framework.

    C++ equivalent:
        (abs(vertex[0]) > 10) && (abs(vertex[0]) < 190) &&
        (vertex[2] > 10) && (vertex[2] < 450) &&
        (
          ((vertex[2] > 250) && (vertex[1] > -190) && (vertex[1] < 100)) ||
          ((vertex[2] < 250) && (abs(vertex[1]) < 190))
        )
    """
    abs_x = x.abs()
    in_x  = (abs_x > 10) & (abs_x < 190)
    in_z  = (z > 10) & (z < 450)
    in_y  = (
        ((z <= 250) & (y > -190) & (y < 190)) |
        ((z >  250) & (y > -190) & (y < 100))
    )
    return in_x & in_z & in_y


# ── Column helpers ────────────────────────────────────────────────────────────

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
    raise KeyError(f"column suffix={suffix!r} not found")


def _find_col_ends(df, *suffix_parts, branch_must_contain=None,
                   branch_must_not_contain=None):
    """Return the first column whose last N non-empty parts equal suffix_parts."""
    n, target = len(suffix_parts), tuple(suffix_parts)
    for col in df.columns:
        parts = tuple(p for p in (col if isinstance(col, tuple) else (col,)) if p != "")
        if len(parts) < n or parts[-n:] != target:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        if branch_must_not_contain and any(branch_must_not_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(f"column ending with {suffix_parts!r} not found")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (KeyError, StopIteration):
        return None


# ── Core: build per-interaction truth + reco-presel metadata ─────────────────

def _build_truth_info(spine_df):
    """
    One row per interaction covering ALL interactions (pre-FM/FV cut).

    Truth flags computed on the unfiltered df (never biased by reco decisions).
    Reco presel flags let the notebook reconstruct cut-stage counts without
    re-scanning the particle-level df.

    FV: uses fiducial_cut_tmp geometric definition (not is_fiducial).
    FM: is_flash_matched==1 AND flash_total_pe>0 (valid_flashmatch proxy).
    """
    il    = list(range(spine_df.index.nlevels - 1))
    inter = spine_df.groupby(level=il).first()
    idx   = inter.index

    nu_col   = _safe(_find_col, spine_df, "nu_id",           branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, spine_df, "current_type",    branch_must_contain=BRANCH_TRUE)
    pid_col  = _safe(_find_col, spine_df, "pid",             branch_must_contain=BRANCH_TRUE)
    pri_col  = _safe(_find_col, spine_df, "is_primary",      branch_must_contain=BRANCH_TRUE)
    ke_col   = _safe(_find_col, spine_df, "ke",              branch_must_contain=BRANCH_TRUE)
    ppd_col  = _safe(_find_col, spine_df, "parent_pdg_code", branch_must_contain=BRANCH_TRUE)
    mct_col  = _safe(_find_col, spine_df, "mct_index",       branch_must_contain=BRANCH_TRUE)
    fm_col   = _safe(_find_col, spine_df, "is_flash_matched",branch_must_not_contain=BRANCH_TRUE)
    pe_col   = _safe(_find_col, spine_df, "flash_total_pe",  branch_must_not_contain=BRANCH_TRUE)

    tvx_col  = _safe(_find_col_ends, spine_df, "vertex", "x", branch_must_contain=BRANCH_TRUE)
    tvy_col  = _safe(_find_col_ends, spine_df, "vertex", "y", branch_must_contain=BRANCH_TRUE)
    tvz_col  = _safe(_find_col_ends, spine_df, "vertex", "z", branch_must_contain=BRANCH_TRUE)
    rvx_col  = _safe(_find_col_ends, spine_df, "vertex", "x", branch_must_not_contain=BRANCH_TRUE)
    rvy_col  = _safe(_find_col_ends, spine_df, "vertex", "y", branch_must_not_contain=BRANCH_TRUE)
    rvz_col  = _safe(_find_col_ends, spine_df, "vertex", "z", branch_must_not_contain=BRANCH_TRUE)

    fv_t_col = _safe(_find_col, spine_df, "is_fiducial", branch_must_contain=BRANCH_TRUE)
    fv_r_col = _safe(_find_col, spine_df, "is_fiducial", branch_must_not_contain=BRANCH_TRUE)

    required = [nu_col, cc_col, pid_col, pri_col, ke_col, ppd_col]
    if any(c is None for c in required):
        missing = [name for name, col in zip(
            ["nu_id", "current_type", "pid", "is_primary", "ke", "parent_pdg_code"],
            required,
        ) if col is None]
        warnings.warn(f"_build_truth_info: missing columns {missing}")
        return pd.DataFrame()

    def _b(s):
        return s.reindex(idx, fill_value=False).astype(bool)

    is_nu = _b(inter[nu_col] >= 0)
    is_cc = _b(inter[cc_col] == 0)

    if tvx_col and tvy_col and tvz_col:
        is_fv_true = _fiducial_cut_tmp(
            inter[tvx_col], inter[tvy_col], inter[tvz_col]
        ).reindex(idx, fill_value=False).astype(bool)
    elif fv_t_col:
        warnings.warn("_build_truth_info: truth vertex coords missing, using is_fiducial")
        is_fv_true = _b(inter[fv_t_col] == 1)
    else:
        is_fv_true = pd.Series(False, index=idx)

    _elec_mask = (
        (spine_df[pid_col] == PID_ELECTRON) &
        (spine_df[pri_col] == 1) &
        (spine_df[ke_col]  > ELECTRON_THRESHOLD_MEV)
    )
    has_true_electron = (
        _elec_mask.groupby(level=il).sum().reindex(idx, fill_value=0) == 1
    )
    has_true_muon = (
        (spine_df[pid_col] == PID_MUON) &
        (spine_df[pri_col] == 1) &
        (spine_df[ke_col]  > MUON_THRESHOLD_MEV)
    ).groupby(level=il).any().reindex(idx, fill_value=False)

    has_pi0 = (
        (spine_df[ppd_col].abs() == 111)
        .groupby(level=il).any()
        .reindex(idx, fill_value=False)
    )

    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                              = 6
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_true_electron]      = 1
    cat[nu &  is_cc &  is_fv_true & has_true_electron]      = 0   # signal
    cat[nu &  is_cc & has_true_muon & has_pi0]               = 2
    cat[nu & ~is_cc & has_pi0]                               = 3
    cat[nu &  is_cc & has_true_muon & ~has_pi0]              = 4
    cat[nu & ~is_cc & ~has_pi0]                              = 5

    if fm_col is not None:
        fm_pass = (spine_df[fm_col] == 1)
        if pe_col is not None:
            fm_pass = fm_pass & (spine_df[pe_col] > 0)
        passed_fm = fm_pass.groupby(level=il).any().reindex(idx, fill_value=False)
    else:
        passed_fm = pd.Series(False, index=idx)

    if rvx_col and rvy_col and rvz_col:
        passed_fv_reco = _fiducial_cut_tmp(
            inter[rvx_col], inter[rvy_col], inter[rvz_col]
        ).reindex(idx, fill_value=False).astype(bool)
    elif fv_r_col:
        warnings.warn("_build_truth_info: reco vertex coords missing, using is_fiducial")
        passed_fv_reco = (
            (spine_df[fv_r_col] == 1)
            .groupby(level=il).any()
            .reindex(idx, fill_value=False)
        )
    else:
        passed_fv_reco = pd.Series(False, index=idx)

    passed_presel = passed_fm & passed_fv_reco

    mct_index_vals = (
        inter[mct_col].fillna(-1).astype(np.int32)
        if mct_col is not None
        else pd.Series(-1, index=idx, dtype=np.int32)
    )

    true_leading_e_ke       = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_costheta = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_p        = pd.Series(np.nan, index=idx, dtype=np.float32)

    try:
        momx_col = _find_col_ends(spine_df, "particles", "momentum", "x",
                                  branch_must_contain=BRANCH_TRUE)
        momy_col = _find_col_ends(spine_df, "particles", "momentum", "y",
                                  branch_must_contain=BRANCH_TRUE)
        momz_col = _find_col_ends(spine_df, "particles", "momentum", "z",
                                  branch_must_contain=BRANCH_TRUE)
        elec_ke_for_max = spine_df[ke_col].where(_elec_mask, -np.inf)
        lead_4idx       = elec_ke_for_max.groupby(level=il).idxmax().dropna()
        if len(lead_4idx) > 0:
            lead_mi      = pd.MultiIndex.from_tuples(
                lead_4idx.values, names=spine_df.index.names
            )
            leading_rows = spine_df.loc[
                lead_mi, [ke_col, momx_col, momy_col, momz_col]
            ].copy()
            leading_rows.index = lead_4idx.index
            px   = leading_rows[momx_col]
            py   = leading_rows[momy_col]
            pz   = leading_rows[momz_col]
            pmag = np.sqrt(px**2 + py**2 + pz**2).replace(0, np.nan)
            true_leading_e_ke       = leading_rows[ke_col].reindex(idx).astype(np.float32)
            true_leading_e_costheta = (pz / pmag).reindex(idx).astype(np.float32)
            true_leading_e_p        = pmag.reindex(idx).astype(np.float32)
    except Exception:
        pass

    return pd.DataFrame({
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
        "true_leading_e_ke"      : true_leading_e_ke,
        "true_leading_e_costheta": true_leading_e_costheta,
        "true_leading_e_p"       : true_leading_e_p,
        "mct_index"              : mct_index_vals,
    }, index=idx).astype({
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
        "mct_index"         : np.int32,
    })


def _get_truth_info(f):
    fname = str(f)
    if fname not in _truth_info_cache:
        _truth_info_cache[fname] = _build_truth_info(_get_spine_df(f))
    return _truth_info_cache[fname]


# ── Public maker 1: evtdf ────────────────────────────────────────────────────

def make_nuecc_evtdf(f):
    """
    Reco+truth SPINE particle-level df, FM+FV preselected, merged with MC CV.
    Weights excluded here — handled by make_nuecc_wgtdf in a separate config.
    """
    spine_df   = _get_spine_df(f)
    truth_info = _get_truth_info(f)

    try:
        mcdf    = make_mcnudf(f, include_weights=False)
        mcdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["mcnu"] + list(c)) for c in mcdf.columns]
        )
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
        del merged, mcdf
    except Exception as e:
        warnings.warn(f"make_nuecc_evtdf: MC CV merge skipped — {e}")

    if not truth_info.empty:
        presel_idx = truth_info.index[truth_info["passed_presel"]]
        spine_df   = spine_df[spine_df.index.droplevel(-1).isin(presel_idx)]

    return spine_df


# ── Public maker 2: truth_info_df ────────────────────────────────────────────

def make_nuecc_truth_info_df(f):
    """
    Per-interaction truth + reco-preselection metadata covering ALL interactions
    before any FM/FV cut.  Loaded by nue_selection.ipynb as truth_info_0.

    3-level index: [__ntuple, entry, rec.dlp..index]
    15 columns (14 truth/presel flags + mct_index).
    """
    return _get_truth_info(f)


def make_nuecc_statsdf(f):
    """Backward-compatible alias for make_nuecc_truth_info_df."""
    return make_nuecc_truth_info_df(f)


# ── Public maker 3: wgtdf — memory-efficient weight processing ───────────────
#
# === Why this looks the way it does ===
#
# Lynn's cafpyana_pandora code (analysis_village/nuecc/makedf/make_nueccdf.py)
# passes a SUBSET of nu indices straight into geniesyst()/bnbsyst() and it
# Just Works — her version of those modules supports preselected input.
#
# This SPINE branch has DIFFERENT versions of bnbsyst/geniesyst that internally
# load full-file weight arrays (shape = total MC nu in file) and align them
# positionally with the input Series. If we pass a subset, we get:
#
#     operands could not be broadcast together with shapes (3696,) (13297,)
#                                                          ↑       ↑
#                                                          presel  full file
#
# So Lynn's exact "filter upstream of bnbsyst" trick is not available here
# without modifying the syst modules. Instead, we get most of Lynn's memory
# win through a different route:
#
#   1.  truth_info is already built and cached → we know which (entry, mct)
#       pairs survive presel without re-touching spine_df.
#   2.  spine_df cache is DROPPED before any weight call. spine_df is the
#       single biggest user of memory in this pipeline (~10-25 GB depending
#       on file size), so freeing it before weights is the largest single
#       memory win.
#   3.  Pass full_ind to bnbsyst (mandatory — see broadcast bug above), then
#       slice the output down to preselected rows IMMEDIATELY and delete the
#       full frame BEFORE pulling geniesyst. The two full syst frames never
#       coexist; only one is alive at any moment, and only for a moment.
#   4.  Final concat is between two NARROW (preselected-only) frames.
#
# Output format is byte-identical to the previous implementation:
#   - index: subset of make_mcnudf's MultiIndex restricted to preselected rows
#   - columns: BNB universes + GENIE universes (multisim_nuniv=100 each)

def make_nuecc_wgtdf(f):
    """
    BNB + GENIE universe weights for FM+FV preselected interactions only.

    Memory profile (per file, with this pipeline):
        peak  ≈  max( bnb_full_frame , genie_full_frame )  +  small constants
              ≈  ~3-6 GB for typical SBND MC files
    (compare to ~25-30 GB in the previous full-pipeline implementation)

    Output format UNCHANGED vs. previous version — no notebook changes needed.
    """
    from makedf import bnbsyst, geniesyst

    truth_info = _get_truth_info(f)
    if truth_info.empty:
        return pd.DataFrame()

    # ── Step 1: presel (entry, mct_index) pairs from truth_info ─────────────
    presel_mct = truth_info.loc[truth_info["passed_presel"], "mct_index"]
    presel_mct = presel_mct[presel_mct >= 0]
    if presel_mct.empty:
        warnings.warn("make_nuecc_wgtdf: no preselected interactions in this file")
        return pd.DataFrame()

    pair_df = pd.DataFrame({
        "entry": np.asarray(presel_mct.index.get_level_values(0)),
        "mct":   presel_mct.astype(np.int64).values,
    }).drop_duplicates().reset_index(drop=True)

    # ── Step 2: free spine_df cache BEFORE weight pulls ─────────────────────
    # spine_df is the single biggest memory user (~10-25 GB). Dropping it
    # here is the dominant memory win. Safe ONLY when this maker runs in a
    # DFS list without make_nuecc_evtdf / make_nuecc_truth_info_df (since
    # those would reuse the cache). The shipped nueCC_mc_weights.py config
    # satisfies this. If you combine wgtdf with evtdf in a single config,
    # comment out the next two lines.
    _spine_cache.clear()
    gc.collect()

    # ── Step 3: load mcdf (CV only — cheap) to build full_ind ───────────────
    # full_ind MUST cover every nu in the file for this fork of
    # bnbsyst/geniesyst (see the broadcast bug discussion above). We also
    # use mcdf.index to identify the preselected sub-index used for slicing
    # weights down after each syst call.
    mcdf = make_mcnudf(f, include_weights=False)

    full_ind = pd.Series(
        np.asarray(mcdf.index.get_level_values(-1), dtype=np.int64),
        index=mcdf.index,
    )

    mcdf_pair_idx = pd.MultiIndex.from_arrays(
        [mcdf.index.get_level_values(0),
         np.asarray(mcdf.index.get_level_values(-1), dtype=np.int64)],
        names=["entry", "mct"],
    )
    presel_pair_idx = pd.MultiIndex.from_arrays(
        [pair_df["entry"].values, pair_df["mct"].values],
        names=["entry", "mct"],
    )
    keep_mask = mcdf_pair_idx.isin(presel_pair_idx)
    presel_mcdf_index = mcdf.index[keep_mask]

    del mcdf, mcdf_pair_idx, presel_pair_idx, keep_mask, pair_df
    gc.collect()

    n_presel = len(presel_mcdf_index)
    n_total  = len(full_ind)
    print(f"  wgtdf: {n_presel} preselected / {n_total} total MC nu rows")

    if n_presel == 0:
        warnings.warn("make_nuecc_wgtdf: no mcdf rows matched preselected pairs")
        del full_ind
        gc.collect()
        return pd.DataFrame()

    # ── Step 4: BNB → slice → free, then GENIE → slice → free ───────────────
    # Each full frame is short-lived. The .copy() on the sliced result forces
    # a real allocation so the underlying full-frame buffer is truly released
    # by del (no view holding the parent array alive).
    out = None

    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=100, slim=True)
        if bnb_wgt is not None and not bnb_wgt.empty:
            print(f"  BNB full: {bnb_wgt.shape[1]} cols, {len(bnb_wgt)} rows  "
                  f"({bnb_wgt.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
            bnb_presel = (
                bnb_wgt
                .loc[bnb_wgt.index.intersection(presel_mcdf_index)]
                .copy()
            )
            del bnb_wgt
            gc.collect()
            print(f"  BNB presel: {bnb_presel.shape[1]} cols, {len(bnb_presel)} rows  "
                  f"({bnb_presel.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
            out = bnb_presel
        elif bnb_wgt is not None:
            del bnb_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: BNB failed — {e}")
    gc.collect()

    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=100, slim=True)
        if genie_wgt is not None and not genie_wgt.empty:
            print(f"  GENIE full: {genie_wgt.shape[1]} cols, {len(genie_wgt)} rows  "
                  f"({genie_wgt.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
            genie_presel = (
                genie_wgt
                .loc[genie_wgt.index.intersection(presel_mcdf_index)]
                .copy()
            )
            del genie_wgt
            gc.collect()
            print(f"  GENIE presel: {genie_presel.shape[1]} cols, {len(genie_presel)} rows  "
                  f"({genie_presel.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
            if out is None:
                out = genie_presel
            else:
                out = multicol_concat(out, genie_presel)
                del genie_presel
        elif genie_wgt is not None:
            del genie_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: GENIE failed — {e}")
    gc.collect()

    del full_ind, presel_mcdf_index
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()

    print(f"  wgtdf output: {len(out)} rows × {out.shape[1]} cols  "
          f"({out.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    return out