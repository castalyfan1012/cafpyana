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
make_nuecc_wgtdf(f)          — BNB + GENIE universe weights (preselected nus only)

Output HDF5 keys (via nueCC_mc.py config)
------------------------------------------
    evt_0         particle-level df, FM+FV preselected, MC-truth merged
    truth_info_0  per-interaction metadata, ALL interactions pre-FM/FV cut
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
    |x| ∈ (5, 190) cm
    z_region1 = (10 < z < 250) & (-190 < y < 190)
    z_region2 = (250 < z < 450) & (-190 < y < 100) & (x < 0)   [TPC 0]
    z_region3 = (250 < z < 450) & (-190 < y < 190) & (x > 0)   [TPC 1]

valid_flashmatch (matches C++ definition)
-----------------------------------------
    flash_times.size() > 0  AND  is_flash_matched == 1  AND  NOT isnan(flash_times[0])
    Falls back to  is_flash_matched == 1  AND  flash_total_pe > 0
    if flash_times cannot be loaded from the file.
"""

import gc
import ctypes
import warnings
import numpy as np
import pandas as pd

from makedf.makedf import make_all_spine_df, make_mcnudf
from pyanalib.pandas_helpers import multicol_merge, multicol_concat

# ── Optional imports for flash_times (ragged branch, loaded separately) ──────
# flash_times is a vector branch (rec.dlp.flash_times) that is NOT part of the
# flat spine_df returned by make_all_spine_df.  We load it independently in
# _get_truth_info and pass it into _build_truth_info for the full FM cut.
try:
    from makedf.makedf import loadbranches as _loadbranches
except ImportError:
    try:
        from makedf.util import loadbranches as _loadbranches
    except ImportError:
        _loadbranches = None

try:
    from makedf.branches import spineint_flashtimes_branches as _SPINE_FT_BRANCHES
except ImportError:
    _SPINE_FT_BRANCHES = None

# ── Constants — must match nue_selection.py exactly ──────────────────────────
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 0.0
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


def _force_free():
    """
    Force CPython + glibc to return freed pages to the OS.

    gc.collect() clears Python-level circular refs but CPython's allocator
    holds freed pages in its own pool so the process RSS stays high even after
    del + gc.collect(). malloc_trim(0) tells glibc to release those pages
    immediately, which is critical after clearing the ~15-20 GB spine_df cache
    before the (much smaller) weight calls.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Linux-only; safe to ignore elsewhere


# ── Fiducial volume ───────────────────────────────────────────────────────────

def _fiducial_cut_tmp(x, y, z):
    abs_x     = x.abs()
    x_region  = (abs_x > 5) & (abs_x < 190)
    z_region1 = (z > 10)  & (z < 250) & (y > -190) & (y < 190)
    z_region2 = (z > 250) & (z < 450) & (y > -190) & (y < 100) & (x < 0)
    z_region3 = (z > 250) & (z < 450) & (y > -190) & (y < 190) & (x > 0)
    return x_region & (z_region1 | z_region2 | z_region3)


# ── Column helpers ────────────────────────────────────────────────────────────

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
    raise KeyError(f"column suffix={suffix!r} not found")


def _find_col_ends(df, *suffix_parts, branch_must_contain=None,
                   branch_must_not_contain=None):
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

def _build_truth_info(spine_df, flash_times_s=None):
    """
    Parameters
    ----------
    spine_df      : merged SPINE reco+truth particle-level DataFrame
    flash_times_s : optional Series with 3-level MultiIndex
                    (entry, dlp_inter_idx, flash_entry_idx) containing
                    rec.dlp.flash_times values.  When supplied the full C++
                    flash-match condition is applied:
                        flash_times.size()>0 && is_flash_matched==1
                        && !isnan(flash_times[0])
                    When None, falls back to is_flash_matched==1 &&
                    flash_total_pe>0.
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
    cat[nu &  is_cc &  is_fv_true & has_true_electron]      = 0
    cat[nu &  is_cc & has_true_muon & has_pi0]               = 2
    cat[nu & ~is_cc & has_pi0]                               = 3
    cat[nu &  is_cc & has_true_muon & ~has_pi0]              = 4
    cat[nu & ~is_cc & ~has_pi0]                              = 5

    # ── Flash-match cut ───────────────────────────────────────────────────────
    if fm_col is not None:
        fm_pass = (spine_df[fm_col] == 1)

        if flash_times_s is not None:
            # Full C++ condition:
            #   flash_times.size()>0 && is_flash_matched==1 && !isnan(flash_times[0])
            #
            # flash_times_s has a 3-level MultiIndex:
            #   (entry, dlp_inter_idx, flash_entry_idx)
            # Grouping by the first (nlevels-1) levels collapses to interaction
            # level.  .first() returns NaN for any interaction absent from the
            # Series (i.e. empty flash_times vector), so notna() correctly
            # captures both size()>0 and !isnan(flash_times[0]) in one step.
            ft_il        = list(range(flash_times_s.index.nlevels - 1))
            first_ft     = flash_times_s.groupby(level=ft_il).first()
            has_valid_ft = first_ft.notna().reindex(idx, fill_value=False).astype(bool)

            fm_inter  = fm_pass.groupby(level=il).any().reindex(idx, fill_value=False)
            passed_fm = fm_inter & has_valid_ft
        else:
            # Fallback proxy: is_flash_matched==1 AND flash_total_pe>0
            # (used when flash_times branch could not be loaded)
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
        # ── Load flash_times (ragged branch, not in spine_df) ────────────────
        # rec.dlp.flash_times is a vector branch that make_all_spine_df does
        # not include.  We load it here and pass into _build_truth_info so the
        # full C++ valid_flashmatch condition can be applied.
        flash_times_s = None
        if _loadbranches is not None and _SPINE_FT_BRANCHES is not None:
            try:
                raw_ft = _loadbranches(f["recTree"], _SPINE_FT_BRANCHES)
                # raw_ft has a 3-level MultiIndex:
                #   (entry, dlp_interaction_idx, flash_entry_idx)
                # Extract as a plain Series; the first two index levels align
                # with the interaction-level index inside _build_truth_info.
                ft_col        = raw_ft.columns[0]
                flash_times_s = raw_ft[ft_col]
            except Exception as e:
                warnings.warn(
                    f"_get_truth_info: flash_times load failed, "
                    f"falling back to flash_total_pe proxy — {e}"
                )

        _truth_info_cache[fname] = _build_truth_info(_get_spine_df(f), flash_times_s)
    return _truth_info_cache[fname]


# ── Public maker 1: evtdf ────────────────────────────────────────────────────

def make_nuecc_evtdf(f):
    """
    Reco+truth SPINE particle-level df, FM+FV preselected, merged with MC CV.
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
    """
    return _get_truth_info(f)


def make_nuecc_statsdf(f):
    """Backward-compatible alias for make_nuecc_truth_info_df."""
    return make_nuecc_truth_info_df(f)


# ── Public maker 3: wgtdf ────────────────────────────────────────────────────
#
# Root cause of the OOM (now fixed):
# ─────────────────────────────────
# The original getsyst.py loaded rec.mc.nu.wgt.univ for ALL nus in the file
# at once via:
#
#   wgts = ak.to_dataframe(f["recTree"]['rec.mc.nu.wgt.univ'].arrays(...))
#
# For 13,297 nus × ~5,000 total universe weights = 66 million rows ≈ 5-10 GB.
# Added to spine_df RSS residual (~15 GB not yet released by glibc), this
# pushes the grid job over 29 GB.
#
# evtdf avoids this because it never calls getsyst.  That's the ONLY
# difference between evtdf (fits 29 GB) and wgtdf (OOM).
#
# Fix (two parts):
# ──────────────────
# 1. getsyst.py: replaced with Lynn's chunked version that reads weight data
#    in 10 MB pieces. Peak from weight loading: ~50 MB instead of 5-10 GB.
#    Crucially, it uses index-aligned `.loc` updates (not numpy broadcasting),
#    so it correctly handles a SUBSET of nu indices without shape errors.
#
# 2. make_nuecc_wgtdf: now passes presel_ind (only the ~3,700 preselected nus)
#    instead of full_ind (all 13,297 nus). Same approach as Lynn's pipeline.
#    Output is already indexed correctly — no post-call slicing needed.
#
# Combined effect on memory:
#   spine_df:    ~15-20 GB (same as evtdf — freed + malloc_trim before wgts)
#   weight data: ~50 MB   (was 5-10 GB — chunked getsyst)
#   Peak:        same as evtdf ← fits within 29 GB

def make_nuecc_wgtdf(f):
    """
    BNB + GENIE universe weights for FM+FV preselected interactions only.

    Requires: getsyst.py = Lynn's chunked version (not the original).
    Output format unchanged: mcdf-indexed, preselected rows, BNB+GENIE cols.
    """
    from makedf import bnbsyst, geniesyst

    # ── Step 1: presel (entry, mct_index) pairs from truth_info ─────────────
    truth_info = _get_truth_info(f)
    if truth_info.empty:
        return pd.DataFrame()

    presel_mct = truth_info.loc[truth_info["passed_presel"], "mct_index"]
    presel_mct = presel_mct[presel_mct >= 0]
    if presel_mct.empty:
        warnings.warn("make_nuecc_wgtdf: no preselected interactions in this file")
        return pd.DataFrame()

    pair_df = pd.DataFrame({
        "entry": np.asarray(presel_mct.index.get_level_values(0)),
        "mct":   presel_mct.astype(np.int64).values,
    }).drop_duplicates().reset_index(drop=True)

    # ── Step 2: free spine_df cache + release OS pages ───────────────────────
    # spine_df is ~15-20 GB. Clearing the cache + malloc_trim drops the RSS
    # back to ~2-3 GB before the weight calls, matching the evtdf budget.
    # Safe in the wgtdf-only config since no other maker reuses spine_df here.
    _spine_cache.clear()
    _truth_info_cache.clear()
    del truth_info, presel_mct
    _force_free()

    # ── Step 3: build presel_ind ─────────────────────────────────────────────
    # presel_ind is a Series indexed by the preselected subset of mcdf.index,
    # with values = per-event nu integer index (the 'inu' used inside getsyst).
    # This mirrors Lynn's nu_indices in _add_weights_to_nueccdf.
    mcdf = make_mcnudf(f, include_weights=False)

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
    print(f"  wgtdf: {n_presel} preselected MC nu rows → passing to syst functions")

    if n_presel == 0:
        warnings.warn("make_nuecc_wgtdf: no mcdf rows matched preselected pairs")
        return pd.DataFrame()

    # presel_ind: index = presel_mcdf_index, values = per-event nu int index
    presel_ind = pd.Series(
        np.asarray(presel_mcdf_index.get_level_values(-1), dtype=np.int64),
        index=presel_mcdf_index,
    )

    # ── Step 4: BNB and GENIE weights for preselected nus only ───────────────
    # Lynn's getsyst reads weight data in 10 MB chunks and uses .loc updates,
    # so it handles subset input correctly and uses negligible peak memory.
    # Output is already indexed by presel_mcdf_index — no slicing needed.
    out = None

    try:
        bnb_wgt = bnbsyst.bnbsyst(f, presel_ind, multisim_nuniv=100, slim=True)
        if bnb_wgt is not None and not bnb_wgt.empty:
            print(f"  BNB: {bnb_wgt.shape[1]} cols, {len(bnb_wgt)} rows  "
                  f"({bnb_wgt.memory_usage(deep=True).sum()/1e9:.3f} GB)")
            out = bnb_wgt
        elif bnb_wgt is not None:
            del bnb_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: BNB failed — {e}")
    _force_free()

    try:
        genie_wgt = geniesyst.geniesyst(f, presel_ind, multisim_nuniv=100, slim=True)
        if genie_wgt is not None and not genie_wgt.empty:
            print(f"  GENIE: {genie_wgt.shape[1]} cols, {len(genie_wgt)} rows  "
                  f"({genie_wgt.memory_usage(deep=True).sum()/1e9:.3f} GB)")
            if out is None:
                out = genie_wgt
            else:
                out = multicol_concat(out, genie_wgt)
                del genie_wgt
        elif genie_wgt is not None:
            del genie_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: GENIE failed — {e}")
    _force_free()

    from makedf.getsyst import getsyst

    EXTRA_XSEC_SYSTS = [
        # MINERvA 2p2h (indices 82-86)
        "MINERvAE2p2h_SBN_v1_E2p2h_A_nu",
        "MINERvAE2p2h_SBN_v1_E2p2h_B_nu",
        "MINERvAE2p2h_SBN_v1_E2p2h_A_nubar",
        "MINERvAE2p2h_SBN_v1_E2p2h_B_nubar",
        "MINERvAq0q3Weighting_SBN_v1_Mnv2p2hGaussEnhancement",
        # MiscInteractionSysts (indices 87-91)
        "MiscInteractionSysts_SBN_v1_C12ToAr40_2p2hScaling_nu",
        "MiscInteractionSysts_SBN_v1_C12ToAr40_2p2hScaling_nubar",
        "MiscInteractionSysts_SBN_v1_nuenuebar_xsec_ratio",
        "MiscInteractionSysts_SBN_v1_nuenumu_xsec_ratio",
        "MiscInteractionSysts_SBN_v1_SPPLowQ2Suppression",
        # NOvAStyleNonResPionNorm (indices 92-114)
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_n_CC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_n_CC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_p_CC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_p_CC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_np_CC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_n_NC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_n_NC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_n_NC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_p_NC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_p_NC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nu_p_NC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_CC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_CC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_CC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_CC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_CC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_CC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_NC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_NC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_n_NC_3Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_NC_1Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_NC_2Pi",
        "NOvAStyleNonResPionNorm_SBN_v1_NR_nubar_p_NC_3Pi",
    ]

    try:
        extra_wgt = getsyst(
            f, EXTRA_XSEC_SYSTS, presel_ind,
            multisim_nuniv=100, slim=True, slimname="extra_xsec"
        )
        if extra_wgt is not None and not extra_wgt.empty:
            print(f"  Extra xsec: {extra_wgt.shape[1]} cols, {len(extra_wgt)} rows  "
                  f"({extra_wgt.memory_usage(deep=True).sum()/1e9:.3f} GB)")
            if out is None:
                out = extra_wgt
            else:
                out = multicol_concat(out, extra_wgt)
                del extra_wgt
        elif extra_wgt is not None:
            del extra_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: extra xsec failed — {e}")
    _force_free()


    del presel_ind, presel_mcdf_index
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()

    print(f"  wgtdf output: {len(out)} rows × {out.shape[1]} cols  "
          f"({out.memory_usage(deep=True).sum()/1e9:.3f} GB)")
    return out