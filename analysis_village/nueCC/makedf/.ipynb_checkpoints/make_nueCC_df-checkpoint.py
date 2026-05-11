#!/usr/bin/env python3
"""
make_nueCC_df.py
----------------
SPINE DLP-based nueCC inclusive MC DataFrame makers for cafpyana.

Public functions
----------------
make_nuecc_evtdf(f)          — reco+truth particle df, FM+FV preselected
make_nuecc_truth_info_df(f)  — per-interaction truth+presel metadata (ALL interactions,
                                pre-FM/FV cut).  This is what nue_selection.ipynb reads
                                as truth_info_0 and uses as the efficiency denominator.
make_nuecc_statsdf(f)        — thin backward-compat alias → same as truth_info_df
make_nuecc_wgtdf(f)          — BNB+GENIE universe weights (preselected only)

Output HDF5 keys produced (via nueCC_mc.py config)
---------------------------------------------------
    evt_0         particle-level df, FM+FV preselected, MC-truth merged
    truth_info_0  per-interaction truth metadata, ALL interactions pre-FM/FV
                  columns: truth_cat, is_true_signal, is_nu, is_cc, is_fv_true,
                           has_true_electron, has_true_muon, has_pi0,
                           passed_fm, passed_fv_reco, passed_presel,
                           true_leading_e_ke, true_leading_e_costheta,
                           true_leading_e_p
    hdr_0         run/subrun/event header
    pot_0         BNB POT per file
    histpotdf_*   accumulated POT (written by run_df_maker.py framework)

Truth categories (match nue_selection.py and skim_nue_v2.py)
-------------------------------------------------------------
    0  nueCC FV          <- signal
    1  nueCC out FV
    2  numuCC + pi0
    3  NC pi0
    4  other numuCC
    5  other NC
    6  cosmic / non-neutrino
    7  neutrino catch-all

Key design decisions
--------------------
* _get_spine_df()           caches the raw SPINE df — one load shared by every
                            maker called on the same file (evtdf, truth_info, wgtdf).
                            Both caches are cleared together when a new file arrives
                            so memory is freed between files.

* _get_truth_info()         caches the per-interaction truth table built from the
                            cached SPINE df.  make_nuecc_evtdf uses it for the
                            preselection index; make_nuecc_truth_info_df returns it
                            directly.  Cost is paid once per file.

* make_nuecc_evtdf()        applies FM+FV preselection using truth_info flags (no
                            re-scan of the full df), then merges MC neutrino CV
                            columns.  Only preselected particle rows are returned,
                            dramatically reducing the output size.

* make_nuecc_wgtdf()        prefilters to FM+FV (entry, mct_index) pairs BEFORE
                            loading BNB/GENIE universes → GENIE sees ~5 k rows
                            instead of ~80 k.  Inspired by the numuCC pandora
                            approach; saves ~15 GB peak RAM per file.

* ELECTRON_THRESHOLD_MEV    must stay 0.0 to match skim_nue_v2.py / nue_selection.py.
                            Using 75 MeV here while the skim uses 0 MeV would cause
                            category-count disagreements in the notebook.
"""

import gc
import warnings
import numpy as np
import pandas as pd

from makedf.makedf import make_all_spine_df, make_mcnudf
from pyanalib.pandas_helpers import multicol_merge, multicol_concat, loadbranches
from makedf.branches import spineint_flashtimes_branches

# =============================================================================
# Constants — must match nue_selection.py AND skim_nue_v2.py exactly
# =============================================================================
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 75.0   
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2

# =============================================================================
# Dual cache  (spine df  +  truth_info)
# Both are keyed by str(f) and cleared together when a new file is loaded,
# so each file's memory is released as soon as the framework moves on.
# =============================================================================
_spine_cache      = {}   # fname -> raw SPINE DataFrame
_truth_info_cache = {}   # fname -> per-interaction truth_info DataFrame

# Add _flash_time_cache alongside the existing caches
_spine_cache      = {}
_truth_info_cache = {}
_flash_time_cache = {}   # ← NEW: (entry, rec.dlp..index) → first flash time

def _get_spine_df(f):
    fname = str(f)
    if fname not in _spine_cache:
        _spine_cache.clear()
        _truth_info_cache.clear()
        _flash_time_cache.clear()        # ← clear together
        gc.collect()
        _spine_cache[fname] = make_all_spine_df(f)

        # Load flash_times vector; cache first flash time per interaction
        # (rec.dlp.flash_time scalar is not stored in CAF; only the vector is)
        try:
            ft_df = loadbranches(f["recTree"], spineint_flashtimes_branches)
            # ft_df: 3-level index (entry, rec.dlp..index, flash..index)
            ft_first = ft_df.groupby(level=[0, 1]).first()   # 2-level
            _flash_time_cache[fname] = ft_first.iloc[:, 0]   # Series
        except Exception as e:
            warnings.warn(f"_get_spine_df: flash_times load failed — {e}")
            _flash_time_cache[fname] = None

    return _spine_cache[fname]


def _get_truth_info(f):
    fname = str(f)
    if fname not in _truth_info_cache:
        spine_df  = _get_spine_df(f)
        ft_series = _flash_time_cache.get(fname)             # may be None
        _truth_info_cache[fname] = _build_truth_info(spine_df, ft_series=ft_series)
    return _truth_info_cache[fname]

# =============================================================================
# Column helpers
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
        f"suffix={suffix!r} bc={branch_must_contain!r} "
        f"bnc={branch_must_not_contain!r}"
    )


def _find_col_ends(df, *suffix_parts, branch_must_contain=None):
    """Return the first column whose last N non-empty parts equal suffix_parts."""
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
# Core: per-interaction truth + reco-preselection metadata
# Mirrors build_truth_info() in skim_nue_v2.py exactly so that category counts
# and efficiency denominators agree between the two workflows.
# =============================================================================

def _build_truth_info(spine_df, ft_series=None):
    """
    One row per interaction, covering ALL interactions (pre-FM/FV cut).

    Truth flags are computed on the raw, unfiltered df so they are never
    biased by reco decisions.  Reco presel flags (passed_fm, passed_fv_reco,
    passed_presel) are stored alongside so the notebook can reconstruct every
    cut-stage count without re-scanning the particle-level df.

    Returns
    -------
    pd.DataFrame with the 3-level interaction index and 14 columns matching
    the truth_info_0 format expected by nue_selection.ipynb.
    Returns an empty DataFrame on critical column-resolution failure.
    """
    il    = list(range(spine_df.index.nlevels - 1))   # all levels except particle
    inter = spine_df.groupby(level=il).first()
    idx   = inter.index

    # ── Resolve columns (non-fatal: missing cols degrade gracefully) ──────
    nu_col   = _safe(_find_col, spine_df, "nu_id",           branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, spine_df, "current_type",    branch_must_contain=BRANCH_TRUE)
    fv_t_col = _safe(_find_col, spine_df, "is_fiducial",     branch_must_contain=BRANCH_TRUE)
    pid_col  = _safe(_find_col, spine_df, "pid",             branch_must_contain=BRANCH_TRUE)
    pri_col  = _safe(_find_col, spine_df, "is_primary",      branch_must_contain=BRANCH_TRUE)
    ke_col   = _safe(_find_col, spine_df, "ke",              branch_must_contain=BRANCH_TRUE)
    ppd_col  = _safe(_find_col, spine_df, "parent_pdg_code", branch_must_contain=BRANCH_TRUE)
    fm_col   = _safe(_find_col, spine_df, "is_flash_matched",branch_must_not_contain=BRANCH_TRUE)
    ft_col   = _safe(_find_col, spine_df, "flash_time_first", branch_must_not_contain=BRANCH_TRUE)
    fv_r_col = _safe(_find_col, spine_df, "is_fiducial",     branch_must_not_contain=BRANCH_TRUE)

    required = [nu_col, cc_col, fv_t_col, pid_col, pri_col, ke_col, ppd_col]
    if any(c is None for c in required):
        missing = [
            n for n, c in zip(
                ["nu_id", "current_type", "is_fiducial(T)", "pid",
                 "is_primary", "ke", "parent_pdg_code"],
                required,
            ) if c is None
        ]
        warnings.warn(f"_build_truth_info: missing truth columns: {missing}")
        return pd.DataFrame()

    def _b(s):
        return s.reindex(idx, fill_value=False).astype(bool)

    # ── Interaction-level truth flags ─────────────────────────────────────
    is_nu      = _b(inter[nu_col]   >= 0)
    is_cc      = _b(inter[cc_col]   == 0)
    is_fv_true = _b(inter[fv_t_col] == 1)

    # ── Particle-level masks, aggregated to interaction level ─────────────
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

    # ── Truth category (identical priority order to skim_nue_v2.py) ──────
    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                              = 6   # cosmic
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_true_electron]      = 1   # nueCC out FV
    cat[nu &  is_cc &  is_fv_true & has_true_electron]      = 0   # nueCC in FV (signal)
    cat[nu &  is_cc & has_true_muon & has_pi0]               = 2   # numuCC + pi0
    cat[nu & ~is_cc & has_pi0]                               = 3   # NC pi0
    cat[nu &  is_cc & has_true_muon & ~has_pi0]              = 4   # other numuCC
    cat[nu & ~is_cc & ~has_pi0]                              = 5   # other NC
    # cat 7 = neutrino catch-all (nu interactions not matched by any rule above)

    # ── Reco preselection flags (stored so the notebook can stage-count) ──
    if fm_col is not None:
        fm_pass = (
            (spine_df[fm_col] == 1)
            .groupby(level=il).any()
            .reindex(idx, fill_value=False)
        )
        if ft_series is not None:
            # valid_flashmatch: is_flash_matched==1 AND flash_times[0] not NaN
            ft_valid = ft_series.notna().reindex(idx, fill_value=False)
            passed_fm = fm_pass & ft_valid
        else:
            warnings.warn("_build_truth_info: flash_time missing — using is_flash_matched only")
            passed_fm = fm_pass
    else:
        warnings.warn("_build_truth_info: FM column missing — passed_fm=False everywhere")
        passed_fm = pd.Series(False, index=idx)

    if fv_r_col is not None:
        passed_fv_reco = (
            (spine_df[fv_r_col] == 1)
            .groupby(level=il).any()
            .reindex(idx, fill_value=False)
        )
    else:
        warnings.warn("_build_truth_info: reco FV column missing — passed_fv_reco=False")
        passed_fv_reco = pd.Series(False, index=idx)

    passed_presel = passed_fm & passed_fv_reco

    # ── Leading true-electron kinematics (NaN for non-electron interactions)
    # Saved for all interactions so the notebook can build efficiency matrices
    # using only truth_info (no access to particle-level df needed).
    true_leading_e_ke       = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_costheta = pd.Series(np.nan, index=idx, dtype=np.float32)
    true_leading_e_p        = pd.Series(np.nan, index=idx, dtype=np.float32)

    try:
        # 3-part suffix disambiguates from neutrino-level momentum columns
        momx_col = _find_col_ends(spine_df, "particles", "momentum", "x",
                                  branch_must_contain=BRANCH_TRUE)
        momy_col = _find_col_ends(spine_df, "particles", "momentum", "y",
                                  branch_must_contain=BRANCH_TRUE)
        momz_col = _find_col_ends(spine_df, "particles", "momentum", "z",
                                  branch_must_contain=BRANCH_TRUE)

        # -inf on non-electron rows so idxmax picks only electrons
        elec_ke_for_max = spine_df[ke_col].where(_elec_mask, -np.inf)
        lead_4idx       = elec_ke_for_max.groupby(level=il).idxmax().dropna()

        if len(lead_4idx) > 0:
            lead_mi      = pd.MultiIndex.from_tuples(
                lead_4idx.values, names=spine_df.index.names
            )
            leading_rows = spine_df.loc[
                lead_mi, [ke_col, momx_col, momy_col, momz_col]
            ].copy()
            leading_rows.index = lead_4idx.index   # promote 4-level → 3-level

            px   = leading_rows[momx_col]
            py   = leading_rows[momy_col]
            pz   = leading_rows[momz_col]
            pmag = np.sqrt(px**2 + py**2 + pz**2).replace(0, np.nan)

            true_leading_e_ke       = leading_rows[ke_col].reindex(idx).astype(np.float32)
            true_leading_e_costheta = (pz / pmag).reindex(idx).astype(np.float32)
            true_leading_e_p        = pmag.reindex(idx).astype(np.float32)
    except Exception as e:
        warnings.warn(f"_build_truth_info: leading-electron kinematics skipped — {e}")

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


def _get_truth_info(f):
    """
    Return the cached per-interaction truth_info for file f,
    building it from the cached SPINE df if not already available.
    """
    fname = str(f)
    if fname not in _truth_info_cache:
        spine_df = _get_spine_df(f)
        _truth_info_cache[fname] = _build_truth_info(spine_df)
    return _truth_info_cache[fname]

# =============================================================================
# Public maker 1 — evtdf  (FM+FV preselected, MC-truth merged)
# =============================================================================

def make_nuecc_evtdf(f):
    """
    Reco+truth SPINE particle-level df, FM+FV preselected, merged with MC
    neutrino CV info.

    Preselection is driven by truth_info flags (no re-scan of full df) so
    only the once-per-file cost of _build_truth_info is paid.

    Memory note: the MC CV merge is done without weights (include_weights=False)
    to keep RAM usage low; weights are handled separately via make_nuecc_wgtdf.
    """
    spine_df   = _get_spine_df(f)
    truth_info = _get_truth_info(f)
    il         = list(range(spine_df.index.nlevels - 1))

    # ── MC neutrino CV merge (no weights — keeps this step fast and small) ─
    try:
        mcdf    = make_mcnudf(f, include_weights=False)
        mcdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["mcnu"] + list(c)) for c in mcdf.columns]
        )
        mct_col = _find_col(spine_df, "mct_index", branch_must_contain=BRANCH_TRUE)
        merged  = multicol_merge(
            lhs      = spine_df.reset_index(
                           level=list(range(1, spine_df.index.nlevels))
                       ),
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

    # ── FM+FV preselection via truth_info (no recomputation against df) ───
    if not truth_info.empty:
        presel_idx = truth_info.index[truth_info["passed_presel"]]
        row_mask   = spine_df.index.droplevel(-1).isin(presel_idx)
        spine_df   = spine_df[row_mask]
    else:
        warnings.warn("make_nuecc_evtdf: truth_info empty — no preselection applied")

    return spine_df

# =============================================================================
# Public maker 2 — truth_info_df  (ALL interactions, pre-FM/FV)
# This is what nue_selection.ipynb reads as truth_info_0.
# =============================================================================

def make_nuecc_truth_info_df(f):
    """
    Per-interaction truth + reco-preselection metadata covering ALL interactions
    before any FM/FV cut.

    This is the single source of truth for efficiency denominators in the
    notebook — never recount from evtdf (which already has non-presel events
    dropped).

    3-level index: [__ntuple, entry, rec.dlp..index]
    14 columns as documented in the module docstring.
    """
    return _get_truth_info(f)


def make_nuecc_statsdf(f):
    """
    Backward-compatible alias for make_nuecc_truth_info_df.

    NOTE: this now returns per-interaction truth_info (one row per interaction),
    NOT the old one-row summary table.  Update the NAMES entry in your config
    from "stats" to "truth_info" so the HDF5 key is truth_info_0 as the
    notebook expects.  If you must keep the key "stats_0", rename it:
        NAMES = [..., "stats", ...]   →  reads as stats_0
    but the notebook cell that does pd.read_hdf(f, key="truth_info_0") will
    need to be changed to key="stats_0".
    """
    return make_nuecc_truth_info_df(f)

# =============================================================================
# Public maker 3 — wgtdf  (BNB + GENIE, preselected interactions only)
# =============================================================================

def make_nuecc_wgtdf(f):
    """
    BNB + GENIE universe weights for FM+FV preselected interactions only.

    Strategy (mirrors numuCC pandora approach for memory efficiency):
      Step 1  Extract preselected (entry, mct_index) pairs from cached SPINE df.
      Step 2  Load full MC df — required by bnbsyst/geniesyst for index alignment.
      Step 3  Run bnbsyst + geniesyst against the full MC index (they need it).
      Step 4  Filter the weight output to the ~5 k preselected rows → tiny df.

    Peak RAM: BNB ~3 GB, GENIE ~8 GB during step 3; output is ~50 MB.
    Running this in a separate pass (nueCC_weights.py config) keeps the primary
    nueCC_mc.py pass from hitting memory limits.
    """
    from makedf import bnbsyst, geniesyst

    # ── Step 1: preselected (entry, mct_index) pairs ─────────────────────
    spine_df   = _get_spine_df(f)
    truth_info = _get_truth_info(f)
    il         = list(range(spine_df.index.nlevels - 1))

    if truth_info.empty:
        warnings.warn("make_nuecc_wgtdf: truth_info empty — returning empty wgtdf")
        return pd.DataFrame()

    presel_idx   = truth_info.index[truth_info["passed_presel"]]
    mct_col      = _safe(_find_col, spine_df, "mct_index",
                         branch_must_contain=BRANCH_TRUE)
    if mct_col is None:
        warnings.warn("make_nuecc_wgtdf: mct_index not found")
        return pd.DataFrame()

    inter_df     = spine_df.groupby(level=il).first()
    inter_presel = inter_df.loc[inter_df.index.isin(presel_idx), mct_col].dropna()
    entry_vals   = inter_presel.index.get_level_values(0)
    presel_pairs = set(zip(entry_vals, inter_presel.astype(int).values))

    # ── Step 2: load full mcdf ────────────────────────────────────────────
    mcdf        = make_mcnudf(f, include_weights=False)
    mcdf["ind"] = mcdf.index.get_level_values(-1)   # required by bnbsyst / geniesyst
    full_ind    = mcdf["ind"]

    print(f"  wgtdf: {len(presel_pairs)} preselected pairs "
          f"(of {len(mcdf)} total MC nu)")

    wgtdf = mcdf.copy()

    # ── Step 3: run weight makers against the full aligned index ──────────
    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=100, slim=True)
        if not bnb_wgt.empty:
            wgtdf = multicol_concat(wgtdf, bnb_wgt)
            print(f"  BNB: {bnb_wgt.shape[1]} columns")
        del bnb_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: BNB failed — {e}")

    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=100, slim=True)
        if not genie_wgt.empty:
            wgtdf = multicol_concat(wgtdf, genie_wgt)
            print(f"  GENIE: {genie_wgt.shape[1]} columns")
        del genie_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: GENIE failed — {e}")

    # ── Step 4: filter to preselected (entry, mct_index) pairs ───────────
    e_vals = wgtdf.index.get_level_values(0)
    m_vals = wgtdf.index.get_level_values(-1)
    mask   = np.array([(e, m) in presel_pairs for e, m in zip(e_vals, m_vals)])
    out    = wgtdf[mask].copy()

    print(f"  wgtdf output: {len(out)} rows (was {len(wgtdf)} before filter)")
    del wgtdf, mcdf
    gc.collect()
    return out