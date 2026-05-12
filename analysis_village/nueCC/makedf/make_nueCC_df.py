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
make_nuecc_wgtdf(f)          — BNB+GENIE universe weights (preselected only)

Output HDF5 keys
----------------
    evt_0         particle-level df, FM+FV preselected, MC-truth merged
    truth_info_0  per-interaction metadata, ALL interactions pre-FM/FV cut
                  columns: truth_cat, is_true_signal, is_nu, is_cc, is_fv_true,
                           has_true_electron, has_true_muon, has_pi0,
                           passed_fm, passed_fv_reco, passed_presel,
                           true_leading_e_ke, true_leading_e_costheta,
                           true_leading_e_p, mct_index
    hdr_0         run/subrun/event header
    pot_0         BNB POT per file

Truth categories
----------------
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

valid_flashmatch proxy
----------------------
    is_flash_matched == 1  AND  flash_total_pe > 0
"""

import gc
import warnings
import numpy as np
import pandas as pd

from makedf.makedf import make_all_spine_df, make_mcnudf, loadbranches
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


def _drop_caches():
    """Release SPINE + truth_info caches. Used by the weights path to reclaim
    memory before the GENIE/BNB universe allocations."""
    _spine_cache.clear()
    _truth_info_cache.clear()
    gc.collect()


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

    # ── Resolve columns ───────────────────────────────────────────────────────
    nu_col   = _safe(_find_col, spine_df, "nu_id",           branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, spine_df, "current_type",    branch_must_contain=BRANCH_TRUE)
    pid_col  = _safe(_find_col, spine_df, "pid",             branch_must_contain=BRANCH_TRUE)
    pri_col  = _safe(_find_col, spine_df, "is_primary",      branch_must_contain=BRANCH_TRUE)
    ke_col   = _safe(_find_col, spine_df, "ke",              branch_must_contain=BRANCH_TRUE)
    ppd_col  = _safe(_find_col, spine_df, "parent_pdg_code", branch_must_contain=BRANCH_TRUE)
    mct_col  = _safe(_find_col, spine_df, "mct_index",       branch_must_contain=BRANCH_TRUE)
    fm_col   = _safe(_find_col, spine_df, "is_flash_matched",branch_must_not_contain=BRANCH_TRUE)
    pe_col   = _safe(_find_col, spine_df, "flash_total_pe",  branch_must_not_contain=BRANCH_TRUE)

    # Vertex columns for geometric FV (truth and reco)
    tvx_col  = _safe(_find_col_ends, spine_df, "vertex", "x", branch_must_contain=BRANCH_TRUE)
    tvy_col  = _safe(_find_col_ends, spine_df, "vertex", "y", branch_must_contain=BRANCH_TRUE)
    tvz_col  = _safe(_find_col_ends, spine_df, "vertex", "z", branch_must_contain=BRANCH_TRUE)
    rvx_col  = _safe(_find_col_ends, spine_df, "vertex", "x", branch_must_not_contain=BRANCH_TRUE)
    rvy_col  = _safe(_find_col_ends, spine_df, "vertex", "y", branch_must_not_contain=BRANCH_TRUE)
    rvz_col  = _safe(_find_col_ends, spine_df, "vertex", "z", branch_must_not_contain=BRANCH_TRUE)

    # Fallback is_fiducial columns (used only if vertex coords unavailable)
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

    # ── Truth flags ───────────────────────────────────────────────────────────
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

    # ── Reco preselection flags ───────────────────────────────────────────────
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

    # ── Leading true-electron kinematics ─────────────────────────────────────
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


# ── Public maker 1: evtdf  (UNCHANGED) ───────────────────────────────────────

def make_nuecc_evtdf(f):
    """
    Reco+truth SPINE particle-level df, FM+FV preselected, merged with MC CV.
    Weights handled by make_nuecc_wgtdf in a separate config run.
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


# ── Public maker 2: truth_info_df  (UNCHANGED) ───────────────────────────────

def make_nuecc_truth_info_df(f):
    """
    Per-interaction truth + reco-preselection metadata covering ALL interactions
    before any FM/FV cut.  Loaded by nue_selection.ipynb as truth_info_0.
    """
    return _get_truth_info(f)


def make_nuecc_statsdf(f):
    """Backward-compatible alias for make_nuecc_truth_info_df."""
    return make_nuecc_truth_info_df(f)


# ── Public maker 3: wgtdf — LIGHTWEIGHT VERSION ──────────────────────────────
#
# This is the only function whose internals changed materially. The output
# format (mcnu-indexed, BNB+GENIE universe columns, preselected rows only)
# is identical to before, so the existing notebook joins still work.
#
# The previous version called _get_spine_df → make_all_spine_df, which loads
# the full SPINE interaction + particle dataframes and merges them. For a
# 100-file batch that's ~4 GB per worker. With -ncpu 10 → ~40 GB just for
# SPINE before any weights are pulled.
#
# The wgtdf only needs to know which (entry, mct_index) pairs are preselected.
# To determine that we need exactly these branches from the SPINE tree:
#
#   per-interaction (true side, "dlp_true"):
#       mct_index, nu_id, current_type, vertex.x/y/z
#       — plus optionally is_fiducial as a fallback
#   per-interaction (reco side, "dlp"):
#       is_flash_matched, flash_total_pe, vertex.x/y/z
#
#   per-particle (true side, "dlp_true.particles"):
#       pid, is_primary, ke
#       — used to enforce the "has_true_electron" signal definition. NOTE:
#         this is only needed for the truth_cat→signal classification, but
#         FOR THE WEIGHTS PATH we don't need signal classification at all,
#         only preselection. So we can SKIP all per-particle branches.
#
# So the wgtdf path becomes ~6–8 short integer/float branches, single-level
# load, no merges. Roughly 50–100 MB per file instead of 4 GB.

def _light_preselected_mct(f):
    """
    Identify (entry, mct_index) pairs that pass preselection (valid FM + reco FV),
    without loading the full SPINE df.

    Uses only the per-interaction true+reco branches needed to evaluate the
    flash-match and reco-FV cuts and to retrieve mct_index.

    Returns
    -------
    pd.MultiIndex with levels ['entry', 'mct'] (unique pairs).
    Empty MultiIndex on failure / no preselected events.
    """
    rec_tree = f["recTree"]

    # --- 1) per-interaction TRUE side: need mct_index only --------------------
    # Note: nu_id and is_cc are NOT needed for preselection (they're truth
    # category flags). The weights modules don't care about truth_cat;
    # they just need the mct_index list. Drop them from the light load.
    try:
        true_int = loadbranches(rec_tree, ["rec.dlp_true.mct_index"])
    except Exception as e:
        warnings.warn(f"_light_preselected_mct: dlp_true.mct_index load failed "
                      f"({e}); falling back to full SPINE load")
        return None

    # --- 2) per-interaction RECO side: FM + reco FV ---------------------------
    # Try the human-readable vertex.x/y/z names first, then the raw I0/I1/I2.
    # If both fail we return None and the caller falls back to the heavy path.
    reco_int = None
    last_err = None
    for vertex_names in (
        ["rec.dlp.vertex.x", "rec.dlp.vertex.y", "rec.dlp.vertex.z"],
        ["rec.dlp.vertex.I0", "rec.dlp.vertex.I1", "rec.dlp.vertex.I2"],
    ):
        try:
            reco_int = loadbranches(rec_tree, [
                "rec.dlp.is_flash_matched",
                "rec.dlp.flash_total_pe",
            ] + vertex_names)
            break
        except Exception as e:
            last_err = e
            reco_int = None

    if reco_int is None:
        warnings.warn(f"_light_preselected_mct: dlp branches load failed "
                      f"({last_err}); falling back to full SPINE load")
        return None

    # --- 3) align true & reco on (entry, rec.dlp..index) ---------------------
    # Both frames share the same per-interaction index after loadbranches.
    # Reset their column MultiIndex to flat names we can look up.
    def _flat(df):
        df = df.copy()
        df.columns = [
            "_".join([str(p) for p in (c if isinstance(c, tuple) else (c,)) if p != ""])
            for c in df.columns
        ]
        return df

    true_int = _flat(true_int)
    reco_int = _flat(reco_int)

    # --- 4) extract the columns we need by suffix matching -------------------
    def _pick(df, *suffixes):
        for s in suffixes:
            for c in df.columns:
                if c.endswith(s):
                    return c
        return None

    mct_c   = _pick(true_int, "mct_index")
    fm_c    = _pick(reco_int, "is_flash_matched")
    pe_c    = _pick(reco_int, "flash_total_pe")
    rvx_c   = _pick(reco_int, "vertex_x", "vertex_I0")
    rvy_c   = _pick(reco_int, "vertex_y", "vertex_I1")
    rvz_c   = _pick(reco_int, "vertex_z", "vertex_I2")

    if mct_c is None or fm_c is None or rvx_c is None or rvy_c is None or rvz_c is None:
        warnings.warn(
            "_light_preselected_mct: needed columns missing "
            f"(mct={mct_c}, fm={fm_c}, vx={rvx_c}, vy={rvy_c}, vz={rvz_c}); "
            "falling back to full SPINE load"
        )
        return None

    # --- 5) per-interaction reduce: FM and FV are per-interaction already ---
    # (no need to groupby — these branches live at dlp..index level)
    fm_pass = reco_int[fm_c] == 1
    if pe_c is not None:
        fm_pass = fm_pass & (reco_int[pe_c] > 0)

    fv_pass = _fiducial_cut_tmp(reco_int[rvx_c], reco_int[rvy_c], reco_int[rvz_c])

    presel_mask = fm_pass & fv_pass

    # --- 6) join mct_index onto preselected interactions --------------------
    # Both dfs have identical index (entry, rec.dlp..index). Use that to align.
    if not true_int.index.equals(reco_int.index):
        # Realign on intersection
        common = true_int.index.intersection(reco_int.index)
        true_int = true_int.loc[common]
        reco_int = reco_int.loc[common]
        presel_mask = presel_mask.loc[common]

    sel = true_int.loc[presel_mask, mct_c].dropna().astype(np.int64)
    sel = sel[sel >= 0]

    if sel.empty:
        # Drop intermediates before returning
        del true_int, reco_int, fm_pass, fv_pass, presel_mask
        gc.collect()
        return pd.MultiIndex.from_arrays([[], []], names=["entry", "mct"])

    # entry level is the first level of the index
    entry_vals = sel.index.get_level_values(0)
    out_idx = pd.MultiIndex.from_arrays(
        [entry_vals, sel.values], names=["entry", "mct"]
    ).unique()

    del true_int, reco_int, fm_pass, fv_pass, presel_mask, sel
    gc.collect()
    return out_idx


def make_nuecc_wgtdf(f):
    """
    BNB + GENIE universe weights for FM+FV preselected interactions only.

    Memory strategy (peak target: well under 5 GB per worker):

      1) Identify preselected (entry, mct_index) pairs using a LIGHT branch
         load — no full SPINE df, no particle-level branches, no merges.
         Cost: ~6 short branches, ~50–100 MB.

      2) Load mcnudf (light — no weights) to get the per-mcnu index frame
         that bnbsyst/geniesyst want as their `full_ind` argument.

      3) Pull BNB universes → filter immediately to preselected rows →
         drop the unfiltered frame → gc.collect.

      4) Pull GENIE universes → filter immediately → drop → gc.collect.

      5) Concat the two narrow (preselected-only) frames and return.

    Output format identical to the previous wgtdf: index is the per-mcnu
    MultiIndex from make_mcnudf, restricted to preselected interactions;
    columns are BNB + GENIE universe weights with their native MultiIndex.
    The notebook joins this against truth_info_0 / evt_0 via mct_index
    exactly as before.
    """
    from makedf import bnbsyst, geniesyst

    # ── Step 1: light preselection ───────────────────────────────────────────
    presel_pair_idx = _light_preselected_mct(f)

    if presel_pair_idx is None:
        # Light path failed — fall back to the original heavy path so the
        # job still produces correct output. (Slow, but at least it works.)
        warnings.warn("make_nuecc_wgtdf: light path failed, using full SPINE load")
        return _make_nuecc_wgtdf_full(f)

    if len(presel_pair_idx) == 0:
        warnings.warn("make_nuecc_wgtdf: no preselected interactions in this file")
        return pd.DataFrame()

    print(f"  wgtdf: {len(presel_pair_idx)} preselected interactions (light path)")

    # ── Step 2: lightweight mcnudf  ──────────────────────────────────────────
    mcdf = make_mcnudf(f, include_weights=False)
    mcdf_entry = mcdf.index.get_level_values(0)
    mcdf_ind   = mcdf.index.get_level_values(-1)

    # Per-mcnu pair index for filtering
    mcdf_pair_idx = pd.MultiIndex.from_arrays(
        [mcdf_entry, np.asarray(mcdf_ind, dtype=np.int64)],
        names=["entry", "mct"],
    )
    keep_mask = mcdf_pair_idx.isin(presel_pair_idx)

    # full_ind is what bnbsyst/geniesyst need — pass them the full range so
    # the universe weights line up with mcdf's index, then filter after.
    full_ind = pd.Series(mcdf_ind, index=mcdf.index)
    presel_mcdf_index = mcdf.index[keep_mask]

    # Free intermediates — we only need full_ind and presel_mcdf_index now
    del mcdf, mcdf_pair_idx, mcdf_entry, mcdf_ind, keep_mask
    gc.collect()

    # ── Step 3: BNB universes ────────────────────────────────────────────────
    out = None
    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=100, slim=True)
        if bnb_wgt is not None and not bnb_wgt.empty:
            bnb_presel = bnb_wgt.loc[bnb_wgt.index.intersection(presel_mcdf_index)]
            del bnb_wgt
            gc.collect()
            print(f"  BNB:   {bnb_presel.shape[1]} cols, {len(bnb_presel)} rows")
            out = bnb_presel
        else:
            del bnb_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: BNB failed — {e}")
    gc.collect()

    # ── Step 4: GENIE universes ──────────────────────────────────────────────
    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=100, slim=True)
        if genie_wgt is not None and not genie_wgt.empty:
            genie_presel = genie_wgt.loc[genie_wgt.index.intersection(presel_mcdf_index)]
            del genie_wgt
            gc.collect()
            print(f"  GENIE: {genie_presel.shape[1]} cols, {len(genie_presel)} rows")
            if out is None:
                out = genie_presel
            else:
                out = multicol_concat(out, genie_presel)
                del genie_presel
        else:
            del genie_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: GENIE failed — {e}")
    gc.collect()

    del full_ind, presel_mcdf_index
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()

    print(f"  wgtdf output: {len(out)} rows × {out.shape[1]} cols")
    return out


# ── Fallback: original heavy-path wgtdf, used only when light path fails ─────

def _make_nuecc_wgtdf_full(f):
    """
    Original heavy wgtdf — kept as a fallback for the case where the light
    branch-subset load fails (e.g. unexpected branch naming on a new ntuple
    version). Use this only when _light_preselected_mct returns None.
    """
    from makedf import bnbsyst, geniesyst

    truth_info = _get_truth_info(f)
    if truth_info.empty:
        _drop_caches()
        return pd.DataFrame()

    presel = truth_info.loc[truth_info["passed_presel"], "mct_index"]
    presel = presel[presel >= 0]
    if presel.empty:
        _drop_caches()
        return pd.DataFrame()

    entry_vals      = presel.index.get_level_values(0)
    presel_pair_idx = pd.MultiIndex.from_arrays(
        [entry_vals, presel.astype(np.int64).values], names=["entry", "mct"]
    ).unique()

    mcdf = make_mcnudf(f, include_weights=False)
    mcdf_entry = mcdf.index.get_level_values(0)
    mcdf_ind   = mcdf.index.get_level_values(-1)
    mcdf_pair_idx = pd.MultiIndex.from_arrays(
        [mcdf_entry, np.asarray(mcdf_ind, dtype=np.int64)], names=["entry", "mct"]
    )
    keep_mask = mcdf_pair_idx.isin(presel_pair_idx)
    full_ind = pd.Series(mcdf_ind, index=mcdf.index)
    presel_mcdf_index = mcdf.index[keep_mask]

    del mcdf, mcdf_pair_idx, mcdf_entry, mcdf_ind, keep_mask
    gc.collect()
    _drop_caches()

    out = None
    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=100, slim=True)
        if bnb_wgt is not None and not bnb_wgt.empty:
            bnb_presel = bnb_wgt.loc[bnb_wgt.index.intersection(presel_mcdf_index)]
            del bnb_wgt; gc.collect()
            out = bnb_presel
    except Exception as e:
        warnings.warn(f"_make_nuecc_wgtdf_full: BNB failed — {e}")
    gc.collect()

    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=100, slim=True)
        if genie_wgt is not None and not genie_wgt.empty:
            genie_presel = genie_wgt.loc[genie_wgt.index.intersection(presel_mcdf_index)]
            del genie_wgt; gc.collect()
            if out is None:
                out = genie_presel
            else:
                out = multicol_concat(out, genie_presel)
                del genie_presel
    except Exception as e:
        warnings.warn(f"_make_nuecc_wgtdf_full: GENIE failed — {e}")
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()
    return out