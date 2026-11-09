#!/usr/bin/env python3
"""
make_nueCC_df.py
----------------
SPINE DLP-based nueCC inclusive MC DataFrame makers for cafpyana.

Public functions
----------------
make_nuecc_evtdf(f)          — reco+truth particle df, FM+FV preselected  (MC)
make_nuecc_evtdf_data(f)     — reco-only particle df, FM+FV preselected   (DATA)
make_nuecc_truth_info_df(f)  — per-interaction truth+presel metadata (ALL interactions)
make_nuecc_statsdf(f)        — alias for make_nuecc_truth_info_df
make_nuecc_wgtdf(f)          — BNB + GENIE universe weights (preselected nus only)
make_nuecc_flashdf(f)        — per-interaction flash-match table, ALL interactions (MC)
make_nuecc_flashdf_data(f)   — same for data (on-beam, off-beam)

Output HDF5 keys (via nueCC_mc.py config)
------------------------------------------
    evt_0         particle-level df, FM+FV preselected, MC-truth merged
    truth_info_0  per-interaction metadata, ALL interactions pre-FM/FV cut
    flash_0       per-interaction flash-match variables, ALL interactions pre-FM/FV
                  cut (cosmic-only interactions prescaled, see FLASHDF_COSMIC_PRESCALE)
    hdr_0         run/subrun/event header
    pot_0         BNB POT per file

Output HDF5 keys (via nueCC_data.py config)
--------------------------------------------
    evt_0         reco-only particle-level df, FM+FV preselected
    flash_0       per-interaction flash-match variables, ALL interactions pre-FM/FV cut
    hdr_0         run/subrun/event header
    pot_0         EXT POT per file (or BNB depending on run type)

Truth categories (from GENIE truth, rec.mc.nu — see _categorize_neutrinos)
--------------------------------------------------------------------------
First matching category wins (same order as the SPINE category config):

    0  nue CC in FV      <- signal: (anti)nue CC, true vertex in FV,
                            exactly 1 primary e+- with KE >= 500 MeV
    1  nue CC not in FV     same electron requirement, true vertex outside FV
    2  numu CC pi0          CC, primary muon KE > 50 MeV, >= 1 primary pi0
    3  nu NC pi0            NC, >= 1 primary pi0
    4  other numu CC        CC, primary muon KE > 50 MeV
    5  other nu NC          NC
    6  other nu             any other neutrino: nue CC whose electron has
                            KE < 500 MeV (true-threshold migration into the
                            reco >= 500 MeV selection), numu CC with a muon
                            KE <= 50 MeV, nutau CC, and matched neutrinos
                            with no rec.mc.nu entry
    7  cosmic               reco interaction not matched to a neutrino

    Each reco interaction is linked to its GENIE neutrino through
    rec.dlp_true.nu_id, which indexes rec.mc.nu. rec.dlp_true.mct_index is
    the MCTruth index and is 0 for every neutrino in a multi-neutrino spill,
    so it cannot be used as the rec.mc.nu key; the output 'mct_index'
    (truth_info and evt) is therefore filled with nu_id.

    Truth particles are NOT taken from the reco-matched dlp_true particle
    table: every reco fragment carries its best-matched true particle, so one
    electron split into two reco fragments was counted twice, and particles
    that were never reconstructed (e.g. a muon) were missing.

Efficiency denominator
----------------------
    truth_info["is_true_signal"] is True for exactly ONE row per true signal
    neutrino: the reco interaction that got furthest (presel > FM > none) if
    the neutrino was split into several reco interactions, or an extra
    "truth-only" row (is_truth_only=True, rec.dlp..index = -1 - nu index,
    passed_* = False) if no reco interaction was matched to it at all.
    So truth_info["is_true_signal"].sum() counts every true signal neutrino
    once.  truth_cat is 0 on ALL reco interactions of a signal neutrino.

Fiducial volume (fiducial_cut_tmp — matches C++ definition)
------------------------------------------------------------
    |x| in (5, 190) cm
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

try:
    from makedf.branches import (spineint_flashscores_branches as _FLASH_SCORE_BRANCHES,
                                 spineint_flashvolumeids_branches as _FLASH_VOL_BRANCHES)
except ImportError:
    _FLASH_SCORE_BRANCHES = _FLASH_VOL_BRANCHES = None

# ── Reco-only branch lists (for data mode) ────────────────────────────────────
try:
    from makedf.makedf import (
        spineint_branches          as _SPINEINT_BRANCHES,
        spinepart_branches         as _SPINEPART_BRANCHES,
        spineint_flashids_branches     as _SPINEINT_FLASHIDS,
        spineint_flashscores_branches  as _SPINEINT_FLASHSCORES,
        spineint_flashtimes_branches   as _SPINEINT_FLASHTIMES,
        spineint_matched_branches      as _SPINEINT_MATCHED,
        spinepart_matched_branches     as _SPINEPART_MATCHED,
    )
    _DATA_BRANCHES_AVAILABLE = True
except ImportError:
    _DATA_BRANCHES_AVAILABLE = False
    warnings.warn(
        "make_nueCC_df: could not import reco-only branch lists from makedf.makedf; "
        "make_nuecc_evtdf_data will not be available."
    )

# ── Constants — must match nue_selection.py exactly ──────────────────────────
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 500.0     # signal: primary e+- KE >= this
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2

# ── Columns dropped from evt (not used by the analysis; saves space) ─────────
#    Only the reco/true copies of these fields go; branches.py stays unchanged
#    because other analyses share it.
EVT_DROP_FIELDS = ("cathode_offset", "is_cathode_crosser")

# ── Flash table (flash_0): one row per reco interaction, before any cut ──────
# MC cosmic-only interactions (truth_cat 7, ~90% of all reco interactions) are
# kept 1 in FLASHDF_COSMIC_PRESCALE (deterministic in entry/interaction index)
# with weight w_prescale = FLASHDF_COSMIC_PRESCALE, so the table stays small and
# weighted histograms are unbiased. Every neutrino interaction and every data
# interaction is kept (w_prescale = 1).
FLASHDF_COSMIC_PRESCALE = 10
# Opflash summary per event: largest flash with time inside this window [us]
# (generous around the 1.6 us BNB spill, so the window itself can be studied).
OPFLASH_BEAM_WINDOW_US  = (-2.0, 4.0)
_OPFLASH_BRANCHES = ["rec.opflashes." + b for b in ("totalpe", "time")]

# ── GENIE truth branches (rec.mc.nu, indexed by dlp_true.nu_id) ──────────────
_MCNU_BRANCHES = ["rec.mc.nu." + b for b in [
    "E", "pdg", "iscc", "genie_mode", "position.x", "position.y", "position.z",
]]
_MCPRIM_BRANCHES = ["rec.mc.nu.prim." + b for b in [
    "pdg", "genE", "genp.x", "genp.y", "genp.z",
]]
ELECTRON_MASS_GEV = 0.000511
MUON_MASS_GEV     = 0.105658

# ── Vertex coordinate I0/I1/I2 → x/y/z mapping ──────────────────────────────
# SPINE data-mode CAFs store reco vertex as I0/I1/I2 instead of x/y/z.
# This mapping ensures both MC and data use the same geometric FV cut.
_COORD_IDX_MAP = {'x': 'I0', 'y': 'I1', 'z': 'I2'}

# ── Module-level caches (one SPINE load shared across all DFS makers per file)
_spine_cache      = {}
_spine_data_cache = {}
_truth_info_cache = {}
_data_presel_cache = {}


# ── MC spine loader (reco + truth merged) ─────────────────────────────────────

def _get_spine_df(f):
    fname = str(f)
    if fname not in _spine_cache:
        _spine_cache.clear()
        _truth_info_cache.clear()
        gc.collect()
        _spine_cache[fname] = _complete_unmatched_reco(make_all_spine_df(f), f)
    return _spine_cache[fname]


_XYZ = {"I0": "x", "I1": "y", "I2": "z"}


def _complete_unmatched_reco(spine_df, f):
    """
    Same reco content for MC as for data (makedf.make_all_spine_df itself is unchanged).

    make_all_spine_df builds the interaction rows from the reco→true match table
    (rec.dlp.match_ids), so a reco interaction with NO interaction-level truth match is
    either missing entirely (all its particles unmatched too) or kept with every
    interaction-level reco field NaN (vertex, is_flash_matched, flash PE, ...). Either way it
    fails FM / FV in MC, while the reco-only data loader keeps it with its real values.

    Here the rows of those interactions are replaced by the reco-only rows (the loader used for
    data, columns renamed to the MC convention: I0/I1/I2 → x/y/z where MC uses that). Truth
    columns stay NaN (nu_id NaN → truth_cat 7, no neutrino); truth columns that the particle
    match did fill are kept. Interactions with an interaction-level match are untouched.
    """
    try:
        da = _get_spine_df_data(f)
    except Exception as e:
        warnings.warn(f"_complete_unmatched_reco: reco-only load failed, MC spine df unchanged — {e}")
        return spine_df
    if da is None or da.empty:
        return spine_df

    mc_cols = set(spine_df.columns)
    ren = {}
    for c in da.columns:
        if c in mc_cols:
            ren[c] = c
            continue
        alt = tuple(_XYZ.get(x, x) for x in (c if isinstance(c, tuple) else (c,)))
        if alt in mc_cols:
            ren[c] = alt
    da = da[list(ren)].rename(columns=ren)

    id_col = next((c for c in spine_df.columns
                   if isinstance(c, tuple) and c[:3] == ("rec", "dlp", "id") and all(x == "" for x in c[3:])), None)
    il = list(range(spine_df.index.nlevels - 1))
    mc_inter = spine_df.index.droplevel(-1)
    ok_inter = (mc_inter[spine_df[id_col].notna().to_numpy()].unique() if id_col is not None
                else mc_inter.unique())
    da_inter = da.index.droplevel(-1)
    fix = da_inter[~da_inter.isin(ok_inter)].unique()
    if len(fix) == 0:
        _spine_data_cache.clear()
        return spine_df

    new = da[da_inter.isin(fix)].reindex(columns=spine_df.columns)
    old = spine_df[mc_inter.isin(fix)]
    if len(old):
        new = new.combine_first(old.reindex(index=new.index))[spine_df.columns]   # keep particle truth
    out = pd.concat([spine_df[~mc_inter.isin(fix)], new]).sort_index()
    # keep the make_all_spine_df dtypes (concat with NaN rows upcasts int → float, float32 → float64)
    for c, dt in spine_df.dtypes.items():
        if out[c].dtype == dt:
            continue
        if np.issubdtype(dt, np.floating) or not out[c].isna().any():
            try:
                out[c] = out[c].astype(dt)
            except (TypeError, ValueError):
                pass
    print(f"  MC spine: {len(fix)} reco interactions without an interaction-level truth match "
          f"taken from the reco-only loader ({int((~fix.isin(mc_inter.unique())).sum())} were missing, "
          f"{int(fix.isin(mc_inter.unique()).sum())} had NaN reco fields)")
    _spine_data_cache.clear()
    del da
    gc.collect()
    return out


# ── Data spine loader (reco-only, no dlp_true branches) ───────────────────────

def _get_spine_df_data(f):
    """
    Reco-only SPINE df for data CAFs.
    """
    fname = str(f)
    if fname not in _spine_data_cache:
        _spine_data_cache.clear()
        gc.collect()

        if not _DATA_BRANCHES_AVAILABLE:
            raise RuntimeError(
                "_get_spine_df_data: reco-only branch lists could not be "
                "imported from makedf.makedf"
            )

        rec = f["recTree"]

        # ── Interaction-level (2-level row index, 4-level col MultiIndex) ─────
        int_df = _loadbranches(rec, _SPINEINT_BRANCHES)
        inter_idx_names = list(int_df.index.names)  # ['entry', 'rec.dlp..index']

        for extra_branches in [_SPINEINT_FLASHIDS, _SPINEINT_FLASHSCORES,
                               _SPINEINT_MATCHED]:
            if extra_branches is None:
                continue
            try:
                extra_df = _loadbranches(rec, extra_branches)
                if extra_df.index.nlevels == int_df.index.nlevels:
                    # flatten both to same depth before joining
                    int_df = int_df.join(extra_df, how="left")
            except Exception:
                pass

        # ── Particle-level (3-level row index, 5-level col MultiIndex) ────────
        part_df = _loadbranches(rec, _SPINEPART_BRANCHES)
        part_idx_names = list(part_df.index.names)

        if _SPINEPART_MATCHED is not None:
            try:
                pm = _loadbranches(rec, _SPINEPART_MATCHED)
                if pm.index.nlevels == part_df.index.nlevels:
                    part_df = part_df.join(pm, how="left")
            except Exception:
                pass

        # ── Flatten column MultiIndex to strings for both dfs ─────────────────
        def _flatten_cols(df):
            return [
                ".".join(p for p in c if p) if isinstance(c, tuple) else c
                for c in df.columns
            ]

        int_flat  = int_df.copy()
        part_flat = part_df.copy()
        int_flat.columns  = _flatten_cols(int_df)
        part_flat.columns = _flatten_cols(part_df)

        # ── Merge on shared interaction index levels ───────────────────────────
        part_reset = part_flat.reset_index()   # 3 idx levels → columns
        int_reset  = int_flat.reset_index()    # 2 idx levels → columns

        # inter_idx_names are plain strings — they now exist as columns in both
        merged = part_reset.merge(int_reset, on=inter_idx_names, how="left")
        merged = merged.set_index(part_idx_names)

        # ── Restore column MultiIndex ──────────────────────────────────────────
        # Build a unified MultiIndex using the deeper (5-level) depth.
        # int cols get padded with trailing '' to match part depth.
        max_depth = max(
            max(len(c) for c in int_df.columns  if isinstance(c, tuple)),
            max(len(c) for c in part_df.columns if isinstance(c, tuple)),
        )

        orig_cols = {}
        for orig, flat in zip(int_df.columns,  _flatten_cols(int_df)):
            orig_cols[flat] = orig
        for orig, flat in zip(part_df.columns, _flatten_cols(part_df)):
            orig_cols[flat] = orig

        def _pad(t, depth):
            return tuple(t) + ('',) * (depth - len(t)) if isinstance(t, tuple) else (t,) + ('',) * (depth - 1)

        new_cols = []
        for c in merged.columns:
            if c in orig_cols:
                new_cols.append(_pad(orig_cols[c], max_depth))
            else:
                new_cols.append(_pad((c,), max_depth))

        merged.columns = pd.MultiIndex.from_tuples(new_cols)
        _spine_data_cache[fname] = merged

    return _spine_data_cache[fname]


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


def _leaf(col):
    """Last non-empty part of a (multi-level) column name."""
    parts = [p for p in (col if isinstance(col, tuple) else (col,)) if p != ""]
    return parts[-1] if parts else ""


def _drop_unused_fields(df):
    """Remove EVT_DROP_FIELDS columns (reco and true) from a particle-level df."""
    drop = [c for c in df.columns if _leaf(c) in EVT_DROP_FIELDS]
    return df.drop(columns=drop) if drop else df


# ══════════════════════════════════════════════════════════════════════════════
# FIX: Vertex column lookup that handles both x/y/z and I0/I1/I2 naming
# ══════════════════════════════════════════════════════════════════════════════
# Data-mode SPINE CAFs store reco vertex coordinates as I0/I1/I2 instead of
# x/y/z. The old code's _find_col_ends(df, "vertex", "x") silently failed for
# data, causing _apply_presel and _build_truth_info to fall back to the
# is_fiducial flag — which uses a DIFFERENT FV boundary than the geometric
# _fiducial_cut_tmp.  This produced an MC/data selection inconsistency visible
# as missing data events near the FV boundary (|x|≈190, |y|≈190, z≈10).
#
# The helper below tries the canonical name first, then the I0/I1/I2 fallback,
# ensuring the same geometric FV is applied to both MC and data.
# ══════════════════════════════════════════════════════════════════════════════

def _find_vertex_col(df, coord, branch_kw=None):
    """
    Find the vertex coordinate column for 'x', 'y', or 'z'.

    Tries ("vertex", "x") first, then ("vertex", "I0") as fallback.
    Works for both reco (branch_kw=None or branch_must_not_contain)
    and truth (branch_kw=BRANCH_TRUE via branch_must_contain).

    Parameters
    ----------
    df       : DataFrame to search
    coord    : 'x', 'y', or 'z'
    branch_kw: None  → no constraint
               str   → if starts with '!' → branch_must_not_contain (reco)
                        otherwise          → branch_must_contain (truth)

    Returns
    -------
    column key or None
    """
    kw = {}
    if branch_kw is not None:
        if branch_kw.startswith('!'):
            kw['branch_must_not_contain'] = branch_kw[1:]
        else:
            kw['branch_must_contain'] = branch_kw

    # Try canonical name: ("vertex", "x")
    col = _safe(_find_col_ends, df, "vertex", coord, **kw)
    if col is not None:
        return col

    # Fallback: ("vertex", "I0") for data-mode SPINE columns
    idx_name = _COORD_IDX_MAP.get(coord)
    if idx_name is not None:
        col = _safe(_find_col_ends, df, "vertex", idx_name, **kw)
    return col


# ── Shared FM + FV preselection logic ─────────────────────────────────────────

def _apply_presel(spine_df, flash_times_s=None, inter=None):
    """
    Apply FM + reco-FV preselection to spine_df (works for both MC and data).

    Parameters
    ----------
    spine_df      : SPINE particle-level DataFrame (2- or 3-level MultiIndex)
    flash_times_s : optional flash_times Series (see _build_truth_info for details)
    inter         : optional spine_df.groupby(level=il).first(), if already built

    Returns
    -------
    passed_fm      : bool Series, interaction-level index
    passed_fv_reco : bool Series, interaction-level index
    inter_idx      : MultiIndex of unique interactions
    il             : list of groupby levels (all but last)
    """
    il    = list(range(spine_df.index.nlevels - 1))
    if inter is None:
        inter = spine_df.groupby(level=il).first()
    idx   = inter.index

    # ── Flash-match ────────────────────────────────────────────────────────
    fm_col = _safe(_find_col, spine_df, "is_flash_matched",
                   branch_must_not_contain=BRANCH_TRUE)
    pe_col = _safe(_find_col, spine_df, "flash_total_pe",
                   branch_must_not_contain=BRANCH_TRUE)

    if fm_col is not None:
        fm_pass = (spine_df[fm_col] == 1)

        if flash_times_s is not None:
            ft_il        = list(range(flash_times_s.index.nlevels - 1))
            first_ft     = flash_times_s.groupby(level=ft_il).first()
            has_valid_ft = first_ft.notna().reindex(idx, fill_value=False).astype(bool)
            fm_inter     = fm_pass.groupby(level=il).any().reindex(idx, fill_value=False)
            passed_fm    = fm_inter & has_valid_ft
        else:
            if pe_col is not None:
                fm_pass = fm_pass & (spine_df[pe_col] > 0)
            passed_fm = fm_pass.groupby(level=il).any().reindex(idx, fill_value=False)
    else:
        passed_fm = pd.Series(False, index=idx)

    # ── Reco FV (fixed: tries x/y/z then I0/I1/I2) ───────────────────────
    rvx_col = _find_vertex_col(spine_df, 'x', f'!{BRANCH_TRUE}')
    rvy_col = _find_vertex_col(spine_df, 'y', f'!{BRANCH_TRUE}')
    rvz_col = _find_vertex_col(spine_df, 'z', f'!{BRANCH_TRUE}')
    fv_r_col = _safe(_find_col, spine_df, "is_fiducial",
                     branch_must_not_contain=BRANCH_TRUE)

    if rvx_col and rvy_col and rvz_col:
        passed_fv_reco = _fiducial_cut_tmp(
            inter[rvx_col], inter[rvy_col], inter[rvz_col]
        ).reindex(idx, fill_value=False).astype(bool)
    elif fv_r_col:
        warnings.warn("_apply_presel: reco vertex coords missing, using is_fiducial")
        passed_fv_reco = (
            (spine_df[fv_r_col] == 1)
            .groupby(level=il).any()
            .reindex(idx, fill_value=False)
        )
    else:
        passed_fv_reco = pd.Series(False, index=idx)

    return passed_fm, passed_fv_reco, idx, il


# ── Flash-match table (all interactions, before FM / FV) ──────────────────────

def _ragged_first(rec, branches, idx, how="first"):
    """Per-interaction summary of a ragged rec.dlp.flash_* branch (one entry per
    matched TPC volume; empty for unmatched interactions): first entry, or the
    number of entries (how='size'). Returns NaN / 0 where absent."""
    if _loadbranches is None or branches is None:
        return pd.Series(np.nan, index=idx)
    try:
        raw = _loadbranches(rec, branches)
    except Exception as e:
        warnings.warn(f"_ragged_first: {branches} not loaded — {e}")
        return pd.Series(np.nan, index=idx)
    s = raw[raw.columns[0]]
    g = s.groupby(level=list(range(s.index.nlevels - 1)))
    out = g.size() if how == "size" else g.first()
    return out.reindex(idx, fill_value=0 if how == "size" else np.nan)


def _opflash_event_summary(rec, entries):
    """Event-level optical-flash summary (all flashes, any interaction):
    n_opflash, largest flash PE / time inside OPFLASH_BEAM_WINDOW_US, largest PE overall."""
    cols = ["evt_n_opflash", "evt_beam_flash_pe", "evt_beam_flash_time", "evt_max_flash_pe"]
    out = pd.DataFrame(np.nan, index=pd.Index(np.unique(entries), name="entry"), columns=cols)
    if _loadbranches is None:
        return out
    try:
        of = _loadbranches(rec, _OPFLASH_BRANCHES)
    except Exception as e:
        warnings.warn(f"_opflash_event_summary: rec.opflashes not loaded — {e}")
        return out
    of.columns = [_leaf(c) for c in of.columns]          # ('rec','opflashes','totalpe') → 'totalpe'
    ent = of.index.get_level_values(0)
    lo, hi = OPFLASH_BEAM_WINDOW_US
    inw = of[(of["time"] > lo) & (of["time"] < hi)]
    g_all = of.groupby(level=0)
    out["evt_n_opflash"]    = g_all.size().reindex(out.index).fillna(0)
    out["evt_max_flash_pe"] = g_all["totalpe"].max().reindex(out.index)
    if len(inw):
        imax = inw.groupby(level=0)["totalpe"].idxmax()
        best = inw.loc[imax.values]
        best.index = best.index.get_level_values(0)
        out["evt_beam_flash_pe"]   = best["totalpe"].reindex(out.index)
        out["evt_beam_flash_time"] = best["time"].reindex(out.index)
    return out


def _build_flash_table(spine_df, rec, passed_fm, passed_fv_reco, truth_cat=None):
    """
    One row per reco interaction (ALL, before flash-match or FV cut) with the
    flash-match variables needed to study the FM cut itself (Lynn TN Fig. 1):

      vtx_x/y/z, passed_fm, passed_fv_reco, is_flash_matched,
      flash_total_pe (−1 if unmatched), flash_hypo_pe, flash_score, flash_time
      (first matched flash, as used by the FM cut), n_flash (matched TPC volumes),
      flash_volume_id, depositions_sum, size, num_primary_particles, is_contained,
      is_time_contained, evt_* opflash summary (see _opflash_event_summary),
      truth_cat (MC only), w_prescale.

    SPINE stores only the winning match (n_flash entries = TPC volumes of that
    one match, same score), so there is no runner-up flash score.
    """
    il    = list(range(spine_df.index.nlevels - 1))
    inter = spine_df.groupby(level=il).first()
    idx   = inter.index

    def col(name):
        # interaction-level reco field only: rec.dlp.<name> (not rec.dlp.particles.<name>,
        # not rec.dlp_true.*) — _find_col matches on the last name part alone
        for c in spine_df.columns:
            parts = [p for p in (c if isinstance(c, tuple) else (c,)) if p != ""]
            if len(parts) >= 2 and parts[-1] == name and parts[-2] == "dlp":
                return inter[c]
        return pd.Series(np.nan, index=idx)

    out = pd.DataFrame(index=idx)
    for k in ("x", "y", "z"):
        c = _find_vertex_col(spine_df, k, f"!{BRANCH_TRUE}")
        out[f"vtx_{k}"] = (inter[c] if c is not None else np.nan)
    out["passed_fm"]             = passed_fm.reindex(idx, fill_value=False).astype(bool)
    out["passed_fv_reco"]        = passed_fv_reco.reindex(idx, fill_value=False).astype(bool)
    out["is_flash_matched"]      = col("is_flash_matched").fillna(0)
    out["flash_total_pe"]        = col("flash_total_pe")
    out["flash_hypo_pe"]         = col("flash_hypo_pe")
    out["flash_score"]           = _ragged_first(rec, _FLASH_SCORE_BRANCHES, idx)
    out["flash_time"]            = _ragged_first(rec, _SPINE_FT_BRANCHES, idx)
    out["n_flash"]               = _ragged_first(rec, _SPINE_FT_BRANCHES, idx, how="size")
    out["flash_volume_id"]       = _ragged_first(rec, _FLASH_VOL_BRANCHES, idx).fillna(-1)
    out["depositions_sum"]       = col("depositions_sum")
    out["size"]                  = col("size")
    out["num_primary_particles"] = col("num_primary_particles")
    out["is_contained"]          = col("is_contained").fillna(0)
    out["is_time_contained"]     = col("is_time_contained").fillna(0)

    ev = _opflash_event_summary(rec, idx.get_level_values(0))
    out = pd.concat([out, ev.reindex(idx.get_level_values(0)).set_axis(idx)], axis=1)

    w = np.ones(len(idx), dtype=np.float32)
    if truth_cat is not None:
        tc = truth_cat.reindex(idx).fillna(7).astype(np.int8)
        out["truth_cat"] = tc
        e   = np.asarray(idx.get_level_values(0), dtype=np.int64)
        i   = np.asarray(idx.get_level_values(1), dtype=np.int64)
        cos = (tc.values == 7)
        keep_cos = ((e * 7919 + i) % FLASHDF_COSMIC_PRESCALE) == 0
        keep = ~cos | keep_cos
        w[cos] = FLASHDF_COSMIC_PRESCALE
        out, w = out[keep], w[keep]
    out["w_prescale"] = w

    return out.astype({
        "vtx_x": np.float32, "vtx_y": np.float32, "vtx_z": np.float32,
        "passed_fm": bool, "passed_fv_reco": bool, "is_flash_matched": np.int8,
        "flash_total_pe": np.float32, "flash_hypo_pe": np.float32,
        "flash_score": np.float32, "flash_time": np.float32, "n_flash": np.int8,
        "flash_volume_id": np.int8, "depositions_sum": np.float32, "size": np.float32,
        "num_primary_particles": np.float32, "is_contained": np.int8,
        "is_time_contained": np.int8, "evt_n_opflash": np.float32,
        "evt_beam_flash_pe": np.float32, "evt_beam_flash_time": np.float32,
        "evt_max_flash_pe": np.float32, "w_prescale": np.float32,
        **({"truth_cat": np.int8} if truth_cat is not None else {}),
    })


# ── GENIE truth: one row per neutrino in rec.mc.nu ───────────────────────────

def _load_genie_truth(f):
    """
    Per-neutrino GENIE truth, indexed (entry, rec.mc.nu index).  The second
    level is what rec.dlp_true.nu_id refers to.

    Columns: E [GeV], pdg, iscc, genie_mode, position.{x,y,z},
             n_e_above  (# primary e+- with KE >= ELECTRON_THRESHOLD_MEV),
             has_mu, has_pi0,
             lead_e_ke [MeV], lead_e_p [MeV/c], lead_e_costheta
             (leading primary e+- by energy, no threshold; NaN if none).
    """
    rec  = f["recTree"]
    nu   = _loadbranches(rec, _MCNU_BRANCHES)
    nu.columns = [".".join(p for p in c[3:] if p) for c in nu.columns]
    idx  = nu.index
    il   = list(range(idx.nlevels))

    prim = _loadbranches(rec, _MCPRIM_BRANCHES)
    prim.columns = [".".join(p for p in c[4:] if p) for c in prim.columns]
    apdg = prim["pdg"].abs()

    e    = prim[apdg == 11]
    e_ke = (e["genE"] - ELECTRON_MASS_GEV) * 1e3
    nu["n_e_above"] = (
        (e_ke >= ELECTRON_THRESHOLD_MEV).groupby(level=il).sum()
        .reindex(idx, fill_value=0).astype(np.int16)
    )
    mu_ke = (prim["genE"][apdg == 13] - MUON_MASS_GEV) * 1e3
    nu["has_mu"] = (
        (mu_ke > MUON_THRESHOLD_MEV).groupby(level=il).any()
        .reindex(idx, fill_value=False).astype(bool)
    )
    nu["has_pi0"] = (
        (apdg == 111).groupby(level=il).any()
        .reindex(idx, fill_value=False).astype(bool)
    )

    # leading primary e+- — idxmax over electron rows only
    nu["lead_e_ke"] = np.nan
    nu["lead_e_p"] = np.nan
    nu["lead_e_costheta"] = np.nan
    if len(e) > 0:
        lead = e_ke.groupby(level=il).idxmax()
        lr   = e.loc[pd.MultiIndex.from_tuples(lead.values, names=e.index.names)]
        lr.index = lead.index
        p    = np.sqrt(lr["genp.x"]**2 + lr["genp.y"]**2 + lr["genp.z"]**2)
        nu["lead_e_ke"]       = e_ke.loc[lead.values].set_axis(lead.index).reindex(idx)
        nu["lead_e_p"]        = (p * 1e3).reindex(idx)
        nu["lead_e_costheta"] = (lr["genp.z"] / p.replace(0, np.nan)).reindex(idx)
    return nu


def _empty_genie_truth():
    cols = ([b.split("rec.mc.nu.")[1] for b in _MCNU_BRANCHES]
            + ["n_e_above", "has_mu", "has_pi0",
               "lead_e_ke", "lead_e_p", "lead_e_costheta"])
    return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_arrays(
        [[], []], names=["entry", "rec.mc.nu..index"]), dtype=float)


def _categorize_neutrinos(g):
    """truth_cat for each GENIE neutrino; first matching category wins, in
    the same order as the SPINE category config (see module docstring)."""
    apdg   = g["pdg"].abs()
    cc     = g["iscc"].astype(bool)
    pi0    = g["has_pi0"].astype(bool)
    mu     = g["has_mu"].astype(bool)
    in_fv  = _fiducial_cut_tmp(g["position.x"], g["position.y"], g["position.z"])
    one_e  = g["n_e_above"] == 1
    nue_cc = cc & (apdg == 12)

    cat = np.select(
        [nue_cc & in_fv & one_e,      # 0  nue CC in FV (signal)
         nue_cc & ~in_fv & one_e,     # 1  nue CC not in FV
         cc & mu & pi0,               # 2  numu CC pi0
         ~cc & pi0,                   # 3  nu NC pi0
         cc & mu,                     # 4  other numu CC
         ~cc],                        # 5  other nu NC
        [0, 1, 2, 3, 4, 5],
        default=6)                    # 6  other nu   (7 = cosmic, set per reco interaction)
    return pd.Series(cat, index=g.index, dtype=np.int8), in_fv.astype(bool), one_e


# ── Core: build per-interaction truth + reco-presel metadata ─────────────────

def _build_truth_info(spine_df, genie, flash_times_s=None):
    """
    Parameters
    ----------
    spine_df      : merged SPINE reco+truth particle-level DataFrame
    genie         : _load_genie_truth(f) (per-neutrino GENIE truth)
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

    nu_col = _safe(_find_col, spine_df, "nu_id", branch_must_contain=BRANCH_TRUE)
    if nu_col is None:
        warnings.warn("_build_truth_info: missing column dlp_true nu_id")
        return pd.DataFrame()

    passed_fm, passed_fv_reco, _, _ = _apply_presel(spine_df, flash_times_s, inter=inter)
    passed_presel = passed_fm & passed_fv_reco

    # ── Link each reco interaction to its GENIE neutrino (entry, nu_id) ───────
    nu_id = inter[nu_col].fillna(-1).astype(np.int32)
    is_nu = (nu_id >= 0)

    g_cat, g_fv, g_one_e = _categorize_neutrinos(genie)
    gi = genie.assign(truth_cat=g_cat, is_fv_true=g_fv, has_true_electron=g_one_e)
    key = pd.MultiIndex.from_arrays([idx.get_level_values(0), nu_id.values])
    g = gi.reindex(key)
    g.index = idx
    has_genie = is_nu & g["pdg"].notna()
    if (is_nu & ~has_genie).any():
        warnings.warn(f"_build_truth_info: {(is_nu & ~has_genie).sum()} reco interactions "
                      "have nu_id >= 0 but no rec.mc.nu entry -> truth_cat 6 (other nu)")

    cat = pd.Series(6, index=idx, dtype=np.int8)    # other nu (incl. no rec.mc.nu entry)
    cat[~is_nu]    = 7                               # cosmic
    cat[has_genie] = g.loc[has_genie, "truth_cat"].astype(np.int8)

    # ── One is_true_signal row per signal neutrino ────────────────────────────
    # (a neutrino split into several reco interactions keeps truth_cat 0 on
    #  all of them, but only the one that got furthest counts in the
    #  denominator)
    sig_rows = pd.DataFrame({
        "_entry": idx.get_level_values(0), "_nu": nu_id.values,
        "rank":  2 * passed_presel.values + passed_fm.values,
    }, index=idx)[(cat == 0).values]
    rep = (sig_rows.sort_values("rank", ascending=False, kind="stable")
           .groupby(["_entry", "_nu"]).head(1).index)
    is_true_signal = pd.Series(idx.isin(rep), index=idx)

    out = pd.DataFrame({
        "truth_cat"              : cat,
        "is_true_signal"         : is_true_signal,
        "is_nu"                  : is_nu,
        "is_cc"                  : has_genie & g["iscc"].fillna(False).astype(bool),
        "is_fv_true"             : has_genie & g["is_fv_true"].fillna(False).astype(bool),
        "has_true_electron"      : has_genie & g["has_true_electron"].fillna(False).astype(bool),
        "has_true_muon"          : has_genie & g["has_mu"].fillna(False).astype(bool),
        "has_pi0"                : has_genie & g["has_pi0"].fillna(False).astype(bool),
        "passed_fm"              : passed_fm,
        "passed_fv_reco"         : passed_fv_reco,
        "passed_presel"          : passed_presel,
        "true_leading_e_ke"      : g["lead_e_ke"],
        "true_leading_e_costheta": g["lead_e_costheta"],
        "true_leading_e_p"       : g["lead_e_p"],
        "mct_index"              : nu_id,        # rec.mc.nu index (see module docstring)
        "nu_pdg"                 : g["pdg"].fillna(0),
        "genie_mode"             : g["genie_mode"].fillna(-1),
        "true_nu_e"              : g["E"],
        "n_true_e"               : g["n_e_above"].fillna(0),
        "is_truth_only"          : False,
    }, index=idx)

    # ── Truth-only rows: signal neutrinos with no reco interaction ────────────
    matched = pd.MultiIndex.from_arrays([idx.get_level_values(0)[is_nu],
                                         nu_id.values[is_nu]])
    gs = gi[(gi["truth_cat"] == 0) & ~gi.index.isin(matched)]
    if len(gs) > 0:
        nu_i = gs.index.get_level_values(1).astype(np.int32)
        miss = pd.DataFrame({
            "truth_cat"              : np.int8(0),
            "is_true_signal"         : True,
            "is_nu"                  : True,
            "is_cc"                  : True,
            "is_fv_true"             : True,
            "has_true_electron"      : True,
            "has_true_muon"          : gs["has_mu"].values,
            "has_pi0"                : gs["has_pi0"].values,
            "passed_fm"              : False,
            "passed_fv_reco"         : False,
            "passed_presel"          : False,
            "true_leading_e_ke"      : gs["lead_e_ke"].values,
            "true_leading_e_costheta": gs["lead_e_costheta"].values,
            "true_leading_e_p"       : gs["lead_e_p"].values,
            "mct_index"              : nu_i,
            "nu_pdg"                 : gs["pdg"].values,
            "genie_mode"             : gs["genie_mode"].values,
            "true_nu_e"              : gs["E"].values,
            "n_true_e"               : gs["n_e_above"].values,
            "is_truth_only"          : True,
        }, index=pd.MultiIndex.from_arrays(
            [gs.index.get_level_values(0), -1 - nu_i], names=idx.names))
        out = pd.concat([out, miss])

    return out.astype({
        "truth_cat"              : np.int8,
        "is_true_signal"         : bool,
        "is_nu"                  : bool,
        "is_cc"                  : bool,
        "is_fv_true"             : bool,
        "has_true_electron"      : bool,
        "has_true_muon"          : bool,
        "has_pi0"                : bool,
        "passed_fm"              : bool,
        "passed_fv_reco"         : bool,
        "passed_presel"          : bool,
        "true_leading_e_ke"      : np.float32,
        "true_leading_e_costheta": np.float32,
        "true_leading_e_p"       : np.float32,
        "mct_index"              : np.int32,
        "nu_pdg"                 : np.int32,
        "genie_mode"             : np.int32,
        "true_nu_e"              : np.float32,
        "n_true_e"               : np.int16,
        "is_truth_only"          : bool,
    })


def _get_truth_info(f):
    fname = str(f)
    if fname not in _truth_info_cache:
        # ── Load flash_times (ragged branch, not in spine_df) ────────────────
        flash_times_s = None
        if _loadbranches is not None and _SPINE_FT_BRANCHES is not None:
            try:
                raw_ft = _loadbranches(f["recTree"], _SPINE_FT_BRANCHES)
                ft_col        = raw_ft.columns[0]
                flash_times_s = raw_ft[ft_col]
            except Exception as e:
                warnings.warn(
                    f"_get_truth_info: flash_times load failed, "
                    f"falling back to flash_total_pe proxy — {e}"
                )

        try:
            genie = _load_genie_truth(f)
        except Exception as e:
            # e.g. cosmic-only MC with an empty rec.mc.nu; any reco interaction
            # with nu_id >= 0 then lands in truth_cat 6 with a warning
            warnings.warn(f"_get_truth_info: GENIE truth (rec.mc.nu) load failed — {e}")
            genie = _empty_genie_truth()

        _truth_info_cache[fname] = _build_truth_info(
            _get_spine_df(f), genie, flash_times_s)
    return _truth_info_cache[fname]


# ── Public maker 1: MC evtdf ─────────────────────────────────────────────────

def make_nuecc_evtdf(f):
    """
    Reco+truth SPINE particle-level df, FM+FV preselected, merged with MC CV.

    The interaction-level rec.dlp_true.mct_index column is overwritten with
    rec.dlp_true.nu_id (the rec.mc.nu index), so (entry, mct_index) is the
    correct key for the mcnu merge here and for the weight alignment
    downstream (nue_helpers.align_weights / load_mcnu_full_selected).
    """
    spine_df   = _get_spine_df(f)
    truth_info = _get_truth_info(f)

    if not truth_info.empty:
        presel_idx = truth_info.index[truth_info["passed_presel"]]
        spine_df   = spine_df[spine_df.index.droplevel(-1).isin(presel_idx)].copy()

    try:
        mct_col = _find_col_ends(spine_df, BRANCH_TRUE, "mct_index")
        nu_col  = _find_col(spine_df, "nu_id", branch_must_contain=BRANCH_TRUE)
        spine_df[mct_col] = spine_df[nu_col].fillna(-1).astype(np.int32)

        mcdf    = make_mcnudf(f, include_weights=False)
        mcdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["mcnu"] + list(c)) for c in mcdf.columns]
        )
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

    return _drop_unused_fields(spine_df)


# ── Public maker 2: data evtdf (reco-only) ───────────────────────────────────

def make_nuecc_evtdf_data(f):
    """
    Reco-only SPINE particle-level df, FM+FV preselected.

    No MC truth merge, no truth categorization.  Intended for data CAFs where
    rec.dlp_true branches are absent.  Uses _get_spine_df_data which builds
    the spine df from reco-only branch lists (spineint + spinepart), avoiding
    the inner join on truth branches that make_all_spine_df performs.

    The FM + reco-FV preselection is identical to the MC path.
    """
    spine_df = _get_spine_df_data(f)

    if spine_df.empty:
        warnings.warn("make_nuecc_evtdf_data: spine_df is empty after reco-only load")
        return spine_df

    # ── Load flash_times for full C++ FM condition ────────────────────────────
    flash_times_s = None
    if _loadbranches is not None and _SPINEINT_FLASHTIMES is not None:
        try:
            raw_ft        = _loadbranches(f["recTree"], _SPINEINT_FLASHTIMES)
            ft_col        = raw_ft.columns[0]
            flash_times_s = raw_ft[ft_col]
        except Exception as e:
            warnings.warn(
                f"make_nuecc_evtdf_data: flash_times load failed, "
                f"falling back to flash_total_pe proxy — {e}"
            )

    passed_fm, passed_fv_reco, idx, il = _apply_presel(spine_df, flash_times_s)
    _data_presel_cache.clear()
    _data_presel_cache[str(f)] = (passed_fm, passed_fv_reco)

    presel_idx = idx[passed_fm & passed_fv_reco]
    n_total  = len(idx)
    n_presel = len(presel_idx)
    print(f"  data evtdf: {n_presel} / {n_total} interactions pass FM+FV presel")

    spine_df = spine_df[spine_df.index.droplevel(-1).isin(presel_idx)]
    return _drop_unused_fields(spine_df)


# ── Public maker 3: truth_info_df ────────────────────────────────────────────

def make_nuecc_truth_info_df(f):
    """
    Per-interaction truth + reco-preselection metadata covering ALL interactions
    before any FM/FV cut.  Loaded by nue_selection.ipynb as truth_info_0.
    """
    return _get_truth_info(f)


def make_nuecc_statsdf(f):
    """Backward-compatible alias for make_nuecc_truth_info_df."""
    return make_nuecc_truth_info_df(f)


# ── Public maker 3b: flash-match table (all interactions) ────────────────────

def make_nuecc_flashdf(f):
    """MC: per-interaction flash-match table for ALL reco interactions (before
    FM / FV), with truth_cat; cosmic-only rows prescaled (see _build_flash_table).
    Uses the same spine df (makedf.make_all_spine_df) and the same FM / FV flags as
    truth_info / evt. Must run after make_nuecc_truth_info_df in the same file
    (shares its caches)."""
    ti = _get_truth_info(f)
    if ti.empty:
        return pd.DataFrame()
    ti = ti[~ti["is_truth_only"]]
    return _build_flash_table(_get_spine_df(f), f["recTree"],
                              ti["passed_fm"], ti["passed_fv_reco"], truth_cat=ti["truth_cat"])


def make_nuecc_flashdf_data(f):
    """Data: per-interaction flash-match table for ALL reco interactions (before
    FM / FV). Uses the FM / FV flags of make_nuecc_evtdf_data for the same file
    (computed again if that maker did not run first)."""
    spine_df = _get_spine_df_data(f)
    if spine_df.empty:
        return pd.DataFrame()
    key = str(f)
    if key in _data_presel_cache:
        passed_fm, passed_fv_reco = _data_presel_cache[key]
    else:
        flash_times_s = None
        if _loadbranches is not None and _SPINEINT_FLASHTIMES is not None:
            try:
                raw_ft = _loadbranches(f["recTree"], _SPINEINT_FLASHTIMES)
                flash_times_s = raw_ft[raw_ft.columns[0]]
            except Exception as e:
                warnings.warn(f"make_nuecc_flashdf_data: flash_times load failed — {e}")
        passed_fm, passed_fv_reco, _, _ = _apply_presel(spine_df, flash_times_s)
    return _build_flash_table(spine_df, f["recTree"], passed_fm, passed_fv_reco)


# Extra cross-section dials (module level: used by both wgtdf makers)
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


# ── Public maker 4: wgtdf ────────────────────────────────────────────────────

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
    _spine_cache.clear()
    _truth_info_cache.clear()
    del truth_info, presel_mct
    _force_free()

    # ── Step 3: build presel_ind ─────────────────────────────────────────────
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
    print(f"  wgtdf: {n_presel} preselected MC nu rows -> passing to syst functions")

    if n_presel == 0:
        warnings.warn("make_nuecc_wgtdf: no mcdf rows matched preselected pairs")
        return pd.DataFrame()

    presel_ind = pd.Series(
        np.asarray(presel_mcdf_index.get_level_values(-1), dtype=np.int64),
        index=presel_mcdf_index,
    )

    # ── Step 4: BNB and GENIE weights for preselected nus only ───────────────
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

    try:
        from makedf import g4syst as g4syst_mod
        g4_wgt = g4syst_mod.g4syst(f, presel_ind)
        if g4_wgt is not None and not g4_wgt.empty:
            print(f"  G4: {g4_wgt.shape[1]} cols, {len(g4_wgt)} rows  "
                  f"({g4_wgt.memory_usage(deep=True).sum()/1e9:.3f} GB)")
            if out is None:
                out = g4_wgt
            else:
                out = multicol_concat(out, g4_wgt)
                del g4_wgt
        elif g4_wgt is not None:
            del g4_wgt
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf: G4 failed — {e}")
    _force_free()

    del presel_ind, presel_mcdf_index
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()

    print(f"  wgtdf output: {len(out)} rows x {out.shape[1]} cols  "
          f"({out.memory_usage(deep=True).sum()/1e9:.3f} GB)")
    return out


def make_nuecc_wgtdf_full(f):
    """
    Per-dial (slim=False) weights for the breakdown plots.
    Compact: multisim dials keep 100 univs; +/-sigma dials keep only ps1;
    morph dials keep their single column. Preselected nus only.

    Returns a frame with the SAME index as make_nuecc_wgtdf, columns are
    MultiIndex (dial_name, variation).
    """
    from makedf import bnbsyst, geniesyst
    from makedf.getsyst import getsyst

    # ── Steps 1-3: identical preselection → presel_ind ───────────────────────
    truth_info = _get_truth_info(f)
    if truth_info.empty:
        return pd.DataFrame()
    presel_mct = truth_info.loc[truth_info["passed_presel"], "mct_index"]
    presel_mct = presel_mct[presel_mct >= 0]
    if presel_mct.empty:
        return pd.DataFrame()

    pair_df = pd.DataFrame({
        "entry": np.asarray(presel_mct.index.get_level_values(0)),
        "mct":   presel_mct.astype(np.int64).values,
    }).drop_duplicates().reset_index(drop=True)

    _spine_cache.clear(); _truth_info_cache.clear()
    del truth_info, presel_mct
    _force_free()

    mcdf = make_mcnudf(f, include_weights=False)
    mcdf_pair_idx = pd.MultiIndex.from_arrays(
        [mcdf.index.get_level_values(0),
         np.asarray(mcdf.index.get_level_values(-1), dtype=np.int64)],
        names=["entry", "mct"])
    presel_pair_idx = pd.MultiIndex.from_arrays(
        [pair_df["entry"].values, pair_df["mct"].values], names=["entry", "mct"])
    keep_mask = mcdf_pair_idx.isin(presel_pair_idx)
    presel_mcdf_index = mcdf.index[keep_mask]
    del mcdf, mcdf_pair_idx, presel_pair_idx, keep_mask, pair_df
    gc.collect()

    if len(presel_mcdf_index) == 0:
        return pd.DataFrame()

    presel_ind = pd.Series(
        np.asarray(presel_mcdf_index.get_level_values(-1), dtype=np.int64),
        index=presel_mcdf_index)

    # ── Helper: shrink a slim=False frame to compact per-dial columns ────────
    def _compact(df):
        """Keep multisim univ_* in full; reduce +/-sigma to ps1; morph as-is."""
        if df is None or df.empty:
            return None
        keep_cols = []
        for name in dict.fromkeys(c[0] for c in df.columns):  # ordered unique
            dcols = [c for c in df.columns if c[0] == name]
            seconds = [c[1] for c in dcols]
            if all('univ_' in s for s in seconds):            # multisim → keep all
                keep_cols.extend(dcols)
            elif 'ps1' in seconds:                            # +/-sigma → ps1 only
                keep_cols.append((name, 'ps1'))
            elif 'morph' in seconds:                          # morph → keep
                keep_cols.append((name, 'morph'))
            # (cv columns and ps2/ms2/... are dropped)
        out = df[keep_cols].copy()
        # downcast to float32 to halve memory
        return out.astype(np.float32)

    out = None
    for fn, kw in [
        (lambda: bnbsyst.bnbsyst(f, presel_ind, multisim_nuniv=100, slim=False), {}),
        (lambda: geniesyst.geniesyst(f, presel_ind, multisim_nuniv=100, slim=False), {}),
        (lambda: getsyst(f, EXTRA_XSEC_SYSTS, presel_ind,
                         multisim_nuniv=100, slim=False), {}),
    ]:
        try:
            w = _compact(fn())
            if w is not None and not w.empty:
                out = w if out is None else multicol_concat(out, w)
                del w
        except Exception as e:
            warnings.warn(f"make_nuecc_wgtdf_full: sub-syst failed — {e}")
        _force_free()

    # g4: already per-channel and small; keep as-is (3×100)
    try:
        from makedf import g4syst as g4syst_mod
        g4 = g4syst_mod.g4syst(f, presel_ind)
        if g4 is not None and not g4.empty:
            g4 = g4.astype(np.float32)
            out = g4 if out is None else multicol_concat(out, g4)
            del g4
    except Exception as e:
        warnings.warn(f"make_nuecc_wgtdf_full: G4 failed — {e}")
    _force_free()

    del presel_ind, presel_mcdf_index
    gc.collect()

    if out is None or out.empty:
        return pd.DataFrame()
    print(f"  wgtdf_full: {len(out)} rows x {out.shape[1]} cols  "
          f"({out.memory_usage(deep=True).sum()/1e9:.3f} GB)")
    return out