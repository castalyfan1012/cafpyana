"""
nue_selection.py
----------------
nueCC inclusive selection for SPINE DLP.

Truth (signal / categories) comes from truth_info, written by
makedf/make_nueCC_df.py from the GENIE neutrino (rec.mc.nu) matched to each
reco interaction through rec.dlp_true.nu_id:
    0 nueCC FV, exactly 1 primary e (KE >= 500 MeV)   1 nueCC out FV
    2 numuCC (mu > 50 MeV) + pi0                       3 NC pi0
    4 other numuCC                                      5 other NC
    6 other nu                                          7 cosmic
is_true_signal marks ONE row per true signal neutrino (the efficiency
denominator; truth-only rows are added for unreconstructed signal), while
truth_cat == 0 marks every reco interaction matched to a signal neutrino
(the purity numerator).  Use stage_counts / signal_selected_mask below for
cut-flow purity and efficiency so split neutrinos are counted once.

Fiducial volume (fiducial_cut_tmp — matches C++ definition)
------------------------------------------------------------
    x_region  = (|x| > 5) & (|x| < 190)
    z_region1 = (10 < z < 250) & (-190 < y < 190)
    z_region2 = (250 < z < 450) & (-190 < y < 100) & (x < 0)   [TPC 0]
    z_region3 = (250 < z < 450) & (-190 < y < 190) & (x > 0)   [TPC 1]

valid_flashmatch (matches C++ definition)
-----------------------------------------
    flash_times.size() > 0  AND  is_flash_matched == 1  AND  NOT isnan(flash_times[0])

    build_cut_flow accepts an optional flash_times_df (3-level MultiIndex Series:
    entry / dlp_inter_idx / flash_entry_idx, from rec.dlp.flash_times).
    When supplied the full condition above is applied.
    When None, falls back to the proxy:  is_flash_matched == 1  AND  flash_total_pe > 0
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from nue_helpers import (
    reco_particles, true_particles,
    reco_interactions, true_interactions,
    inter_levels,
)

# ── Thresholds (MeV) — must match make_nueCC_df.py ───────────────────────────
ELECTRON_THRESHOLD_MEV = 500.0
MUON_THRESHOLD_MEV     = 50.0
PHOTON_THRESHOLD_MEV   = 25.0
PION_THRESHOLD_MEV     = 25.0
PROTON_THRESHOLD_MEV   = 50.0

# PID codes
PID_PHOTON   = 0
PID_ELECTRON = 1
PID_MUON     = 2
PID_PION     = 3
PID_PROTON   = 4

# ── Cut names / labels ────────────────────────────────────────────────────────
CUT_NAMES = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    "single_electron",
]
CUT_LABELS = [
    "No cut",
    "Flash match",
    "In FV",
    "Final state topology",
]

# ── Shower-quality cuts ──────────────────────────────────────────────────────
# Thresholds from the cumulative threshold scan (nh.threshold_plots_cumulative).
# The THRESH_* names are kept because other cells/modules may import them.
THRESH_PRIMARY_SCORE = 0.99     # leading e primary softmax (I1)   >
THRESH_PID_SCORE     = 0.895      # leading e PID softmax (I1)       >
THRESH_VERTEX_DIST   = 3.95      # conversion gap [cm]              <
THRESH_DEDX          = 3.20      # start dE/dx upper edge [MeV/cm]  <   ← re-check after re-scan
THRESH_DEDX_LO       = 1.28      # start dE/dx lower edge [MeV/cm]  >   (0 = no lower edge)
THRESH_DIR_SPREAD    = 0.50      # directional spread               <   (not used)
THRESH_AXIAL_SPREAD  = 0.00      # axial spread                     >   (not used)

# Registry of every shower cut the code knows how to apply:
#   cut name : (leading-electron column, direction, threshold, table label)
#   direction ">" / "<" with a number, or "<>" with (lo, hi) for a window lo < x < hi
# A cut is APPLIED only if its name is listed in CUT_NAMES_MORE below.
SHOWER_CUTS = {
    "electron_primary_score": (("primary_scores", "I1"),   ">", THRESH_PRIMARY_SCORE, "Electron primary score"),
    "electron_pid_score"    : (("pid_scores", "I1"),       ">", THRESH_PID_SCORE,     "Electron PID score"),
    "vertex_distance"       : (("vertex_distance", ""),    "<", THRESH_VERTEX_DIST,   "Conversion gap"),
    "directional_spread"    : (("directional_spread", ""), "<", THRESH_DIR_SPREAD,    "Directional spread"),
    "axial_spread"          : (("axial_spread", ""),       ">", THRESH_AXIAL_SPREAD,  "Axial spread"),
    "start_dedx"            : (("start_dedx", ""),         "<>", (THRESH_DEDX_LO, THRESH_DEDX), "Start dE/dx"),
}

# Cuts applied, in order.  Comment a line out to drop that cut — nothing else
# needs to change (labels, cut flow and tables all follow this list).
CUT_NAMES_MORE = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    "single_electron",
    "electron_primary_score",
    "electron_pid_score",
    "vertex_distance",
    # "directional_spread",
    # "axial_spread",
    "start_dedx",
]

CUT_LABEL_MAP = dict(zip(CUT_NAMES, CUT_LABELS))
CUT_LABEL_MAP.update({k: v[3] for k, v in SHOWER_CUTS.items()})

_unknown = [c for c in CUT_NAMES_MORE if c not in CUT_LABEL_MAP]
if _unknown:
    raise ValueError(f"CUT_NAMES_MORE has cuts not defined in SHOWER_CUTS: {_unknown}")
CUT_LABELS_MORE = [CUT_LABEL_MAP[c] for c in CUT_NAMES_MORE]

THRESH_SB_DEDX_LO = 3.0   # MeV/cm — sideband dE/dx lower bound
THRESH_SB_DEDX_HI = 6.0   # MeV/cm — sideband dE/dx upper bound


# ── Fiducial volume ───────────────────────────────────────────────────────────

def _fiducial_cut_tmp(x, y, z):
    """
    SBND FV: fiducial_cut_tmp from SPINE analysis framework (updated based on SBND DENT force May 2026).

    x_region  = (|x| > 5) & (|x| < 190)
    z_region1 = (10 < z < 250) & (-190 < y < 190)
    z_region2 = (250 < z < 450) & (-190 < y < 100) & (x < 0)
    z_region3 = (250 < z < 450) & (-190 < y < 190) & (x > 0)
    contained = x_region & (z_region1 | z_region2 | z_region3)

    x, y, z: pandas Series (vertex coordinates in cm).
    Returns: boolean Series.
    """
    abs_x     = x.abs()
    x_region  = (abs_x > 5) & (abs_x < 190)
    z_region1 = (z > 10)  & (z < 250) & (y > -190) & (y < 190)
    z_region2 = (z > 250) & (z < 450) & (y > -190) & (y < 100) & (x < 0)
    z_region3 = (z > 250) & (z < 450) & (y > -190) & (y < 190) & (x > 0)
    return x_region & (z_region1 | z_region2 | z_region3)


# ── Public particle-level masks ───────────────────────────────────────────────

def primary_electron_mask(evtdf):
    """Particle-level mask: good primary reco electrons above threshold."""
    rp = reco_particles(evtdf)
    return (
        (rp.is_primary == 1) &
        (rp.pid == PID_ELECTRON) &
        (rp.ke > ELECTRON_THRESHOLD_MEV)
    )


def primary_muon_mask(evtdf):
    """Particle-level mask: good primary reco muons above threshold."""
    rp = reco_particles(evtdf)
    return (
        (rp.is_primary == 1) &
        (rp.pid == PID_MUON) &
        (rp.ke > MUON_THRESHOLD_MEV)
    )


# ── Truth classification ──────────────────────────────────────────────────────

def classify_truth(evtdf, verbose=True):
    """
    Approximate truth category per reconstructed interaction from the
    reco-matched dlp_true particles in evtdf.

    Only a fallback for plotting helpers when no truth categories are passed.
    The analysis uses truth_info['truth_cat'] from make_nueCC_df.py, which is
    built from the GENIE neutrino (no reco-fragment duplicates, no missing
    unreconstructed muons) — the two can differ for a few % of interactions.

    Categories (same numbering as truth_info, first match wins)
    ------------------------------------------------------------
    0  nueCC FV (signal)   1  nueCC out FV
    2  numuCC + pi0         3  NC pi0
    4  other numuCC         5  other NC
    6  other nu             7  cosmic
    """
    il  = inter_levels(evtdf)
    idx = evtdf.groupby(level=il).size().index

    if verbose:
        print("classify_truth: reading true-interaction flags …")

    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)

    is_nu = (ti.nu_id >= 0).groupby(level=il).any().reindex(idx, fill_value=False)
    is_cc = (ti.current_type == 0).groupby(level=il).any().reindex(idx, fill_value=False)

    # FV: geometric cut on true vertex (fiducial_cut_tmp)
    is_fv = (
        _fiducial_cut_tmp(ti.vertex.x, ti.vertex.y, ti.vertex.z)
        .groupby(level=il).any()
        .reindex(idx, fill_value=False)
    )

    true_elec = (
        (tp.pid == PID_ELECTRON) &
        (tp.is_primary == 1) &
        (tp.ke > ELECTRON_THRESHOLD_MEV)
    ).groupby(level=il).sum() == 1

    true_muon = (
        (tp.pid == PID_MUON) &
        (tp.is_primary == 1) &
        (tp.ke > MUON_THRESHOLD_MEV)
    ).groupby(level=il).any().reindex(idx, fill_value=False)

    has_pi0 = (
        tp.parent_pdg_code.abs() == 111
    ).groupby(level=il).any().reindex(idx, fill_value=False)

    if verbose:
        print(f"  is_nu        : {is_nu.sum():>8,}")
        print(f"  is_cc        : {is_cc.sum():>8,}")
        print(f"  is_fv (true) : {is_fv.sum():>8,}")
        print(f"  true_elec    : {true_elec.sum():>8,}")
        print(f"  true_muon    : {true_muon.sum():>8,}")
        print(f"  has_pi0      : {has_pi0.sum():>8,}")

    true_elec = true_elec.reindex(idx, fill_value=False)
    nu  = is_nu.to_numpy()
    cat = pd.Series(np.select(
        [~nu,
         nu &  is_cc &  is_fv & true_elec,
         nu &  is_cc & ~is_fv & true_elec,
         nu &  is_cc & true_muon & has_pi0,
         nu & ~is_cc & has_pi0,
         nu &  is_cc & true_muon,
         nu & ~is_cc],
        [7, 0, 1, 2, 3, 4, 5], default=6), index=idx, dtype=int)

    if verbose:
        labels = {
            0: "nueCC FV (signal)",  1: "nueCC out FV",
            2: "numuCC + pi0",       3: "NC pi0",
            4: "other numuCC",       5: "other NC",
            6: "other neutrino",     7: "cosmic",
        }
        print("\n=== Truth category summary ===")
        for c, lbl in labels.items():
            print(f"  cat {c:2d}  {lbl:<22s}: {(cat == c).sum():>8,}")
        print(f"  {'total':<26s}: {len(cat):>8,}")

    return cat


# ── Full cut flow ─────────────────────────────────────────────────────────────

def build_cut_flow(evtdf, flash_times_df=None):
    """
    Apply the nueCC selection sequentially.

    Parameters
    ----------
    evtdf          : merged SPINE reco+truth particle-level DataFrame.
    flash_times_df : optional Series with 3-level MultiIndex
                     (entry, dlp_inter_idx, flash_entry_idx) containing
                     rec.dlp.flash_times values.  When supplied the full C++
                     valid_flashmatch condition is applied:
                         flash_times.size()>0 && is_flash_matched==1
                         && !isnan(flash_times[0])
                     When None, falls back to the proxy:
                         is_flash_matched==1 AND flash_total_pe>0

    Cuts
    ----
    valid_flashmatch  see above
    fiducial          fiducial_cut_tmp geometric cut on reco vertex
    single_electron   >= 1 primary reco electron above threshold

    Returns
    -------
    dict keyed by CUT_NAMES; each value:
        {"inter_index": pd.Index, "particle_mask": pd.Series}
    """
    il        = inter_levels(evtdf)
    all_inter = evtdf.groupby(level=il).size().index
    results   = {}

    def _pmask(idx):
        return pd.Series(evtdf.index.droplevel(-1).isin(idx), index=evtdf.index)

    steps   = tqdm(CUT_NAMES, desc="Building cut flow", leave=True)
    current = all_inter

    for step in steps:
        steps.set_postfix(n=len(current))

        if step == "precut":
            results["precut"] = {
                "inter_index":   current,
                "particle_mask": pd.Series(True, index=evtdf.index),
            }

        elif step == "valid_flashmatch":
            ri = reco_interactions(evtdf)

            if flash_times_df is not None:
                # Full C++ condition:
                #   flash_times.size()>0 && is_flash_matched==1 && !isnan(flash_times[0])
                #
                # flash_times_df has a 3-level MultiIndex:
                #   (entry, dlp_inter_idx, flash_entry_idx)
                # groupby on the first (nlevels-1) levels collapses to
                # interaction level. .first().notna() captures both size()>0
                # and !isnan(flash_times[0]) in one step; interactions absent
                # from flash_times_df (empty vector) are filled False.
                ft_il        = list(range(flash_times_df.index.nlevels - 1))
                first_ft     = flash_times_df.groupby(level=ft_il).first()
                has_valid_ft = first_ft.notna()

                fm_matched   = (ri.is_flash_matched == 1).groupby(level=il).any()
                has_valid_ft = has_valid_ft.reindex(fm_matched.index, fill_value=False)
                fm_inter     = fm_matched & has_valid_ft
            else:
                # Proxy fallback: is_flash_matched==1 AND flash_total_pe>0
                fm_mask  = (ri.is_flash_matched == 1) & (ri.flash_total_pe > 0)
                fm_inter = fm_mask.groupby(level=il).any()

            current = current.intersection(fm_inter[fm_inter].index)
            results["valid_flashmatch"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }

        elif step == "fiducial":
            ri = reco_interactions(evtdf)
            ri_df = ri._df
        
            # Try standard x/y/z columns (MC path: vertex.x/y/z)
            # Fall back to I0/I1/I2 columns (data path: vertex.I0/I1/I2)
            def _get_vertex_coord(ri_df, coord):
                """Get vertex coordinate Series, handling both MC and data column naming."""
                # MC: attribute access ri.vertex.x works → ('vertex', 'x', '') or similar
                # Data: columns are ('vertex', 'I0', ''), ('vertex', 'I1', ''), ('vertex', 'I2', '')
                coord_to_idx = {'x': 'I0', 'y': 'I1', 'z': 'I2'}
                # Try direct MultiIndex lookup for data naming
                fallback_key = (coord_to_idx[coord], '')
                for c in ri_df.columns:
                    if isinstance(c, tuple):
                        if 'vertex' in c and coord in c:
                            return ri_df[c]
                        if 'vertex' in c and coord_to_idx[coord] in c:
                            return ri_df[c]
                raise KeyError(f"vertex {coord} not found in reco_interactions columns")
        
            try:
                # MC path: ri.vertex.x/y/z via attribute chaining
                vx = ri.vertex.x
                vy = ri.vertex.y
                vz = ri.vertex.z
            except AttributeError:
                # Data path: vertex columns named I0/I1/I2
                vx = _get_vertex_coord(ri_df, 'x')
                vy = _get_vertex_coord(ri_df, 'y')
                vz = _get_vertex_coord(ri_df, 'z')
        
            fv_mask  = _fiducial_cut_tmp(vx, vy, vz)
            fv_inter = fv_mask.groupby(level=il).any()
            current  = current.intersection(fv_inter[fv_inter].index)
            results["fiducial"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }

        elif step == "single_electron":
            elec_counts = primary_electron_mask(evtdf).groupby(level=il).sum()
            elec_inter  = elec_counts[elec_counts >= 1].index
            current     = current.intersection(elec_inter)
            results["single_electron"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }

    return results


# ── Leading PRIMARY electron (the one that defines the topology) ─────────────

def leading_primary_electron(evtdf):
    """
    One row per interaction: the highest-KE primary reco electron above
    ELECTRON_THRESHOLD_MEV — the same particle used by single_electron and by
    leading_electron_ke / _p / _costheta.  Index = interaction levels.

    (The old shower_qual_cuts took the highest-KE reco electron of ANY kind,
    which can be a non-primary fragment, so the quality cuts could be applied
    to a different particle than the one being measured.)
    """
    il     = inter_levels(evtdf)
    rp_df  = reco_particles(evtdf)._df
    m      = primary_electron_mask(evtdf)
    ke     = rp_df[('ke', '')][m]
    lead_i = ke.groupby(level=il).idxmax().dropna()
    lead_mi = pd.MultiIndex.from_tuples(list(lead_i.values), names=rp_df.index.names)
    lead   = rp_df.loc[lead_mi].copy()
    lead.index = lead_i.index
    return lead


def _shower_cut_pass(step, lead):
    """Interaction-level boolean Series: True = leading primary electron passes `step`."""
    if step not in SHOWER_CUTS:
        raise ValueError(f"Unknown shower quality cut: {step!r} "
                         f"(defined: {list(SHOWER_CUTS)})")
    col, cdir, thr, _ = SHOWER_CUTS[step]
    if col not in lead.columns:
        raise KeyError(f"{step}: column {col} not in reco particle columns")
    v = lead[col]                                       # NaN fails every cut
    if cdir == "<>":
        lo, hi = thr
        return (v > lo) & (v < hi)
    if cdir not in (">", "<"):
        raise ValueError(f"{step}: unknown cut direction {cdir!r}")
    return (v > thr) if cdir == ">" else (v < thr)


# ── Extended cut flow (shower quality) ───────────────────────────────────────

def shower_qual_cuts(evtdf, flash_times_df=None):
    """
    Full cut-flow including the cuts after single_electron listed in
    CUT_NAMES_MORE.  Shower variables are taken from the leading PRIMARY
    electron (see leading_primary_electron).
    """
    cut_flow = build_cut_flow(evtdf, flash_times_df=flash_times_df)
    row_inter = evtdf.index.droplevel(-1)          # computed once

    def _pmask(idx):
        return pd.Series(row_inter.isin(idx), index=evtdf.index)

    additional_cuts = CUT_NAMES_MORE[CUT_NAMES_MORE.index("single_electron") + 1:]
    if not additional_cuts:
        return cut_flow

    lead    = leading_primary_electron(evtdf)
    steps   = tqdm(additional_cuts, desc="Shower quality cuts", leave=True)
    current = cut_flow["single_electron"]["inter_index"]

    for step in steps:
        steps.set_postfix(n=len(current))
        ok      = _shower_cut_pass(step, lead)
        current = current.intersection(ok[ok].index)
        cut_flow[step] = {
            "inter_index":   current,
            "particle_mask": _pmask(current),
        }

    return cut_flow


# ── True-signal denominator ───────────────────────────────────────────────────

def true_signal_index(evtdf):
    """Return interaction index of reco interactions matched to true signal
    (reco-matched approximation, see classify_truth).

    NOT an efficiency denominator: evtdf only holds preselected interactions.
    Use truth_info['is_true_signal'] (one row per true signal neutrino).
    """
    il = inter_levels(evtdf)
    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)

    is_nu = (ti.nu_id >= 0).groupby(level=il).any()
    is_cc = (ti.current_type == 0).groupby(level=il).any()
    is_fv = (
        _fiducial_cut_tmp(ti.vertex.x, ti.vertex.y, ti.vertex.z)
        .groupby(level=il).any()
    )

    prim_elec = (
        (tp.pid == PID_ELECTRON) &
        (tp.is_primary == 1) &
        (tp.ke > ELECTRON_THRESHOLD_MEV)
    ).groupby(level=il).sum() == 1

    sig = is_nu & is_cc & is_fv & prim_elec
    return sig[sig].index


# ── Purity / efficiency from truth_info (neutrino-level signal) ──────────────

def signal_nu_keys(truth_info):
    """(file, entry, nu_id) per truth_info row — all index levels but the
    interaction one, plus nu_id — used to group the reco interactions of one
    neutrino.  nu_id is the 'mct_index' column, which make_nueCC_df.py fills
    with rec.dlp_true.nu_id."""
    if "mct_index" not in truth_info.columns:
        raise KeyError("truth_info has no 'mct_index' (nu_id) column — "
                       "load it (TRUTH_COLS) or remake the dfs")
    idx = truth_info.index
    lvls = [idx.get_level_values(i) for i in range(idx.nlevels - 1)]
    return pd.MultiIndex.from_arrays(lvls + [truth_info["mct_index"].to_numpy()])


def signal_selected_mask(truth_info, stage_idx):
    """Bool array aligned to truth_info: True on the is_true_signal row of every
    true signal neutrino with at least one truth_cat==0 interaction in stage_idx.

    The is_true_signal row is chosen per neutrino by make_nueCC_df.py (presel >
    FM rank), so it is not always the reco interaction that survives a later cut
    when a neutrino is split into several interactions; this lifts the
    selection to the neutrino level.
    """
    sig_row = truth_info["is_true_signal"].to_numpy(dtype=bool)
    in_stage = truth_info.index.isin(stage_idx)
    if "mct_index" not in truth_info.columns:          # old dfs: representative row only
        return sig_row & in_stage
    keys = signal_nu_keys(truth_info)
    cat0 = truth_info["truth_cat"].to_numpy() == 0
    sel_keys = keys[cat0 & in_stage].unique()
    return sig_row & keys.isin(sel_keys)


def stage_counts(truth_info, stage_idx):
    """Cut-flow numbers for one stage.

    n_sel   : reco interactions selected
    n_sig   : selected interactions with truth_cat == 0 (purity numerator,
              same counting as the stacked histograms)
    n_nu    : true signal neutrinos with >= 1 selected signal interaction
              (efficiency numerator)
    purity  : n_sig / n_sel            (everything selected in the denominator)
    eff     : n_nu / N true signal     (all true signal neutrinos, pre-cut)
    """
    reco = ~truth_info["is_truth_only"].to_numpy(dtype=bool) \
        if "is_truth_only" in truth_info.columns else np.ones(len(truth_info), bool)
    in_stage = truth_info.index.isin(stage_idx) & reco
    n_sel = int(in_stage.sum())
    n_sig = int((in_stage & (truth_info["truth_cat"].to_numpy() == 0)).sum())
    n_nu  = int(signal_selected_mask(truth_info, stage_idx).sum())
    n_den = int(truth_info["is_true_signal"].sum())
    return dict(n_sel=n_sel, n_sig=n_sig, n_nu=n_nu,
                purity=n_sig / n_sel if n_sel else float("nan"),
                eff=n_nu / n_den if n_den else float("nan"))


def true_signal_table(truth_info, stage_idx_dict):
    """One row per true signal neutrino (the efficiency / response denominator):
    true kinematics + one bool column sel_<stage> per stage (neutrino-level,
    see signal_selected_mask).  Index = the is_true_signal rows of truth_info."""
    sig = truth_info["is_true_signal"].to_numpy(dtype=bool)
    out = pd.DataFrame(index=truth_info.index[sig])
    for src, dst in [("true_leading_e_ke", "true_ke"),
                     ("true_leading_e_costheta", "true_costheta"),
                     ("true_leading_e_p", "true_p")]:
        out[dst] = truth_info.loc[sig, src].to_numpy(dtype="float32")
    if "mct_index" in truth_info.columns:
        out["nu_id"] = truth_info.loc[sig, "mct_index"].to_numpy()
    if "is_truth_only" in truth_info.columns:
        out["is_truth_only"] = truth_info.loc[sig, "is_truth_only"].to_numpy(dtype=bool)
    out["truth_cat"] = np.int8(0)
    out["is_sig"] = True
    for name, idx in stage_idx_dict.items():
        out[f"sel_{name}"] = signal_selected_mask(truth_info, idx)[sig]
    return out


# ── Leading-electron observables ──────────────────────────────────────────────

def leading_electron_ke(evtdf, particle_mask=None):
    """KE [MeV] of the leading primary reco electron per interaction."""
    il = inter_levels(evtdf)
    m  = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    return reco_particles(evtdf).ke[m].groupby(level=il).max()


def true_leading_electron_ke(evtdf):
    """KE [MeV] of the leading primary true electron per interaction."""
    il = inter_levels(evtdf)
    tp = true_particles(evtdf)
    m  = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1) & tp.ke.notna()
    if m.sum() == 0:
        return pd.Series(dtype=float)
    return tp.ke[m].groupby(level=il).max()


def leading_electron_p(evtdf, particle_mask=None):
    """Momentum magnitude [MeV/c] of the leading primary reco electron."""
    il      = inter_levels(evtdf)
    m       = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    rp      = reco_particles(evtdf)
    idx_max = rp.ke[m].groupby(level=il).idxmax().dropna()
    return rp.p.loc[idx_max].set_axis(idx_max.index)


def true_leading_electron_p(evtdf):
    """Momentum magnitude [MeV/c] of the leading primary true electron."""
    il      = inter_levels(evtdf)
    tp      = true_particles(evtdf)
    m       = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1)
    if m.sum() == 0:
        return pd.Series(dtype=float)
    idx_max = tp.energy_init[m].groupby(level=il).idxmax().dropna()
    return tp.p.loc[idx_max].set_axis(idx_max.index)


def leading_electron_costheta(evtdf, particle_mask=None):
    """cos(theta) of the leading primary reco electron w.r.t. the beam axis."""
    il      = inter_levels(evtdf)
    m       = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    rp      = reco_particles(evtdf)
    idx_max = rp.ke[m].groupby(level=il).idxmax().dropna()
    mom     = rp.momentum
    px, py, pz = mom.x.loc[idx_max], mom.y.loc[idx_max], mom.z.loc[idx_max]
    pmag    = np.sqrt(px**2 + py**2 + pz**2)
    return pd.Series((pz / pmag).values, index=idx_max.index)


def true_leading_electron_costheta(evtdf):
    """cos(theta) of the leading primary true electron w.r.t. the beam axis."""
    il      = inter_levels(evtdf)
    tp      = true_particles(evtdf)
    m       = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1)
    if m.sum() == 0:
        return pd.Series(dtype=float)
    mom     = tp.momentum
    gpx, gpy, gpz = mom.x[m], mom.y[m], mom.z[m]
    cos     = gpz / np.sqrt(gpx**2 + gpy**2 + gpz**2)
    idx_max = tp.energy_init[m].groupby(level=il).idxmax().dropna()
    return cos.loc[idx_max].set_axis(idx_max.index)