"""
nue_selection.py
----------------
nueCC inclusive selection for SPINE DLP.

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
ELECTRON_THRESHOLD_MEV = 75.0
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

# Cut labels
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

CUT_NAMES_MORE = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    "single_electron",
    "electron_primary_score",
    "electron_pid_score",
    "vertex_distance",
    # "start_dedx",
    # "axial_spread",
    # "directional_spread"
]
CUT_LABELS_MORE = [
    "No cut",
    "Flash match",
    "In FV",
    "Final state topology",
    "Electron primary score",
    "Electron PID score",
    "Conversion gap",
    # "Start dE/dx",
    # "Axial spread",
    # "Opening angle"
]

THRESH_PRIMARY_SCORE = 0.99
THRESH_PID_SCORE     = 0.915
THRESH_VERTEX_DIST   = 2.65
THRESH_DIR_SPREAD    = 0.199
THRESH_AXIAL_SPREAD  = 0.02
THRESH_DEDX          = 7.95

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
    Assign each reconstructed interaction to a truth category.

    FV uses fiducial_cut_tmp on the true interaction vertex (geometric cut,
    not is_fiducial). Must match make_nueCC_df.py _build_truth_info exactly.

    Categories (priority order, first match wins)
    -----------------------------------------------
    0  nueCC FV (signal)   1  nueCC out FV
    2  numuCC + pi0         3  NC pi0
    4  other numuCC         5  other NC
    6  cosmic               7  neutrino catch-all
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

    cat = pd.Series(7, index=idx, dtype=int)
    cat[~is_nu] = 6
    nu = is_nu
    cat[nu &  is_cc & ~is_fv  & true_elec]            = 1
    cat[nu &  is_cc &  is_fv  & true_elec]            = 0   # signal
    cat[nu &  is_cc & true_muon & has_pi0]             = 2
    cat[nu & ~is_cc & has_pi0]                         = 3
    cat[nu &  is_cc & true_muon & ~has_pi0]            = 4
    cat[nu & ~is_cc & ~has_pi0]                        = 5

    if verbose:
        labels = {
            0: "nueCC FV (signal)",  1: "nueCC out FV",
            2: "numuCC + pi0",       3: "NC pi0",
            4: "other numuCC",       5: "other NC",
            6: "cosmic",             7: "other neutrino",
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


# ── Extended cut flow (shower quality) ───────────────────────────────────────

def shower_qual_cuts(evtdf, flash_times_df=None):
    """
    Full cut-flow including shower-quality cuts after single_electron.
    Leading reco electron computed once and reused for all extra cuts.

    Parameters
    ----------
    evtdf          : merged SPINE reco+truth particle-level DataFrame.
    flash_times_df : passed through to build_cut_flow for the FM cut.
                     See build_cut_flow docstring for details.
    """
    cut_flow = build_cut_flow(evtdf, flash_times_df=flash_times_df)
    il       = inter_levels(evtdf)

    def _pmask(idx):
        return pd.Series(evtdf.index.droplevel(-1).isin(idx), index=evtdf.index)

    additional_cuts = CUT_NAMES_MORE[CUT_NAMES_MORE.index("single_electron") + 1:]
    if not additional_cuts:
        return cut_flow

    rp    = reco_particles(evtdf)
    rp_df = rp._df
    KE_COL = ('ke', '')
    ele_mask = rp.pid == PID_ELECTRON
    leading_electron = rp_df.loc[
        rp_df[ele_mask][KE_COL].groupby(level=il).idxmax()
    ]

    steps   = tqdm(additional_cuts, desc="Shower quality cuts", leave=True)
    current = cut_flow["single_electron"]["inter_index"]

    for step in steps:
        steps.set_postfix(n=len(current))

        if step == "electron_primary_score":
            soft_inter = leading_electron[('primary_scores', 'I1')] > THRESH_PRIMARY_SCORE
        elif step == "electron_pid_score":
            soft_inter = leading_electron[('pid_scores', 'I1')] > THRESH_PID_SCORE
        elif step == "vertex_distance":
            soft_inter = leading_electron[('vertex_distance', '')] < THRESH_VERTEX_DIST
        # elif step == "start_dedx":
        #     soft_inter = leading_electron[('start_dedx', '')] < THRESH_DEDX
        # elif step == "axial_spread":
        #     soft_inter = leading_electron[('axial_spread', '')] > THRESH_AXIAL_SPREAD
        # elif step == "directional_spread":
        #     soft_inter = leading_electron[('directional_spread', '')] < THRESH_DIR_SPREAD
        else:
            raise ValueError(f"Unknown shower quality cut: {step}")

        soft_inter = soft_inter.groupby(level=il).any()
        current    = current.intersection(soft_inter[soft_inter].index)
        cut_flow[step] = {
            "inter_index":   current,
            "particle_mask": _pmask(current),
        }

    return cut_flow


# ── True-signal denominator ───────────────────────────────────────────────────

def true_signal_index(evtdf):
    """Return interaction index of all true nueCC-in-FV signal interactions.
    FV uses fiducial_cut_tmp on true vertex (matches classify_truth and
    make_nueCC_df.py _build_truth_info).
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