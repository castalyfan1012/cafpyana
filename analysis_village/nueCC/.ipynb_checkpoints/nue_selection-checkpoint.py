"""
nue_selection.py
----------------
nueCC inclusive selection for SPINE DLP.

Truth categories (match make_nueCC_df.py exactly)
--------------------------------------------------
    0  nueCC FV          <- signal
    1  nueCC out FV
    2  numuCC + pi0
    3  NC pi0
    4  other numuCC
    5  other NC
    6  cosmic / non-neutrino
    7  neutrino catch-all

valid_flashmatch (matches C++ definition)
-----------------------------------------
    C++: flash_times.size() > 0 && is_flash_matched == 1 && !isnan(flash_times[0])
    Proxy used here: is_flash_matched == 1 AND flash_total_pe > 0
    (flash_total_pe is a scalar branch already in spineint_branches;
     flash_total_pe > 0 is equivalent to the C++ size/nan check.)
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
]
CUT_LABELS_MORE = [
    "No cut",
    "Flash match",
    "In FV",
    "Final state topology",
    "Electron primary score",
    "Electron PID score",
    "Conversion gap",
]

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
    Priority order: first match wins.

    Categories
    ----------
    0  neutrino CC  fiducial  single true electron    <- nueCC signal in FV
    1  neutrino CC !fiducial  single true electron    <- nueCC out of FV
    2  neutrino CC  single true muon + pi0
    3  neutrino NC  pi0
    4  neutrino CC  single true muon (no pi0)
    5  neutrino NC  (no pi0)
    6  non-neutrino                                   <- cosmic / dirt
    7  neutrino catch-all
    """
    il  = inter_levels(evtdf)
    idx = evtdf.groupby(level=il).size().index

    if verbose:
        print("classify_truth: reading true-interaction flags …")

    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)

    is_nu = (ti.nu_id >= 0).groupby(level=il).any().reindex(idx, fill_value=False)
    is_cc = (ti.current_type == 0).groupby(level=il).any().reindex(idx, fill_value=False)
    is_fv = (ti.is_fiducial == 1).groupby(level=il).any().reindex(idx, fill_value=False)

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

def build_cut_flow(evtdf):
    """
    Apply the nueCC selection sequentially, starting from the preselected evtdf
    (FM+FV already applied at the DataFrame-maker level).

    Cuts
    ----
    valid_flashmatch  is_flash_matched==1 AND flash_total_pe>0
    fiducial          is_fiducial==1
    single_electron   >= 1 primary reco electron above threshold

    Returns
    -------
    dict keyed by CUT_NAMES; each value has
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
            # Proxy for C++ valid_flashmatch:
            #   flash_times.size()>0 && is_flash_matched==1 && !isnan(flash_times[0])
            # flash_total_pe>0 is equivalent: a valid matched flash always has PE>0.
            fm_mask = (ri.is_flash_matched == 1) & (ri.flash_total_pe > 0)
            fm_inter = fm_mask.groupby(level=il).any()
            current  = current.intersection(fm_inter[fm_inter].index)
            results["valid_flashmatch"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }

        elif step == "fiducial":
            fv_mask  = reco_interactions(evtdf).is_fiducial == 1
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

def shower_qual_cuts(evtdf):
    """
    Full cut-flow including shower-quality cuts after single_electron.
    The leading reco electron is computed once and reused for all extra cuts.
    """
    cut_flow = build_cut_flow(evtdf)
    il       = inter_levels(evtdf)

    def _pmask(idx):
        return pd.Series(evtdf.index.droplevel(-1).isin(idx), index=evtdf.index)

    additional_cuts = CUT_NAMES_MORE[CUT_NAMES_MORE.index("single_electron") + 1:]
    if not additional_cuts:
        return cut_flow

    # Compute leading electron once
    rp    = reco_particles(evtdf)
    rp_df = rp._df
    KE_COL = ('ke', '')
    ele_mask = rp.pid == PID_ELECTRON
    ele      = rp_df[ele_mask]
    leading_electron = rp_df.loc[
        ele[KE_COL].groupby(level=il).idxmax()
    ]

    steps   = tqdm(additional_cuts, desc="Shower quality cuts", leave=True)
    current = cut_flow["single_electron"]["inter_index"]

    for step in steps:
        steps.set_postfix(n=len(current))

        if step == "electron_primary_score":
            soft_inter = leading_electron[('primary_scores', 'I1')] > 0.99

        elif step == "electron_pid_score":
            soft_inter = leading_electron[('pid_scores', 'I1')] > 0.915

        elif step == "vertex_distance":
            soft_inter = leading_electron[('vertex_distance', '')] < 3.9

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
    """Return interaction index of all true nueCC-in-FV signal interactions."""
    il = inter_levels(evtdf)
    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)

    is_nu = (ti.nu_id >= 0).groupby(level=il).any()
    is_cc = (ti.current_type == 0).groupby(level=il).any()
    is_fv = (ti.is_fiducial == 1).groupby(level=il).any()

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
    il  = inter_levels(evtdf)
    m   = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    rp      = reco_particles(evtdf)
    idx_max = rp.ke[m].groupby(level=il).idxmax().dropna()
    return rp.p.loc[idx_max].set_axis(idx_max.index)


def true_leading_electron_p(evtdf):
    """Momentum magnitude [MeV/c] of the leading primary true electron."""
    il  = inter_levels(evtdf)
    tp  = true_particles(evtdf)
    m   = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1)
    if m.sum() == 0:
        return pd.Series(dtype=float)
    idx_max = tp.energy_init[m].groupby(level=il).idxmax().dropna()
    return tp.p.loc[idx_max].set_axis(idx_max.index)


def leading_electron_costheta(evtdf, particle_mask=None):
    """cos(theta) of the leading primary reco electron w.r.t. the beam axis."""
    il  = inter_levels(evtdf)
    m   = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    rp      = reco_particles(evtdf)
    idx_max = rp.ke[m].groupby(level=il).idxmax().dropna()
    mom     = rp.momentum
    px = mom.x.loc[idx_max]
    py = mom.y.loc[idx_max]
    pz = mom.z.loc[idx_max]
    pmag = np.sqrt(px**2 + py**2 + pz**2)
    return pd.Series((pz / pmag).values, index=idx_max.index)


def true_leading_electron_costheta(evtdf):
    """cos(theta) of the leading primary true electron w.r.t. the beam axis."""
    il  = inter_levels(evtdf)
    tp  = true_particles(evtdf)
    m   = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1)
    if m.sum() == 0:
        return pd.Series(dtype=float)
    mom     = tp.momentum
    gpx, gpy, gpz = mom.x[m], mom.y[m], mom.z[m]
    gp      = np.sqrt(gpx**2 + gpy**2 + gpz**2)
    cos     = gpz / gp
    idx_max = tp.energy_init[m].groupby(level=il).idxmax().dropna()
    return cos.loc[idx_max].set_axis(idx_max.index)