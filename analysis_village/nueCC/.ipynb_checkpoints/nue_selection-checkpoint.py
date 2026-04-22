"""
nue_selection.py (STREAMLINED + SIMPLIFIED)
----------------
Cleaned-up version per your request:
- Removed all tiny private helpers (_good_primary, _primary_of_pid, _interaction_index,
  _inter_any, _interactions_where, _fiducial_inter_mask, _flashmatch_inter_mask,
  _flash_time_inter_mask). They are now inlined directly in `build_cut_flow`.
- Flash-time logic completely removed (only `is_flash_matched` is used – as you asked).
- Only the major, public functions remain:
  - `build_cut_flow`
  - `classify_truth`
  - `true_signal_index`
  - leading-electron observables
- Code is much shorter and easier to read while doing exactly the same thing.

(Everything else – thresholds, FV, truth categories, etc. – is unchanged.)
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from nue_helpers import (
    reco_particles, true_particles,
    reco_interactions, true_interactions,
    inter_levels,
)

# ============================================================
# Thresholds (MeV) – must match the TOML configuration
# ============================================================
ELECTRON_THRESHOLD_MEV = 0.0
MUON_THRESHOLD_MEV = 50.0
PHOTON_THRESHOLD_MEV = 25.0
PION_THRESHOLD_MEV = 25.0
PROTON_THRESHOLD_MEV = 50.0

# PID codes
PID_PHOTON = 0
PID_ELECTRON = 1
PID_MUON = 2
PID_PION = 3
PID_PROTON = 4

# Cut labels (used for tables/plots)
CUT_NAMES = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    # "no_muons",
    "single_electron",
]
CUT_LABELS = [
    "No cut",
    "Flash match",
    "In FV",
    # "No primary muons",
    "Final state topology",
]

CUT_NAMES_MORE = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    # "no_muons",
    "single_electron",
    "electron_pid_score",
    "electron_primary_score",
    "start_dedx",
    "vertex_distance",
    "directional_spread",
    "axial_spread"
]
CUT_LABELS_MORE = [
    "No cut",
    "Flash match",
    "In FV",
    # "No primary muons",
    "Final state topology",
    "Electron PID score",
    "Electron Primary score",
    "Reco start dE/dx",
    "Conversion gap",
    "Directional spread",
    "Axial spread"
]

# ============================================================
# Public particle-level masks (kept – they are useful and only 1–2 lines)
# ============================================================
def primary_electron_mask(evtdf):
    """Particle-level mask for good primary electrons above threshold."""
    rp = reco_particles(evtdf)
    return (
        (rp.is_primary == 1) &
        (rp.pid == PID_ELECTRON) &
        (rp.ke > ELECTRON_THRESHOLD_MEV)
    )

def primary_muon_mask(evtdf):
    """Particle-level mask for good primary muons above threshold."""
    rp = reco_particles(evtdf)
    return (
        (rp.is_primary == 1) &
        (rp.pid == PID_MUON) &
        (rp.ke > MUON_THRESHOLD_MEV)
    )


# ============================================================
# Truth classification (unchanged – already clean)
# ============================================================
def classify_truth(evtdf, verbose=True):
    """
    Assign each reconstructed interaction to a truth category matching the
    TOML definitions.  Uses raw columns only – no derived flavor flags.
 
    Categories (priority order, first match wins)
    -----------------------------------------------
    0  neutrino  CC   fiducial   single_electron (true)   ← nueCC signal in FV
    1  neutrino  CC  !fiducial   single_electron (true)   ← nueCC out of FV
    2  neutrino  CC   single_muon (true)   pi0 (true)
    3  neutrino  NC   pi0 (true)
    4  neutrino  CC   single_muon (true)                  (no pi0 left after cat 2)
    5  neutrino  NC                                        (no pi0 left after cat 3)
    6  !neutrino                                           cosmic / dirt
    7  neutrino   (catch-all – anything not matched above)
    """
    il  = inter_levels(evtdf)
    idx = evtdf.groupby(level=il).size().index
 
    if verbose:
        print("classify_truth: reading true-interaction flags …")
 
    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)
    ri = reco_interactions(evtdf)
 
    # ── Neutrino flag (nu_id >= 0) ──────────────────────────────────────────
    is_nu = (ti.nu_id >= 0).groupby(level=il).any().reindex(idx, fill_value=False)
 
    # ── CC flag (current_type == 0) ─────────────────────────────────────────
    is_cc = (ti.current_type == 0).groupby(level=il).any().reindex(idx, fill_value=False)
 
    # ── Fiducial flag (true is_fiducial) ────────────────────────────────────
    is_fv = (ti.is_fiducial == 1).groupby(level=il).any().reindex(idx, fill_value=False)
 
    # ── True single-electron: >= 1 primary true electron above threshold ────
    true_elec_counts = (
        (tp.pid == PID_ELECTRON) &
        (tp.is_primary == 1) &
        (tp.ke > ELECTRON_THRESHOLD_MEV)
    ).groupby(level=il).sum()
    
    true_elec = (true_elec_counts == 1)
 
    # ── True single-muon: >= 1 primary true muon above threshold ────────────
    true_muon = (
        (tp.pid == PID_MUON) &
        (tp.is_primary == 1) &
        (tp.ke > MUON_THRESHOLD_MEV)
    ).groupby(level=il).any().reindex(idx, fill_value=False)
 
    # ── True pi0: any particle whose parent PDG is 111 ──────────────────────
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
 
    # ── Assign categories (priority order) ──────────────────────────────────
    cat = pd.Series(7, index=idx, dtype=int)   # default: neutrino catch-all
 
    # Cat 6: cosmic / non-neutrino  (highest priority override)
    cat[~is_nu] = 6
 
    # The following only apply to neutrino interactions
    nu = is_nu  # shorthand
 
    cat[nu &  is_cc & ~is_fv  & true_elec] = 1   # nueCC out of FV (before FV cut)
    cat[nu &  is_cc &  is_fv  & true_elec] = 0   # nueCC in FV  ← signal
    cat[nu &  is_cc & true_muon & has_pi0]  = 2   # numuCC + pi0
    cat[nu & ~is_cc & has_pi0]              = 3   # NC + pi0
    cat[nu &  is_cc & true_muon & ~has_pi0] = 4   # other numuCC
    cat[nu & ~is_cc & ~has_pi0]             = 5   # other NC
    # cat 7 remains for any neutrino interaction not caught above
 
    if verbose:
        labels = {
            0: "nueCC FV (signal)",   1: "nueCC out FV",
            2: "numuCC + π⁰",         3: "NC π⁰",
            4: "other numuCC",        5: "other NC",
            6: "cosmic",              7: "other neutrino",
        }
        print("\n=== Truth category summary ===")
        for c, lbl in labels.items():
            print(f"  cat {c:2d}  {lbl:<22s}: {(cat == c).sum():>8,}")
        print(f"  {'total':<26s}: {len(cat):>8,}")
 
    return cat

# ============================================================
# Full cut flow (now self-contained – no private helpers)
# ============================================================
def build_cut_flow(evtdf):
    """
    Apply the nueCC selection sequentially.
 
    Cuts follow the TOML definition:
        valid_flashmatch  reco  is_flash_matched == 1
        fiducial          reco  is_fiducial == 1          (no custom FV needed)
        no_muons          reco  no primary muon above threshold
        single_electron   reco  >= 1 primary electron above threshold
 
    The shower_qual step is intentionally left empty here; it is filled by
    apply_shower_qual_cuts() once optimal thresholds are known.
 
    Returns
    -------
    dict  keyed by CUT_NAMES, each value is
          {"inter_index": pd.Index, "particle_mask": pd.Series}
    """
    il          = inter_levels(evtdf)
    all_inter   = evtdf.groupby(level=il).size().index
    results     = {}
 
    def _pmask(idx):
        return pd.Series(
            evtdf.index.droplevel(-1).isin(idx),
            index=evtdf.index,
        )
 
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
            fm_mask  = reco_interactions(evtdf).is_flash_matched == 1
            fm_inter = fm_mask.groupby(level=il).any()
            fm_inter = fm_inter[fm_inter].index
            current  = current.intersection(fm_inter)
            results["valid_flashmatch"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }
 
        elif step == "fiducial":
            # Use is_fiducial directly – no custom FV definition needed
            fv_mask  = reco_interactions(evtdf).is_fiducial == 1
            fv_inter = fv_mask.groupby(level=il).any()
            fv_inter = fv_inter[fv_inter].index
            current  = current.intersection(fv_inter)
            results["fiducial"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }
 
        # elif step == "no_muons":
        #     muon_inter = primary_muon_mask(evtdf).groupby(level=il).any()
        #     muon_inter = muon_inter[muon_inter].index
        #     current    = current.difference(muon_inter)
        #     results["no_muons"] = {
        #         "inter_index":   current,
        #         "particle_mask": _pmask(current),
        #     }
 
        elif step == "single_electron":
            elec_counts = primary_electron_mask(evtdf).groupby(level=il).sum()
            elec_inter  = elec_counts[elec_counts >= 1].index
            current     = current.intersection(elec_inter)
            results["single_electron"] = {
                "inter_index":   current,
                "particle_mask": _pmask(current),
            }

 
    return results


def shower_qual_cuts(evtdf):
    """
    Full cut-flow including all shower-quality cuts.
    Re-uses build_cut_flow for the common early steps (fast) and only
    applies the new cuts afterwards.

    OPTIMIZED: the leading electron is computed ONCE outside the loop.
    All shower-quality cuts operate on that same leading electron.
    """
    # ── Reuse the already-working early cuts (precut → single_electron) ──
    cut_flow = build_cut_flow(evtdf)
    il = inter_levels(evtdf)

    def _pmask(idx):
        return pd.Series(
            evtdf.index.droplevel(-1).isin(idx),
            index=evtdf.index,
        )

    # The new cuts start right after "single_electron"
    additional_cuts = CUT_NAMES_MORE[CUT_NAMES_MORE.index("single_electron") + 1 :]

    if not additional_cuts:
        return cut_flow

    # ── OPTIMIZATION: compute leading electron ONCE (all remaining cuts use it) ──
    rp = reco_particles(evtdf)
    rp_df = rp._df                                      # plain pandas DataFrame (2-level MultiIndex)

    # Column keys inside rp._df (after ParticleView strips the common prefix)
    KE_COL = ('ke', '')

    # Select electrons (only done once)
    ele_mask = rp.pid == PID_ELECTRON
    ele = rp_df[ele_mask]

    # Leading electron per interaction (max KE)
    ke_series = ele[KE_COL]
    leading_electron_idx = ke_series.groupby(level=il).idxmax()
    leading_electron = rp_df.loc[leading_electron_idx]   # ← reused for every cut

    # ── Apply the extra shower-quality cuts sequentially ──
    steps = tqdm(additional_cuts, desc="Applying extra shower-quality cuts", leave=True)
    current = cut_flow["single_electron"]["inter_index"]

    for step in steps:
        steps.set_postfix(n=len(current))

        if step == "electron_pid_score":
            col = ('pid_scores', 'I1')
            soft_inter = leading_electron[col] > 0.935

        elif step == "electron_primary_score":
            col = ('primary_scores', 'I1')
            soft_inter = leading_electron[col] > 0.99

        elif step == "start_dedx":
            col = ('start_dedx', '')
            soft_inter = leading_electron[col] < 3         

        elif step == "vertex_distance":
            col = ('vertex_distance', '')
            soft_inter = leading_electron[col] < 1.05          

        elif step == "directional_spread":
            col = ('directional_spread', '')
            soft_inter = leading_electron[col] < 0.026         

        elif step == "axial_spread":
            col = ('axial_spread', '')
            soft_inter = leading_electron[col] > 0.315          

        else:
            raise ValueError(f"Unknown extra cut: {step}")

        # ── Interaction-level reduction (common to all cuts) ──
        soft_inter = soft_inter.groupby(level=il).any()
        soft_inter = soft_inter[soft_inter].index

        current = current.intersection(soft_inter)

        cut_flow[step] = {
            "inter_index": current,
            "particle_mask": _pmask(current),
        }

    return cut_flow
    
    
# ============================================================
# True-signal denominator for efficiency (unchanged)
# ============================================================
def true_signal_index(evtdf):
    il = inter_levels(evtdf)
    ti = true_interactions(evtdf)
    tp = true_particles(evtdf)
    is_nu = (ti.nu_id >= 0).groupby(level=il).any()
    is_cc = (ti.current_type == 0).groupby(level=il).any()
    is_fv = (ti.is_fiducial == 1).groupby(level=il).any()
    elec_counts = (
        (tp.pid == PID_ELECTRON) &
        (tp.is_primary == 1) &
        (tp.ke > ELECTRON_THRESHOLD_MEV)
    ).groupby(level=il).sum()
    
    prim_elec = (elec_counts == 1)
    # no_prim_muon = ~(
    #     (tp.pid == PID_MUON) &
    #     (tp.is_primary == 1) &
    #     (tp.ke > MUON_THRESHOLD_MEV)
    # ).groupby(level=il).any()
    sig = is_nu & is_cc & is_fv & prim_elec #& no_prim_muon
    return sig[sig].index

# ============================================================
# Leading-electron observables (unchanged – they use the kept masks)
# ============================================================
def leading_electron_ke(evtdf, particle_mask=None):
    il = inter_levels(evtdf)
    m = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    return reco_particles(evtdf).ke[m].groupby(level=il).max()

def leading_electron_costheta(evtdf, particle_mask=None):
    il = inter_levels(evtdf)
    m = primary_electron_mask(evtdf)
    if particle_mask is not None:
        m = m & particle_mask
    rp = reco_particles(evtdf)
    ke = rp.ke[m]
    idx_max = ke.groupby(level=il).idxmax().dropna()
    mom = rp.momentum
    px = mom.x.loc[idx_max]
    py = mom.y.loc[idx_max]
    pz = mom.z.loc[idx_max]
    pmag = np.sqrt(px**2 + py**2 + pz**2)
    return pd.Series((pz / pmag).values, index=idx_max.index)

def true_leading_electron_ke(evtdf):
    """Strong debug version to see what tp.ke really contains"""
    il = inter_levels(evtdf)
    tp = true_particles(evtdf)
    
    m = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1) & tp.ke.notna()
    
    if m.sum() == 0:
        print("ERROR: No primary electrons found!")
        return pd.Series(dtype=float)
    
    raw_ke = tp.ke[m]
    
    print("=== RAW tp.ke VALUES (primary electrons) ===")
    print(f"Number of primary electrons : {len(raw_ke):,}")
    print(f"Raw min  : {raw_ke.min():.4f}")
    print(f"Raw max  : {raw_ke.max():.4f}")
    print(f"Raw mean : {raw_ke.mean():.4f}")
    print(f"Raw median: {raw_ke.median():.4f}")
    print("=====================================")
    
    # Test different scalings
    ke_mev_try1 = raw_ke * 1000.0      # assume GeV → MeV
    ke_mev_try2 = raw_ke               # assume already MeV
    
    print(f"After *1000 (GeV→MeV)  → min/max: {ke_mev_try1.min():.1f} – {ke_mev_try1.max():.1f} MeV")
    print(f"After *1     (already MeV) → min/max: {ke_mev_try2.min():.1f} – {ke_mev_try2.max():.1f} MeV")
    
    # Use the more reasonable one (we'll decide based on output)
    # For now, return the raw so we can see
    return raw_ke.groupby(level=il).max()

def true_leading_electron_costheta(evtdf):
    il = inter_levels(evtdf)
    tp = true_particles(evtdf)
    m = (tp.pid == PID_ELECTRON) & (tp.is_primary == 1)
    if m.sum() == 0:
        return pd.Series(dtype=float)
    mom = tp.momentum
    gpx, gpy, gpz = mom.x[m], mom.y[m], mom.z[m]
    gp = np.sqrt(gpx**2 + gpy**2 + gpz**2)
    cos = gpz / gp
    ge = tp.energy_init[m]
    imax = ge.groupby(level=il).idxmax().dropna()
    return cos.loc[imax].set_axis(imax.index)