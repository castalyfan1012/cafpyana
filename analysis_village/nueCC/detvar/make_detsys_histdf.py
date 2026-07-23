"""
make_detsys_histdf.py
---------------------
Lightweight det-sys histogram extractor for combined flat CAFs.

Selection applied (matching nue_selection.py):
  1. FM + reco FV          (matches passed_presel in make_nueCC_df.py)
  2. is_primary == 1       (primary particle)
  3. pid == 1              (electron candidate; SPINE PID: 0=photon,1=electron,
                            2=muon,3=pion,4=proton,5=kaon)
  4. primary_scores.0 > PRIMARY_SCORE_CUT   (shower-like primary score)
  5. pid_scores.1 > PID_SCORE_CUT           (electron PID score, index 1)
  6. ke > ELECTRON_THRESHOLD_MEV            (above 75 MeV threshold)

Confirmed available branches (from check_detsys.py output):
  pid, pid_scores.0-.5, primary_scores.0-.1, is_primary,
  momentum.0/.1/.2, start_dir.0/.1/.2, ke, vertex_distance (all NaN)
"""

import warnings
import numpy as np
import pandas as pd
import uproot
import awkward as ak

# ── Selection constants — match nue_selection.py exactly ─────────────────────
PRIMARY_SCORE_CUT      = 0.99    # primary_scores.0 threshold
PID_SCORE_CUT          = 0.915   # pid_scores.1 (electron) threshold
ELECTRON_THRESHOLD_MEV = 75.0    # minimum KE [MeV]
PID_ELECTRON           = 1       # SPINE PID index for electron
M_ELECTRON_MEV         = 0.511


def _fv(vx, vy, vz):
    """Fiducial volume — identical to _fiducial_cut_tmp in make_nueCC_df.py."""
    abs_x = np.abs(vx)
    return (
        (abs_x > 5) & (abs_x < 190) & (
            ((vz > 10)  & (vz < 250) & (vy > -190) & (vy < 190)) |
            ((vz > 250) & (vz < 450) & (vy > -190) & (vy < 100) & (vx < 0)) |
            ((vz > 250) & (vz < 450) & (vy > -190) & (vy < 190) & (vx > 0))
        )
    )


def make_nuecc_detsys_seldf(path, verbose=True):
    """
    Read a det-var combined flat CAF and return a DataFrame with one row
    per selected electron candidate, with reco_p and reco_costheta.

    The flat CAF structure:
      - Interaction-level branches: 116420 * var * dtype
        (jagged per entry; one list per event = interactions in that event)
      - Particle-level branches:    116420 * var * dtype
        (jagged per entry; one list per event = ALL particles in that event,
        not separated by interaction — this is a flat CAF artifact)

    Strategy:
      1. Apply FM+FV on interactions → find passing entries
      2. Among particles in passing entries, apply shower quality cuts
         (is_primary, pid==electron, primary_score, pid_score, ke threshold)
      3. Take the highest-KE electron candidate per entry as the shower proxy

    Parameters
    ----------
    path    : str   xrootd or local path to the combined flat CAF
    verbose : bool

    Returns
    -------
    pd.DataFrame with columns: reco_p, reco_costheta, reco_ke
    """
    f   = uproot.open(path)
    rec = f["recTree"]

    # ── Step 1: FM + FV on interaction-level branches ─────────────────────────
    fm = ak.to_numpy(ak.flatten(rec["rec.dlp.is_flash_matched"].array(), axis=1))
    pe = ak.to_numpy(ak.flatten(rec["rec.dlp.flash_total_pe"].array(),    axis=1))
    vx = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.0"].array(),          axis=1))
    vy = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.1"].array(),          axis=1))
    vz = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.2"].array(),          axis=1))

    passed_fm  = (fm == 1) & np.isfinite(pe.astype(float)) & (pe > 0)
    passed_fv  = _fv(vx, vy, vz)
    sel_inter  = passed_fm & passed_fv

    # Map interaction-level selection to entry-level
    inter_counts   = ak.to_numpy(
        ak.num(rec["rec.dlp.is_flash_matched"].array(), axis=1)
    )
    entry_of_inter = np.repeat(np.arange(len(inter_counts)), inter_counts)
    passing_entries = np.unique(entry_of_inter[sel_inter])

    if verbose:
        print(f"  [{path.split('/')[-1]}]")
        print(f"    Total interactions : {len(fm):,}")
        print(f"    Pass FM            : {passed_fm.sum():,}")
        print(f"    Pass FV            : {passed_fv.sum():,}")
        print(f"    Pass FM+FV         : {sel_inter.sum():,}")
        print(f"    Entries with >=1 FM+FV interaction: {len(passing_entries):,}")

    # ── Step 2: Load particle-level branches for passing entries ──────────────
    # All particle branches are 116420 * var * dtype (per-entry, not per-interaction)
    def load_part(branch):
        return rec[branch].array()[passing_entries]

    is_primary  = load_part("rec.dlp.particles.is_primary")
    pid         = load_part("rec.dlp.particles.pid")
    pri_score   = load_part("rec.dlp.particles.primary_scores.0")
    pid_score_e = load_part("rec.dlp.particles.pid_scores.1")   # electron score
    ke          = load_part("rec.dlp.particles.ke")
    px          = load_part("rec.dlp.particles.momentum.0")
    py          = load_part("rec.dlp.particles.momentum.1")
    pz          = load_part("rec.dlp.particles.momentum.2")
    sdz         = load_part("rec.dlp.particles.start_dir.2")    # cos(theta)

    pmag = np.sqrt(px**2 + py**2 + pz**2)

    # ── Step 3: Shower quality cuts — match nue_selection.py ──────────────────
    electron_mask = (
        (is_primary == 1) &
        (pid == PID_ELECTRON) &
        (pri_score > PRIMARY_SCORE_CUT) &
        (pid_score_e > PID_SCORE_CUT) &
        (ke > ELECTRON_THRESHOLD_MEV)
    )

    n_with_electron = ak.to_numpy(ak.any(electron_mask, axis=1)).sum()
    if verbose:
        print(f"    Entries with >=1 electron candidate: {n_with_electron:,}")

    # ── Step 4: Leading electron candidate per entry (highest KE) ─────────────
    # Replace non-electron ke with -inf so argmax selects only electrons
    ke_electron = ak.where(electron_mask, ke, -np.inf)
    lead_idx    = ak.argmax(ke_electron, axis=1, keepdims=True)

    # Extract leading particle kinematics
    lead_ke   = ak.to_numpy(ak.flatten(ke[lead_idx],   axis=None))
    lead_pmag = ak.to_numpy(ak.flatten(pmag[lead_idx], axis=None))
    lead_cos  = ak.to_numpy(ak.flatten(sdz[lead_idx],  axis=None))

    # Keep only entries where a valid electron candidate was found
    has_electron = ak.to_numpy(ak.any(electron_mask, axis=1))
    valid = (
        has_electron &
        (lead_ke > ELECTRON_THRESHOLD_MEV) &
        np.isfinite(lead_pmag) &
        np.isfinite(lead_cos)
    )

    lead_ke   = lead_ke[valid]
    lead_pmag = lead_pmag[valid]
    lead_cos  = lead_cos[valid]

    df = pd.DataFrame({
        "reco_p":        lead_pmag,
        "reco_costheta": lead_cos,
        "reco_ke":       lead_ke,
    })

    if verbose:
        print(f"    Output rows (electron candidates): {len(df):,}")
        if len(df) > 0:
            print(f"    reco_ke  mean±std : "
                  f"{df.reco_ke.mean():.1f} ± {df.reco_ke.std():.1f} MeV")
            print(f"    reco_p   mean±std : "
                  f"{df.reco_p.mean():.1f} ± {df.reco_p.std():.1f} MeV/c")
            print(f"    reco_cos mean±std : "
                  f"{df.reco_costheta.mean():.3f} ± {df.reco_costheta.std():.3f}")

    return df