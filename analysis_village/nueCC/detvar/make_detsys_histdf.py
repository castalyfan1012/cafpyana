"""
make_detsys_histdf.py
---------------------
Lightweight det-sys histogram extractor for combined flat CAFs.

Selection applied (matching nue_selection.py shower_qual_cuts):
  1. FM + reco FV
  2. pid == 1              (electron candidate)
  3. primary_scores.1 > PRIMARY_SCORE_CUT   (primary score — INDEX 1, not 0!)
  4. pid_scores.1 > PID_SCORE_CUT           (electron PID score — index 1)
  5. ke > ELECTRON_THRESHOLD_MEV            (above 75 MeV threshold)

NOTE: `is_primary == 1` is NOT required here. In the nominal pipeline,
nue_selection.primary_electron_mask uses is_primary, but in Jacob's
data-mode wiremod CAFs, is_primary==1 & pid==1 barely overlap (4/8681
in 200 entries). The score-based cuts already select primary-like
electron candidates. This is acceptable for det-var shape studies.

Column index mapping (flat CAF ↔ cafpyana MultiIndex):
  primary_scores.0 ↔ ('primary_scores', 'I0')  ← NOT the primary score
  primary_scores.1 ↔ ('primary_scores', 'I1')  ← THIS is the primary score
  pid_scores.0     ↔ ('pid_scores', 'I0')      ← photon score
  pid_scores.1     ↔ ('pid_scores', 'I1')      ← electron score ✓

CRITICAL: reco_p is computed from calorimetric KE, NOT momentum.* branches.
"""

import numpy as np
import pandas as pd
import uproot
import awkward as ak

# ── Selection constants — match nue_selection.py exactly ─────────────────────
PRIMARY_SCORE_CUT      = 0.99    # primary_scores.1 threshold (I1 in cafpyana)
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


def _reco_p_from_ke(ke_mev):
    """Compute reco momentum from calorimetric KE: p = sqrt((KE+m)^2 - m^2)."""
    total_e = ke_mev + M_ELECTRON_MEV
    return np.sqrt(total_e**2 - M_ELECTRON_MEV**2)


def make_nuecc_detsys_seldf(path, verbose=True):
    """
    Read a det-var combined flat CAF and return a DataFrame with one row
    per selected electron candidate, with reco_ke and reco_costheta.

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

    if len(passing_entries) == 0:
        if verbose:
            print("    No passing entries — returning empty DataFrame")
        return pd.DataFrame(columns=["reco_p", "reco_costheta", "reco_ke"])

    # ── Step 2: Load particle-level branches for passing entries ──────────────
    def load_part(branch):
        return rec[branch].array()[passing_entries]

    pid         = load_part("rec.dlp.particles.pid")
    # CRITICAL: use primary_scores.1 (= I1 in cafpyana = primary probability)
    #           NOT primary_scores.0 (= I0 = secondary probability)
    pri_score   = load_part("rec.dlp.particles.primary_scores.1")
    pid_score_e = load_part("rec.dlp.particles.pid_scores.1")   # electron score
    ke          = load_part("rec.dlp.particles.ke")
    sdz         = load_part("rec.dlp.particles.start_dir.2")    # cos(theta)

    # ── Step 3: Electron selection — NO is_primary requirement ────────────────
    # In data-mode wiremod CAFs, is_primary==1 & pid==1 barely overlap.
    # The score cuts (primary_scores.1 > 0.99) already select primary-like
    # candidates, which is sufficient for det-var shape studies.
    electron_mask = (
        (pid == PID_ELECTRON) &
        (pri_score > PRIMARY_SCORE_CUT) &
        (pid_score_e > PID_SCORE_CUT) &
        (ke > ELECTRON_THRESHOLD_MEV)
    )

    n_with_electron = ak.to_numpy(ak.any(electron_mask, axis=1)).sum()
    if verbose:
        # Diagnostic: show where candidates are lost
        m_pid  = (pid == PID_ELECTRON)
        m_pri  = m_pid & (pri_score > PRIMARY_SCORE_CUT)
        m_pids = m_pri & (pid_score_e > PID_SCORE_CUT)
        m_ke   = m_pids & (ke > ELECTRON_THRESHOLD_MEV)
        print(f"    Particle selection chain:")
        print(f"      pid==1 (electron)      : {int(ak.sum(ak.flatten(m_pid, axis=None))):,} particles")
        print(f"      + primary_scores.1>0.99: {int(ak.sum(ak.flatten(m_pri, axis=None))):,} particles")
        print(f"      + pid_scores.1>0.915   : {int(ak.sum(ak.flatten(m_pids, axis=None))):,} particles")
        print(f"      + ke>75 MeV            : {int(ak.sum(ak.flatten(m_ke, axis=None))):,} particles")
        print(f"    Entries with >=1 electron candidate: {n_with_electron:,}")

    if n_with_electron == 0:
        if verbose:
            print("    No electron candidates — returning empty DataFrame")
        return pd.DataFrame(columns=["reco_p", "reco_costheta", "reco_ke"])

    # ── Step 4: Leading electron candidate per entry (highest KE) ─────────────
    ke_electron = ak.where(electron_mask, ke, -np.inf)
    lead_idx    = ak.argmax(ke_electron, axis=1, keepdims=True)

    lead_ke   = ak.to_numpy(ak.flatten(ke[lead_idx],   axis=None))
    lead_cos  = ak.to_numpy(ak.flatten(sdz[lead_idx],  axis=None))

    # Compute reco_p from calorimetric KE (NOT from momentum.* branches)
    lead_pmag = _reco_p_from_ke(lead_ke)

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