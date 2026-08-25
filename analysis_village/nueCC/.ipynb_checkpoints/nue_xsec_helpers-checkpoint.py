"""
nue_xsec_helpers.py  (v2)
-------------------------
Helper utilities for the SBND nueCC inclusive cross-section analysis.

Updated to align with:
  - The systematics notebook sample loading (same DF files, sel_topo, covariances)
  - Lynn Tung's technote (v10) methodology: CCBC, Wiener-SVD, FDTs, binning
  - OmniFold comparison infrastructure

Key additions over v1:
  - Lynn's binning and physical constants
  - CCBC (Conditional Covariance Background Constraint) implementation
  - Covariance loading from saved_syst/ NPZ files
  - Unified response matrix / efficiency from sel_topo
  - Fake data test driver with CCBC support
  - OmniFold-compatible plot infrastructure
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2 as scipy_chi2
from collections import defaultdict


# =============================================================================
# Physical constants  (from Lynn's technote v10, Sec 6.1)
# =============================================================================

# Fiducial volume from make_nueCC_df.py (same as syst notebook)
#   Region 1: z in (10,250), y in (-190,190), full x  → 2*185 * 380 * 240
#   Region 2: z in (250,450), y in (-190,100), x<0    → 185 * 290 * 200
#   Region 3: z in (250,450), y in (-190,190), x>0    → 185 * 380 * 200
V_FV_R1 = 2 * 185.0 * 380.0 * 240.0  # cm³
V_FV_R2 = 185.0 * 290.0 * 200.0
V_FV_R3 = 185.0 * 380.0 * 200.0
V_FV = V_FV_R1 + V_FV_R2 + V_FV_R3   # 5.85e7 cm³

RHO_LAR     = 1.3954       # g/cm³ (SBND LAr density, syst notebook)
N_AVOGADRO  = 6.02214076e23
M_AR        = 39.948       # g/mol
N_TARGETS   = RHO_LAR * V_FV * N_AVOGADRO / M_AR  # ~1.23e30

# Lynn's values (different FV definition and density)
LYNN_RHO       = 1.38434      # g/cm³
LYNN_V_FV      = 58_534_000   # cm³
LYNN_N_TARGETS = LYNN_RHO * LYNN_V_FV * N_AVOGADRO / M_AR  # 4.88e31
LYNN_FLUX      = 9.65e-11     # cm⁻² per POT (νe + ν̄e)

# Default: use our FV definition (matches syst notebook)
# Set USE_LYNN_CONSTANTS = True to use Lynn's values for comparison
USE_LYNN_CONSTANTS = False


def get_constants():
    """Return (N_targets, integrated_flux) consistent with the chosen FV."""
    if USE_LYNN_CONSTANTS:
        return LYNN_N_TARGETS, LYNN_FLUX
    return N_TARGETS, LYNN_FLUX  # use our N_targets but Lynn's flux


# =============================================================================
# Binning definitions  (Lynn's technote v10, Sec 2.3)
# =============================================================================

# Lynn's analysis binning (GeV for energy, unitless for cosθ)
# First and last bins are underflow/overflow
LYNN_ENERGY_BINS_GEV = np.array([0.5, 0.7, 0.95, 1.25, 1.7, 2.5])
LYNN_COS_BINS        = np.array([-1.0, 0.6, 0.75, 0.85, 0.925, 1.0])

# Convert to MeV for SPINE-based analysis (our reco_ke is in MeV)
LYNN_ENERGY_BINS_MEV = LYNN_ENERGY_BINS_GEV * 1000.0

# Our analysis binning (from syst notebook, in MeV)
# These are the covariance-matrix binning — must match saved_syst/ NPZ files
SYST_KE_BINS  = np.linspace(0, 2000, 8)   # 7 bins
SYST_COS_BINS = np.linspace(-1, 1, 11)    # 10 bins

# Custom variable-width binning for xsec (from nue_xsec_helpers NueCCBinning2D)
NUE_COSTHETA_BINS = np.array([-1.0, 0.0, 0.5, 0.75, 0.85, 0.95, 1.001])
NUE_KE_BINS_2D = np.array([
    [0,  300,  700, 1200, 2000.],
    [0,  200,  500, 1000, 2000.],
    [0,  200,  500, 1000, 2000.],
    [0,  150,  400,  900, 2000.],
    [0,  150,  400,  900, 2000.],
    [0,  150,  400,  900, 2000.],
], dtype=float)


# =============================================================================
# Category labels and colors  (consistent with syst notebook)
# =============================================================================

CAT_LABELS = {
    0:  r"$\nu_e$ CC (FV)",          1:  r"$\nu_e$ CC (out FV)",
    2:  r"$\nu_\mu$ CC $\pi^0$",     3:  r"$\nu$ NC $\pi^0$",
    4:  r"Other $\nu_\mu$ CC",       5:  r"Other $\nu$ NC",
    6:  r"Cosmic",                   7:  r"Offbeam",
    9:  r"Dirt $\nu$",              -1:  r"Unknown",
}
CAT_COLORS = {
    0:  "#56B4E9",   1:  "#0072B2",   2:  "green",     3:  "#009E73",
    4:  "#E69F00",   5:  "#CC79A7",   6:  "#F0E442",   7:  "#BDBDBD",
    9:  "#D2691E",  -1:  "#7F7F7F",
}


# =============================================================================
# VariableConfig  (simplified for xsec notebook)
# =============================================================================

class VariableConfig:
    """Binning + labels for a single kinematic variable."""
    def __init__(self, name, bins, xlabel, xlabel_reco=None, xlabel_true=None,
                 unit='', save_name=None):
        self.name       = name
        self.bins       = np.asarray(bins, dtype=float)
        self.n_bins     = len(self.bins) - 1
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.bin_widths  = np.diff(self.bins)
        self.xlabel      = xlabel
        self.xlabel_reco = xlabel_reco or xlabel
        self.xlabel_true = (xlabel_true or xlabel).replace('Reco', 'True')
        self.unit        = unit
        self.save_name   = save_name or name
        # For cafpyana compatibility
        self.var_labels   = [xlabel, self.xlabel_reco, self.xlabel_true]
        self.var_plot_name = xlabel


# Pre-built configs
def var_reco_ke():
    return VariableConfig(
        'reco_ke', SYST_KE_BINS,
        r'Reco leading-$e^-$ KE [MeV]',
        save_name='reco_ke')

def var_reco_costheta():
    return VariableConfig(
        'reco_costheta', SYST_COS_BINS,
        r'Reco leading-$e^-$ $\cos\theta$',
        save_name='reco_costheta')

def var_lynn_energy():
    return VariableConfig(
        'reco_ke', LYNN_ENERGY_BINS_MEV,
        r'Electron Energy $E_{e^-}$ [MeV]',
        save_name='electron_energy_lynn')

def var_lynn_costheta():
    return VariableConfig(
        'reco_costheta', LYNN_COS_BINS,
        r'Electron Direction $\cos\theta$',
        save_name='electron_costheta_lynn')


# =============================================================================
# NueCCBinning2D  (unchanged from v1)
# =============================================================================

class NueCCBinning2D:
    def __init__(self, costheta_bins=None, ke_bins_2d=None):
        self.diff_costheta_bins    = np.array(costheta_bins or NUE_COSTHETA_BINS)
        self.diff_momentum_bins_2d = np.array(ke_bins_2d or NUE_KE_BINS_2D, dtype=float)
        self.n_costheta_bins = len(self.diff_costheta_bins) - 1
        assert self.diff_momentum_bins_2d.shape[0] == self.n_costheta_bins

    @property
    def costheta_labels(self):
        lo = self.diff_costheta_bins[:-1]
        hi = np.minimum(self.diff_costheta_bins[1:], 1.0)
        return [f'${l:.2f} < \\cos\\theta_e < {h:.2f}$' for l, h in zip(lo, hi)]

def make_nuecc_binning2d(costheta_bins=None, ke_bins_2d=None):
    return NueCCBinning2D(costheta_bins=costheta_bins, ke_bins_2d=ke_bins_2d)


# =============================================================================
# Equal-stats binning
# =============================================================================

def make_equal_stats_bins(reference_vals, n_bins, lo=None, hi=None):
    vals = np.asarray(reference_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if lo is not None: vals = vals[vals >= lo]
    if hi is not None: vals = vals[vals < hi]
    if len(vals) == 0:
        raise ValueError("No finite values in [lo, hi)")
    pcts = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(vals, pcts)
    if lo is not None: edges[0] = lo
    if hi is not None: edges[-1] = hi
    edges = np.unique(edges)
    if len(edges) < n_bins + 1:
        warnings.warn(f"Only {len(edges)-1} unique edges (requested {n_bins})")
    return edges


# =============================================================================
# Response matrix, efficiency, smearing matrix
# =============================================================================

def build_response_and_efficiency(sel_topo, stage_col, reco_col, true_col,
                                  reco_bins, true_bins, pot_scale=1.0):
    """Build response matrix R_{aμ} = ε_μ * M_{aμ} and efficiency vector.

    Parameters
    ----------
    sel_topo : DataFrame with selection booleans, truth_cat, reco/true columns
    stage_col : str, e.g. 'sel_vertex_distance'
    reco_col, true_col : str column names
    reco_bins, true_bins : 1D arrays

    Returns
    -------
    dict with keys: response, smearing, efficiency, reco_vs_true,
                    sig_reco_cv, bkg_reco_cv, sig_true_cv
    """
    eps = 1e-8
    sel = sel_topo[sel_topo[stage_col]].copy()
    sig = (sel['truth_cat'] == 0)
    bkg = sel['truth_cat'].between(1, 6)

    # Signal reco and true
    sig_reco = sel.loc[sig, reco_col].dropna().clip(reco_bins[0], reco_bins[-1] - eps)
    sig_true = sel.loc[sig, true_col].dropna().clip(true_bins[0], true_bins[-1] - eps)
    bkg_reco = sel.loc[bkg, reco_col].dropna().clip(reco_bins[0], reco_bins[-1] - eps)

    # CV histograms (weighted)
    sig_reco_cv, _ = np.histogram(sig_reco, bins=reco_bins,
                                   weights=np.full(len(sig_reco), pot_scale))
    bkg_reco_cv, _ = np.histogram(bkg_reco, bins=reco_bins,
                                   weights=np.full(len(bkg_reco), pot_scale))

    # Truth histogram for efficiency denominator (all true signal, not just selected)
    all_sig = sel_topo[sel_topo['is_sig']]
    all_true = all_sig[true_col].dropna().clip(true_bins[0], true_bins[-1] - eps)
    sig_true_all, _ = np.histogram(all_true, bins=true_bins)

    # Selected signal truth histogram
    sig_true_sel, _ = np.histogram(sig_true, bins=true_bins)

    # Efficiency
    eff = np.divide(sig_true_sel, sig_true_all,
                    out=np.zeros(len(true_bins)-1, dtype=float),
                    where=sig_true_all > 0)

    # 2D reco-vs-true (smearing matrix, unnormalized)
    valid = sig & sel[reco_col].notna() & sel[true_col].notna()
    reco_vs_true, _, _ = np.histogram2d(
        sel.loc[valid, true_col].clip(true_bins[0], true_bins[-1] - eps),
        sel.loc[valid, reco_col].clip(reco_bins[0], reco_bins[-1] - eps),
        bins=[true_bins, reco_bins])

    # Smearing matrix M (row-normalized)
    row_sums = reco_vs_true.sum(axis=1, keepdims=True)
    smearing = np.divide(reco_vs_true, row_sums,
                         out=np.zeros_like(reco_vs_true),
                         where=row_sums > 0)

    # Response matrix R = ε × M
    response = eff[:, np.newaxis] * smearing

    return dict(
        response=response, smearing=smearing, efficiency=eff,
        reco_vs_true=reco_vs_true,
        sig_reco_cv=sig_reco_cv, bkg_reco_cv=bkg_reco_cv,
        sig_true_cv=sig_true_all.astype(float),
        sig_true_sel=sig_true_sel.astype(float),
    )


# =============================================================================
# Covariance loading from saved_syst/ NPZ files
# =============================================================================

def load_syst_covariances(syst_dir, var_name, sources=None):
    """Load per-source covariance matrices from the systematics notebook output.

    Parameters
    ----------
    syst_dir : str, e.g. './saved_syst'
    var_name : str, e.g. 'reco_ke' or 'reco_costheta'
    sources  : list of str, e.g. ['flux','genie','extra_xsec','g4','mcstat','pot','ntargets']
               If None, loads all .npz files found.

    Returns
    -------
    dict: {source_name: {'cov_ms_ms': array, 'cov_bs_bs': array, ...}}
    """
    vdir = os.path.join(syst_dir, var_name)
    if not os.path.isdir(vdir):
        warnings.warn(f"Syst dir not found: {vdir}")
        return {}

    result = {}
    if sources is None:
        sources = []
        for f in os.listdir(vdir):
            if f.endswith('_cov_matrices.npz') and not f.startswith('total'):
                src = f.replace('_cov_matrices.npz', '')
                sources.append(src)

    for src in sources:
        fpath = os.path.join(vdir, f'{src}_cov_matrices.npz')
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        entry = {}
        for key in ['cov_ms_ms', 'cov_bs_bs', 'cov_ms_bs', 'cov_bs_ms']:
            if key in d:
                v = d[key]
                # Handle dict-wrapped entries
                if v.ndim == 0:
                    v = v.item()
                    if isinstance(v, dict) and 'cov' in v:
                        v = v['cov']
                entry[key] = np.asarray(v, dtype=float)
        if entry:
            result[src] = entry
    return result


def build_total_covariance(cov_dict, block='cov_ms_ms'):
    """Sum covariance matrices from all sources."""
    matrices = [v[block] for v in cov_dict.values() if block in v]
    if not matrices:
        return None
    return sum(matrices)


# =============================================================================
# CCBC: Conditional Covariance Background Constraint
# (Lynn's technote Sec 5, Eqs 9-19)
# =============================================================================

def ccbc_constrain(
    B_S_cv, phi_S_cv,
    n_C_cv,
    D_C,
    V_BS_BS, V_phiS_phiS,
    V_BS_nC, V_phiS_nC,
    V_nC_nC,
    V_phiS_BS=None,
    D_C_stat=None,
):
    """Apply CCBC to constrain background and total prediction in signal region.

    Parameters
    ----------
    B_S_cv      : 1D array, CV background in signal region
    phi_S_cv    : 1D array, CV signal in signal region
    n_C_cv      : 1D array, CV total (sig+bkg) in control region
    D_C         : 1D array, measured data in control region
    V_BS_BS     : 2D, covariance of B_S with B_S
    V_phiS_phiS : 2D, covariance of phi_S with phi_S
    V_BS_nC     : 2D, cross-covariance of B_S with n_C
    V_phiS_nC   : 2D, cross-covariance of phi_S with n_C
    V_nC_nC     : 2D, covariance of n_C with n_C
    V_phiS_BS   : 2D, cross-covariance of phi_S with B_S (optional)
    D_C_stat    : 1D, data statistical errors in control region

    Returns
    -------
    dict with keys:
        B_constr     : constrained background prediction
        V_BS_constr  : constrained background covariance
        V_phiS_BS_constr : constrained signal-background cross-covariance
        m_S          : constrained total prediction (phi_S_cv + B_constr)
        V_mS_mS      : constrained total covariance
    """
    n_S = len(B_S_cv)
    n_C = len(n_C_cv)

    if V_phiS_BS is None:
        V_phiS_BS = np.zeros((n_S, n_S))

    # Eq 12: add data stat uncertainty to control region covariance
    V_nC_tilde = V_nC_nC.copy()
    if D_C_stat is not None:
        V_nC_tilde += np.diag(D_C_stat ** 2)
    else:
        V_nC_tilde += np.diag(np.maximum(D_C, 1.0))  # Poisson

    # Invert
    try:
        V_nC_inv = np.linalg.inv(V_nC_tilde)
    except np.linalg.LinAlgError:
        V_nC_inv = np.linalg.pinv(V_nC_tilde)

    # Eq 13: constrained background
    delta_C = D_C - n_C_cv
    B_constr = B_S_cv + V_BS_nC @ V_nC_inv @ delta_C

    # Eq 14: constrained background covariance
    V_BS_constr = V_BS_BS - V_BS_nC @ V_nC_inv @ V_BS_nC.T

    # Eq 15: constrained signal-background cross-covariance
    V_phiS_BS_constr = V_phiS_BS - V_phiS_nC @ V_nC_inv @ V_BS_nC.T

    # Eq 16: constrained total
    m_S = phi_S_cv + B_constr

    # Eq 19: constrained total covariance
    V_mS_mS = (V_phiS_phiS
               + V_phiS_BS_constr
               + V_phiS_BS_constr.T
               + V_BS_constr)

    return dict(
        B_constr=B_constr,
        V_BS_constr=V_BS_constr,
        V_phiS_BS_constr=V_phiS_BS_constr,
        m_S=m_S,
        V_mS_mS=V_mS_mS,
    )


def build_ccbc_covariances_from_universes(
    sel_topo, sb_sel, stage_col,
    reco_col, reco_bins,
    wgt_aligned, univ_cols_dict,
    pot_scale, sb_pot_scale=None,
):
    """Build all covariance blocks needed for CCBC from universe weights.

    This constructs V(B_S, B_S), V(phi_S, phi_S), V(B_S, n_C), V(phi_S, n_C),
    and V(n_C, n_C) from systematic universe histograms.

    Parameters
    ----------
    sel_topo : DataFrame, signal region selected events
    sb_sel   : DataFrame, sideband/control region selected events
    stage_col : str, final selection column
    reco_col  : str, reco variable column
    reco_bins : 1D array
    wgt_aligned : DataFrame of universe weights
    univ_cols_dict : dict {source: list_of_columns}
    pot_scale : float

    Returns
    -------
    dict of covariance matrices
    """
    eps = 1e-8
    nb = len(reco_bins) - 1
    if sb_pot_scale is None:
        sb_pot_scale = pot_scale

    # CV histograms
    sel = sel_topo[sel_topo[stage_col]]
    sig_mask = (sel['truth_cat'] == 0).values
    bkg_mask = sel['truth_cat'].between(1, 6).values
    reco_vals = sel[reco_col].clip(reco_bins[0], reco_bins[-1] - eps).values

    sig_cv, _ = np.histogram(reco_vals[sig_mask], bins=reco_bins,
                              weights=np.full(sig_mask.sum(), pot_scale))
    bkg_cv, _ = np.histogram(reco_vals[bkg_mask], bins=reco_bins,
                              weights=np.full(bkg_mask.sum(), pot_scale))

    # Control region CV
    sb_reco = sb_sel[reco_col].clip(reco_bins[0], reco_bins[-1] - eps).dropna().values
    nc_cv, _ = np.histogram(sb_reco, bins=reco_bins,
                             weights=np.full(len(sb_reco), sb_pot_scale))

    # Accumulate covariances
    V_ss = np.zeros((nb, nb))
    V_bb = np.zeros((nb, nb))
    V_sb = np.zeros((nb, nb))
    V_bn = np.zeros((nb, nb))
    V_sn = np.zeros((nb, nb))
    V_nn = np.zeros((nb, nb))
    n_total = 0

    for src, cols in univ_cols_dict.items():
        if not cols:
            continue
        for col in cols:
            # Signal region
            w = wgt_aligned[col].reindex(sel.index).fillna(1.0).values
            w = np.clip(w, 0, 10) * pot_scale
            su, _ = np.histogram(reco_vals[sig_mask], bins=reco_bins,
                                  weights=w[sig_mask])
            bu, _ = np.histogram(reco_vals[bkg_mask], bins=reco_bins,
                                  weights=w[bkg_mask])
            # Control region (simplified — would need sb weights)
            nu = nc_cv.copy()  # Placeholder

            ds = su - sig_cv
            db = bu - bkg_cv
            dn = nu - nc_cv

            V_ss += np.outer(ds, ds)
            V_bb += np.outer(db, db)
            V_sb += np.outer(ds, db)
            V_bn += np.outer(db, dn)
            V_sn += np.outer(ds, dn)
            V_nn += np.outer(dn, dn)
            n_total += 1

    if n_total > 0:
        for V in [V_ss, V_bb, V_sb, V_bn, V_sn, V_nn]:
            V /= n_total

    return dict(
        V_phiS_phiS=V_ss, V_BS_BS=V_bb, V_phiS_BS=V_sb,
        V_BS_nC=V_bn, V_phiS_nC=V_sn, V_nC_nC=V_nn,
        sig_cv=sig_cv, bkg_cv=bkg_cv, nc_cv=nc_cv,
    )


# =============================================================================
# Cross-section extraction  (Lynn's Eq 20)
# =============================================================================

def extract_differential_xsec(unfolded_signal, bins, target_pot,
                               n_targets=None, flux=None):
    """Compute flux-averaged differential cross-section.

    dσ/dx_μ = Σ_a U_μa (m_a - B_a) / (Φ · N_targets · Δx_μ)

    Parameters
    ----------
    unfolded_signal : 1D array, background-subtracted unfolded signal in true bins
    bins : 1D array, true bin edges
    target_pot : float, POT for normalization
    n_targets : float (default: from get_constants())
    flux : float (default: from get_constants())

    Returns
    -------
    xsec : 1D array, differential cross-section per bin
    """
    if n_targets is None or flux is None:
        nt, fl = get_constants()
        if n_targets is None: n_targets = nt
        if flux is None: flux = fl

    bin_widths = np.diff(bins)
    denominator = flux * target_pot * n_targets * bin_widths
    xsec = unfolded_signal / np.where(denominator > 0, denominator, 1.0)
    return xsec


# =============================================================================
# Fake data test driver  (with optional CCBC)
# =============================================================================

def run_fake_data_test(sel_topo, stage_col, reco_col, true_col,
                       reco_bins, true_bins,
                       response, model_truth, cov_reco,
                       scale_dict, label,
                       pot_scale,
                       ccbc_result=None,
                       WienerSVD_fn=None,
                       C_type=2, Norm_type=0):
    """Run a fake data test with optional CCBC.

    Parameters
    ----------
    sel_topo : DataFrame
    scale_dict : {truth_cat: weight_factor}
    ccbc_result : dict from ccbc_constrain (if using CCBC)
    WienerSVD_fn : callable (from cafpyana)

    Returns
    -------
    dict with unfolding results
    """
    eps = 1e-8
    sel = sel_topo[sel_topo[stage_col]].copy()
    sig = (sel['truth_cat'] == 0)
    bkg = sel['truth_cat'].between(1, 6)

    # Build fake data weights
    wgt = np.full(len(sel), pot_scale, dtype=float)
    for cat, scale in scale_dict.items():
        wgt[sel['truth_cat'].values == cat] *= scale

    reco_vals = sel[reco_col].clip(reco_bins[0], reco_bins[-1] - eps).values

    fake_sig, _ = np.histogram(reco_vals[sig], bins=reco_bins, weights=wgt[sig.values])
    fake_bkg, _ = np.histogram(reco_vals[bkg], bins=reco_bins, weights=wgt[bkg.values])

    # CV background
    cv_bkg, _ = np.histogram(reco_vals[bkg], bins=reco_bins,
                              weights=np.full(bkg.sum(), pot_scale))

    # Background-subtracted measurement
    if ccbc_result is not None:
        bkg_pred = ccbc_result['B_constr']
        cov_use = ccbc_result['V_mS_mS']
    else:
        bkg_pred = cv_bkg
        cov_use = cov_reco

    measured = fake_sig + fake_bkg - bkg_pred

    # Alternative truth
    all_sig = sel_topo[sel_topo['is_sig']]
    true_vals = all_sig[true_col].dropna().clip(true_bins[0], true_bins[-1] - eps)

    alt_truth_wgt = np.ones(len(true_vals))
    if 0 in scale_dict:
        alt_truth_wgt *= scale_dict[0]
    alt_truth, _ = np.histogram(true_vals, bins=true_bins, weights=alt_truth_wgt)

    result = dict(
        measured=measured, fake_sig=fake_sig, fake_bkg=fake_bkg,
        bkg_pred=bkg_pred, alt_truth=alt_truth.astype(float),
        label=label,
    )

    if WienerSVD_fn is not None:
        unfold = WienerSVD_fn(response, model_truth, measured, cov_use,
                               C_type=C_type, Norm_type=Norm_type)
        result['unfold'] = unfold

    return result


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_heatmap(matrix, title, vcfg, axis_labels=None,
                 fmt=".2e", cmap="viridis", save_path=None):
    """Plot a covariance/correlation matrix heatmap."""
    import re
    try:
        from analysis_village.unfolding.utils import bin_range_labels, get_text_color
        raw = bin_range_labels(vcfg.bins)
        tick_labels = [re.sub(r'(\d+)\.0+(?!\d)', r'\1', l) for l in raw]
        _get_tc = get_text_color
    except Exception:
        tick_labels = [f'{vcfg.bins[i]:.0f}-{vcfg.bins[i+1]:.0f}'
                       for i in range(vcfg.n_bins)]
        _get_tc = lambda v: 'white'

    n = vcfg.n_bins
    fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
    unif = np.arange(n + 1, dtype=float)
    tick_pos = 0.5 * (unif[:-1] + unif[1:])
    extent = [unif[0], unif[-1], unif[0], unif[-1]]

    im = ax.imshow(matrix, extent=extent, origin='lower',
                   cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=9)

    xlbl, ylbl = axis_labels if axis_labels else [vcfg.xlabel] * 2
    ax.set_xlabel(xlbl, fontsize=11)
    ax.set_ylabel(ylbl, fontsize=11)
    ax.set_title(title, fontsize=13)

    # Cell annotations
    def _fmt_val(v):
        if np.isnan(v): return ""
        av = abs(v)
        if av < 1: return f"{v:.4f}"
        elif av < 1e3: return f"{v:.2f}"
        m, e = f"{v:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"

    fsize = max(5, 9 - n // 3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                try: tc = _get_tc(v)
                except: tc = 'white'
                ax.text(tick_pos[j], tick_pos[i], _fmt_val(v),
                        ha='center', va='center', color=tc, fontsize=fsize)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_uncertainty_budget(vcfg, cv_total, cov_dict, title="Uncertainty Budget",
                            save_path=None):
    """Fractional uncertainty per source as step-function lines."""
    bins = vcfg.bins
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 5))
    total_cov = np.zeros((vcfg.n_bins, vcfg.n_bins))

    for i, (label, cov) in enumerate(cov_dict.items()):
        total_cov += cov
        unc = np.sqrt(np.diag(cov).clip(0))
        frac = np.divide(unc, cv_total, out=np.zeros_like(unc),
                         where=cv_total > 0)
        if np.all(frac == 0): continue
        ax.step(bins, np.append(frac, frac[-1]), where="post",
                label=label, lw=1.8, color=colors[i % len(colors)])

    tot_unc = np.sqrt(np.diag(total_cov).clip(0))
    tot_frac = np.divide(tot_unc, cv_total, out=np.zeros_like(tot_unc),
                         where=cv_total > 0)
    ax.step(bins, np.append(tot_frac, tot_frac[-1]), where="post",
            label="Total", color="black", ls="--", lw=2.5)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(vcfg.xlabel, fontsize=12)
    ax.set_ylabel(r"Fractional unc. $\sigma / N$", fontsize=12)
    ax.set_title(f"{title}: {vcfg.save_name}", fontsize=13)
    ax.set_ylim(0, max(tot_frac.max() * 1.5, 0.05))
    ax.legend(fontsize=10, frameon=False)
    ax.grid(alpha=0.2, ls="--")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_unfolded_result(vcfg, unfold_result, model, reco_signal=None,
                          title="Unfolded Result", save_path=None):
    """Unfolded points with error bars vs Ac × truth model."""
    bins = vcfg.bins
    centers = vcfg.bin_centers
    U = unfold_result["unfold"]
    Uc = unfold_result["UnfoldCov"]
    Uerr = np.sqrt(np.diag(Uc))
    Ac = unfold_result["AddSmear"]
    model_smear = Ac @ model

    chi2_val = float((U - model_smear) @ np.linalg.inv(Uc) @ (U - model_smear))
    ndf = len(U)
    pval = 1 - scipy_chi2.cdf(chi2_val, ndf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(centers, U, yerr=Uerr, fmt="ko", capsize=3, elinewidth=1.5,
                markersize=5, label="Unfolded")
    ax.step(bins, np.append(model_smear, model_smear[-1]),
            where="post", color="crimson", lw=2, label=r"$A_c \times$ Truth")
    if reco_signal is not None:
        ax.plot(centers, reco_signal, "o", color="steelblue", alpha=0.7,
                ms=5, label="Reco. signal")

    chi2_str = rf"$\chi^2/\mathrm{{ndf}} = {chi2_val:.2f}/{ndf}$  $(p = {pval:.2f})$"
    ax.text(0.05, 0.95, chi2_str, transform=ax.transAxes, fontsize=12,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", alpha=0.6, ec="none"))

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(vcfg.xlabel, fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.2, ls="--")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_xsec_comparison(vcfg, xsec_wiener, xsec_unc_wiener,
                          xsec_omnifold=None, xsec_unc_omnifold=None,
                          truth_xsec=None,
                          title=r"SBND $\nu_e$ CC Inclusive",
                          save_path=None):
    """Plot cross-section with Wiener-SVD and optional OmniFold comparison."""
    bins = vcfg.bins
    centers = vcfg.bin_centers

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    ax_main, ax_ratio = axes

    # Truth
    if truth_xsec is not None:
        ax_main.step(bins, np.append(truth_xsec, truth_xsec[-1]),
                     where='post', color='black', lw=2, label='MC Truth')

    # Wiener-SVD
    ax_main.errorbar(centers - 0.01 * np.diff(bins), xsec_wiener,
                     yerr=xsec_unc_wiener, fmt='s', color='steelblue',
                     capsize=3, markersize=5, lw=1.5, label='Wiener-SVD')

    # OmniFold
    if xsec_omnifold is not None:
        ax_main.errorbar(centers + 0.01 * np.diff(bins), xsec_omnifold,
                         yerr=xsec_unc_omnifold, fmt='o', color='red',
                         capsize=3, markersize=5, lw=1.5, label='OmniFold')

    ax_main.set_ylabel(r'd$\sigma$/d$x$ [arb.]', fontsize=12)
    ax_main.set_title(title, fontsize=13)
    ax_main.legend(fontsize=10, frameon=False)
    ax_main.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Ratio
    if truth_xsec is not None:
        ref = np.where(truth_xsec > 0, truth_xsec, 1.0)
        r_w = xsec_wiener / ref
        r_w_err = xsec_unc_wiener / ref
        ax_ratio.errorbar(centers - 0.01 * np.diff(bins), r_w, yerr=r_w_err,
                          fmt='s', color='steelblue', capsize=3, markersize=5)
        if xsec_omnifold is not None:
            r_o = xsec_omnifold / ref
            r_o_err = (xsec_unc_omnifold or np.zeros_like(xsec_omnifold)) / ref
            ax_ratio.errorbar(centers + 0.01 * np.diff(bins), r_o, yerr=r_o_err,
                              fmt='o', color='red', capsize=3, markersize=5)

    ax_ratio.axhline(1.0, color='gray', ls='--', lw=1)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(vcfg.xlabel, fontsize=12)
    ax_ratio.set_ylabel('Ratio to Truth', fontsize=11)
    ax_ratio.set_xlim(bins[0], bins[-1])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, axes


# =============================================================================
# NPZ archiving  (save_cov_npz from v1, unchanged)
# =============================================================================

def save_cov_npz(filename, response=None, true_signal=None,
                  measured_signal=None, measured_background=None,
                  cov_ms_ms=None, cov_bs_bs=None,
                  cov_ms_bs=None, cov_bs_ms=None, **extra):
    arrays = {}
    if response            is not None: arrays["response"]  = response
    if true_signal         is not None: arrays["true_signal"] = true_signal
    if measured_signal     is not None: arrays["ms"]         = measured_signal
    if measured_background is not None: arrays["bs"]         = measured_background
    if cov_ms_ms           is not None: arrays["cov_ms_ms"]  = np.array(cov_ms_ms)
    if cov_bs_bs           is not None: arrays["cov_bs_bs"]  = np.array(cov_bs_bs)
    if cov_ms_bs           is not None: arrays["cov_ms_bs"]  = np.array(cov_ms_bs)
    if cov_bs_ms           is not None: arrays["cov_bs_ms"]  = np.array(cov_bs_ms)
    arrays.update(extra)
    np.savez_compressed(filename, **arrays)
    print(f"Saved {len(arrays)} arrays → {filename}")