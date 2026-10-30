"""
nue_xsec_helpers.py  (v3)
-------------------------
Additions over v2:
  - FDT helper functions (tilt, interaction-mode scaling)
  - Comprehensive Wiener-SVD vs OmniFold comparison utilities
  - Cleaned up: diagnostics as printout, plots as plots
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2 as scipy_chi2
from collections import defaultdict


# =============================================================================
# Physical constants
# =============================================================================

V_FV_R1 = 2 * 185.0 * 380.0 * 240.0
V_FV_R2 = 185.0 * 290.0 * 200.0
V_FV_R3 = 185.0 * 380.0 * 200.0
V_FV = V_FV_R1 + V_FV_R2 + V_FV_R3

RHO_LAR     = 1.3954
N_AVOGADRO  = 6.02214076e23
M_AR        = 39.948
N_TARGETS   = RHO_LAR * V_FV * N_AVOGADRO / M_AR

LYNN_RHO       = 1.38434
LYNN_V_FV      = 58_534_000
LYNN_N_TARGETS = LYNN_RHO * LYNN_V_FV * N_AVOGADRO / M_AR
LYNN_FLUX      = 9.65e-11

USE_LYNN_CONSTANTS = False

def get_constants():
    if USE_LYNN_CONSTANTS:
        return LYNN_N_TARGETS, LYNN_FLUX
    return N_TARGETS, LYNN_FLUX


# =============================================================================
# Binning
# =============================================================================

LYNN_ENERGY_BINS_GEV = np.array([0.5, 0.7, 0.95, 1.25, 1.7, 2.5])
LYNN_COS_BINS        = np.array([-1.0, 0.6, 0.75, 0.85, 0.925, 1.0])
LYNN_ENERGY_BINS_MEV = LYNN_ENERGY_BINS_GEV * 1000.0

SYST_KE_BINS  = np.linspace(0, 2000, 8)
SYST_COS_BINS = np.linspace(-1, 1, 11)

NUE_COSTHETA_BINS = np.array([-1.0, 0.0, 0.5, 0.75, 0.85, 0.95, 1.001])
NUE_KE_BINS_2D = np.array([
    [0, 300, 700, 1200, 2000.],
    [0, 200, 500, 1000, 2000.],
    [0, 200, 500, 1000, 2000.],
    [0, 150, 400,  900, 2000.],
    [0, 150, 400,  900, 2000.],
    [0, 150, 400,  900, 2000.],
], dtype=float)


class NueCCBinning2D:
    def __init__(self, costheta_bins=None, ke_bins_2d=None):
        self.diff_costheta_bins = np.array(costheta_bins or NUE_COSTHETA_BINS)
        self.diff_momentum_bins_2d = np.array(ke_bins_2d or NUE_KE_BINS_2D, dtype=float)
        self.n_costheta_bins = len(self.diff_costheta_bins) - 1

    @property
    def costheta_labels(self):
        lo = self.diff_costheta_bins[:-1]
        hi = np.minimum(self.diff_costheta_bins[1:], 1.0)
        return [f'${l:.2f} < \\cos\\theta_e < {h:.2f}$' for l, h in zip(lo, hi)]

def make_nuecc_binning2d(costheta_bins=None, ke_bins_2d=None):
    return NueCCBinning2D(costheta_bins=costheta_bins, ke_bins_2d=ke_bins_2d)
# =============================================================================
# Categories
# =============================================================================

CAT_LABELS = {
    0: r"$\nu_e$ CC (FV)", 1: r"$\nu_e$ CC (out FV)",
    2: r"$\nu_\mu$ CC $\pi^0$", 3: r"$\nu$ NC $\pi^0$",
    4: r"Other $\nu_\mu$ CC", 5: r"Other $\nu$ NC",
    6: r"Cosmic", 7: r"Offbeam", 9: r"Dirt $\nu$", -1: r"Unknown",
}
CAT_COLORS = {
    0: "#56B4E9", 1: "#0072B2", 2: "green", 3: "#009E73",
    4: "#E69F00", 5: "#CC79A7", 6: "#F0E442", 7: "#BDBDBD",
    9: "#D2691E", -1: "#7F7F7F",
}


# =============================================================================
# VariableConfig
# =============================================================================

class VariableConfig:
    def __init__(self, name, bins, xlabel, xlabel_reco=None, xlabel_true=None,
                 unit='', save_name=None):
        self.name        = name
        self.bins        = np.asarray(bins, dtype=float)
        self.n_bins      = len(self.bins) - 1
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.bin_widths  = np.diff(self.bins)
        self.xlabel      = xlabel
        self.xlabel_reco = xlabel_reco or xlabel
        self.xlabel_true = (xlabel_true or xlabel).replace('Reco', 'True')
        self.unit        = unit
        self.save_name   = save_name or name
        self.var_labels  = [xlabel, self.xlabel_reco, self.xlabel_true]
        self.var_plot_name = xlabel


# =============================================================================
# Covariance loading
# =============================================================================

def load_syst_covariances(syst_dir, var_name, sources=None):
    vdir = os.path.join(syst_dir, var_name)
    if not os.path.isdir(vdir):
        warnings.warn(f"Syst dir not found: {vdir}")
        return {}
    result = {}
    if sources is None:
        sources = []
        for f in os.listdir(vdir):
            if f.endswith('_cov_matrices.npz') and not f.startswith('total'):
                sources.append(f.replace('_cov_matrices.npz', ''))
    for src in sources:
        fpath = os.path.join(vdir, f'{src}_cov_matrices.npz')
        if not os.path.exists(fpath): continue
        d = np.load(fpath, allow_pickle=True)
        entry = {}
        for key in ['cov_ms_ms', 'cov_bs_bs', 'cov_ms_bs', 'cov_bs_ms']:
            if key in d:
                v = d[key]
                if v.ndim == 0:
                    v = v.item()
                    if isinstance(v, dict) and 'cov' in v:
                        v = v['cov']
                entry[key] = np.asarray(v, dtype=float)
        if entry:
            result[src] = entry
    return result


def build_total_covariance(cov_dict, block='cov_ms_ms'):
    matrices = [v[block] for v in cov_dict.values() if block in v]
    if not matrices: return None
    return sum(matrices)


# =============================================================================
# CCBC
# =============================================================================

def ccbc_constrain(B_S_cv, phi_S_cv, n_C_cv, D_C,
                   V_BS_BS, V_phiS_phiS, V_BS_nC, V_phiS_nC, V_nC_nC,
                   V_phiS_BS=None, D_C_stat=None):
    n_S = len(B_S_cv)
    if V_phiS_BS is None:
        V_phiS_BS = np.zeros((n_S, n_S))
    V_nC_tilde = V_nC_nC.copy()
    if D_C_stat is not None:
        V_nC_tilde += np.diag(D_C_stat ** 2)
    else:
        V_nC_tilde += np.diag(np.maximum(D_C, 1.0))
    try:
        V_nC_inv = np.linalg.inv(V_nC_tilde)
    except np.linalg.LinAlgError:
        V_nC_inv = np.linalg.pinv(V_nC_tilde)
    delta_C = D_C - n_C_cv
    B_constr = B_S_cv + V_BS_nC @ V_nC_inv @ delta_C
    V_BS_constr = V_BS_BS - V_BS_nC @ V_nC_inv @ V_BS_nC.T
    V_phiS_BS_constr = V_phiS_BS - V_phiS_nC @ V_nC_inv @ V_BS_nC.T
    m_S = phi_S_cv + B_constr
    V_mS_mS = V_phiS_phiS + V_phiS_BS_constr + V_phiS_BS_constr.T + V_BS_constr
    return dict(B_constr=B_constr, V_BS_constr=V_BS_constr,
                V_phiS_BS_constr=V_phiS_BS_constr, m_S=m_S, V_mS_mS=V_mS_mS)


# =============================================================================
# Cross-section extraction
# =============================================================================

def extract_differential_xsec(unfolded_signal, bins, target_pot,
                               n_targets=None, flux=None):
    if n_targets is None or flux is None:
        nt, fl = get_constants()
        if n_targets is None: n_targets = nt
        if flux is None: flux = fl
    bin_widths = np.diff(bins)
    denom = flux * target_pot * n_targets * bin_widths
    return unfolded_signal / np.where(denom > 0, denom, 1.0)


# =============================================================================
# FDT helpers
# =============================================================================

def apply_tilt_weights(values, alpha, lo, hi):
    """Linear tilt: w = 1 + α·(x - x_mean)/x_range.
    Same as OmniFold MakePlots.py FDS."""
    x_mean = 0.5 * (lo + hi)
    x_range = hi - lo
    if x_range <= 0:
        return np.ones_like(values)
    return np.clip(1.0 + alpha * (values - x_mean) / x_range, 0.0, 10.0)


def find_interaction_mode_col(df):
    """Find the interaction mode column in a DataFrame, trying various names."""
    candidates = ['truth_intmode', 'intmode', 'interaction_mode',
                  'genie_intmode', 'truth_genie_intmode']
    for c in candidates:
        if c in df.columns:
            return c
    # Try multi-index
    for col in df.columns:
        s = str(col).lower()
        if 'intmode' in s or 'interaction_mode' in s:
            return col
    return None


def build_fdt_weights_intmode(sel, all_sig, mode_value, scale_factor):
    """Build event weights for interaction-mode FDT.

    Parameters
    ----------
    sel : DataFrame, selected events (signal + background)
    all_sig : DataFrame, all true signal events (for truth weights)
    mode_value : int (0=QE, 1=RES, 2=DIS, 3=COH, 10=MEC)
    scale_factor : float

    Returns
    -------
    evt_wgt : array, per-event weights for sel
    truth_wgt : array, per-event weights for all_sig
    """
    evt_wgt = np.ones(len(sel))
    truth_wgt = np.ones(len(all_sig))

    col = find_interaction_mode_col(sel)
    if col is not None:
        sig_m = (sel['truth_cat'] == 0).values
        evt_wgt[sig_m & (sel[col].values == mode_value)] = scale_factor
        # Also scale background events of this mode
        bkg_m = sel['truth_cat'].between(1, 6).values
        evt_wgt[bkg_m & (sel[col].values == mode_value)] = scale_factor

    tcol = find_interaction_mode_col(all_sig)
    if tcol is not None:
        truth_wgt[all_sig[tcol].values == mode_value] = scale_factor

    return evt_wgt, truth_wgt


def build_fdt_weights_tilt(sel, all_sig, true_col, bins, alpha):
    """Build event weights for OmniFold-style tilt FDT."""
    lo, hi = bins[0], bins[-1]
    evt_wgt = np.ones(len(sel))
    truth_wgt = np.ones(len(all_sig))

    sig_m = (sel['truth_cat'] == 0).values
    if true_col in sel.columns:
        vals = sel[true_col].clip(lo, hi).values
        evt_wgt[sig_m] = apply_tilt_weights(vals[sig_m], alpha, lo, hi)

    if true_col in all_sig.columns:
        truth_wgt = apply_tilt_weights(
            all_sig[true_col].clip(lo, hi).values, alpha, lo, hi)

    return evt_wgt, truth_wgt


def build_fdt_weights_pi0(sel, all_sig, scale_factor):
    """Scale events with primary π⁰ (truth_cat 2 or 3)."""
    evt_wgt = np.ones(len(sel))
    pi0_mask = sel['truth_cat'].isin([2, 3]).values
    evt_wgt[pi0_mask] = scale_factor

    truth_wgt = np.ones(len(all_sig))
    if 'has_pi0' in all_sig.columns:
        truth_wgt[all_sig['has_pi0'].values] = scale_factor
    return evt_wgt, truth_wgt


def build_fdt_weights_low_energy(sel, all_sig, true_col, threshold_mev, scale_factor):
    """Suppress signal events with true energy below threshold."""
    evt_wgt = np.ones(len(sel))
    truth_wgt = np.ones(len(all_sig))

    sig_m = (sel['truth_cat'] == 0).values
    if true_col in sel.columns:
        low = (sel[true_col].values < threshold_mev)
        evt_wgt[sig_m & low] = scale_factor

    if true_col in all_sig.columns:
        truth_wgt[all_sig[true_col].values < threshold_mev] = scale_factor

    return evt_wgt, truth_wgt


def run_single_fdt(sel_final, sel_topo, var_name, true_col, bins, true_bins,
                   Response, Model, Cov, evt_wgt, truth_wgt,
                   WienerSVD_fn, C_type=2, Norm_type=0):
    """Run one fake data test and return results dict.

    Returns
    -------
    dict with: unfold, c2_nom, c2_mod, p_nom, p_mod, ndf, alt_truth, Measured
    """
    _eps = 1e-8
    rv = sel_final[var_name].clip(bins[0], bins[-1]-_eps).values
    bm = sel_final['truth_cat'].between(1, 6).values

    fake_total, _ = np.histogram(rv, bins=bins, weights=evt_wgt)
    cv_bkg_raw, _ = np.histogram(rv[bm], bins=bins)
    Meas = (fake_total - cv_bkg_raw).astype(float)

    all_sig = sel_topo[sel_topo['is_sig']]
    tv = all_sig[true_col].dropna().clip(true_bins[0], true_bins[-1]-_eps)
    alt_truth, _ = np.histogram(tv, bins=true_bins,
                                 weights=truth_wgt[:len(tv)])

    uf = WienerSVD_fn(Response, Model, Meas, Cov, C_type=C_type, Norm_type=Norm_type)
    U = uf['unfold']; Uc = uf['UnfoldCov']; Ac = uf['AddSmear']
    ms_nom = Ac @ Model
    ms_alt = Ac @ alt_truth.astype(float)

    c2n = float((U - ms_nom) @ np.linalg.inv(Uc) @ (U - ms_nom))
    c2a = float((U - ms_alt) @ np.linalg.inv(Uc) @ (U - ms_alt))
    ndf = len(U)

    return dict(
        unfold=uf, alt_truth=alt_truth.astype(float),
        Measured=Meas, ms_nom=ms_nom, ms_alt=ms_alt,
        c2_nom=c2n, c2_mod=c2a, ndf=ndf,
        p_nom=1 - scipy_chi2.cdf(c2n, ndf),
        p_mod=1 - scipy_chi2.cdf(c2a, ndf),
    )


def plot_fdt(vcfg, result, label, true_label, save_path=None):
    """Plot a single FDT result (Lynn Fig 23/24 style)."""
    U = result['unfold']['unfold']
    Uc = result['unfold']['UnfoldCov']
    Uerr = np.sqrt(np.diag(Uc).clip(0))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(vcfg.bin_centers, U, yerr=Uerr, fmt='ko', capsize=3,
                ms=4, lw=1.2, label='Unfolded fake data')
    ax.step(vcfg.bins, np.append(result['ms_nom'], result['ms_nom'][-1]),
            where='post', color='#0072B2', lw=1.5, ls='--',
            label=rf"$A_c \otimes$ baseline ($\chi^2$={result['c2_nom']:.1f}/{result['ndf']}, "
                  rf"p={result['p_nom']:.2f})")
    ax.step(vcfg.bins, np.append(result['ms_alt'], result['ms_alt'][-1]),
            where='post', color='#D55E00', lw=2,
            label=rf"$A_c \otimes$ modified ($\chi^2$={result['c2_mod']:.1f}/{result['ndf']}, "
                  rf"p={result['p_mod']:.2f})")
    ax.fill_between(vcfg.bins,
                    np.append(result['ms_alt'] - Uerr, (result['ms_alt'] - Uerr)[-1]),
                    np.append(result['ms_alt'] + Uerr, (result['ms_alt'] + Uerr)[-1]),
                    step='post', alpha=0.1, color='gray', label='norm. unc.')
    ax.set_xlabel(true_label); ax.set_ylabel('Events')
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(vcfg.bins[0], vcfg.bins[-1])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


# =============================================================================
# Wiener-SVD vs OmniFold comparison
# =============================================================================

def compute_comparison_metrics(w_cov, of_cov, w_mean, of_mean):
    """Compute all comparison metrics between two covariance matrices.

    Returns dict of metrics for printout (not plotting).
    """
    n_w, n_o = w_cov.shape[0], of_cov.shape[0]

    w_dg = np.sqrt(np.diag(w_cov).clip(1e-10))
    of_dg = np.sqrt(np.diag(of_cov).clip(1e-10))
    w_corr = w_cov / np.outer(w_dg, w_dg)
    of_corr = of_cov / np.outer(of_dg, of_dg)

    w_cond = np.linalg.cond(w_cov)
    of_cond = np.linalg.cond(of_cov)

    w_eig = np.linalg.eigvalsh(w_cov)
    of_eig = np.linalg.eigvalsh(of_cov)
    w_edof = w_eig.sum() / w_eig.max()
    of_edof = of_eig.sum() / of_eig.max()

    w_offdiag = w_corr[np.triu_indices(n_w, k=1)]
    of_offdiag = of_corr[np.triu_indices(n_o, k=1)]

    n = min(n_w, n_o)
    w_frac = w_dg[:n] / np.abs(w_mean[:n]).clip(1)
    of_frac = of_dg[:n] / np.abs(of_mean[:n]).clip(1)

    # Shape-only uncertainty
    w_norm = w_mean[:n] / w_mean[:n].sum()
    of_norm = of_mean[:n] / of_mean[:n].sum()
    N_w, N_of = w_mean[:n].sum(), of_mean[:n].sum()
    J_w = np.eye(n)/N_w - np.outer(w_norm, np.ones(n))/N_w
    J_of = np.eye(n)/N_of - np.outer(of_norm, np.ones(n))/N_of
    w_shape_cov = J_w @ w_cov[:n,:n] @ J_w.T
    of_shape_cov = J_of @ of_cov[:n,:n] @ J_of.T
    w_shape = np.sqrt(np.diag(w_shape_cov).clip(0)) / w_norm.clip(1e-6) * 100
    of_shape = np.sqrt(np.diag(of_shape_cov).clip(0)) / of_norm.clip(1e-6) * 100

    return dict(
        w_cond=w_cond, of_cond=of_cond,
        w_edof=w_edof, of_edof=of_edof,
        n_w=n_w, n_o=n_o,
        w_mean_offdiag=np.abs(w_offdiag).mean(),
        of_mean_offdiag=np.abs(of_offdiag).mean(),
        w_frac_neg=(w_offdiag < 0).mean(),
        of_frac_neg=(of_offdiag < 0).mean(),
        w_total_frac=np.sqrt(np.sum(w_cov)) / np.abs(w_mean).sum() * 100,
        of_total_frac=np.sqrt(np.sum(of_cov)) / np.abs(of_mean).sum() * 100,
        w_frac=w_frac, of_frac=of_frac,
        w_shape=w_shape, of_shape=of_shape,
        w_corr=w_corr, of_corr=of_corr,
        w_eig=np.sort(w_eig)[::-1], of_eig=np.sort(of_eig)[::-1],
    )


def print_comparison_metrics(metrics, var_name):
    """Print comparison metrics as text table."""
    m = metrics
    print(f"\n{'='*55}")
    print(f"  Wiener-SVD vs OmniFold: {var_name}")
    print(f"{'='*55}")
    print(f"  {'Metric':<26} {'Wiener-SVD':>12} {'OmniFold':>12}")
    print(f"  {'─'*50}")
    print(f"  {'Condition number':<26} {m['w_cond']:>12.1f} {m['of_cond']:>12.1f}")
    print(f"  {'Effective DoF':<26} {m['w_edof']:>12.2f} {m['of_edof']:>12.2f}")
    print(f"  {'  (out of n_bins)':<26} {m['n_w']:>12d} {m['n_o']:>12d}")
    print(f"  {'Mean |off-diag corr|':<26} {m['w_mean_offdiag']:>12.3f} {m['of_mean_offdiag']:>12.3f}")
    print(f"  {'Frac negative corr':<26} {m['w_frac_neg']:>12.2f} {m['of_frac_neg']:>12.2f}")
    print(f"  {'Total frac unc [%]':<26} {m['w_total_frac']:>12.1f} {m['of_total_frac']:>12.1f}")
    print(f"  {'─'*50}")
    print(f"  Per-bin frac unc [%]:")
    print(f"    Wiener-SVD: {np.round(m['w_frac']*100, 1)}")
    print(f"    OF:    {np.round(m['of_frac']*100, 1)}")
    print(f"  Shape-only unc [%]:")
    print(f"    Wiener-SVD: {np.round(m['w_shape'], 1)}")
    print(f"    OF:    {np.round(m['of_shape'], 1)}")

    # Interpretation
    print(f"\n  Interpretation:")
    if m['of_cond'] > 1e4:
        print(f"    ⚠ OF cov near-singular (cond={m['of_cond']:.0f}) — χ² tests unreliable")
    if m['of_edof'] < 2:
        print(f"    ⚠ OF eff. DoF < 2 — differential shape info very limited")
    if m['w_edof'] > m['of_edof']:
        print(f"    → Wiener-SVD extracts more independent info ({m['w_edof']:.1f} vs {m['of_edof']:.1f} eff. DoF)")
    if m['of_total_frac'] < m['w_total_frac']:
        print(f"    → OF has lower total unc ({m['of_total_frac']:.1f}% vs {m['w_total_frac']:.1f}%)")


def plot_comparison_correlations(metrics, true_label, save_path=None):
    """Plot side-by-side correlation matrices (no text annotations)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(metrics['w_corr'], origin='lower', aspect='auto',
                          vmin=-1, vmax=1, cmap='RdBu_r')
    axes[0].set_title('Wiener-SVD (truth space)', fontsize=10)
    axes[0].set_xlabel(true_label); axes[0].set_ylabel(true_label)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(metrics['of_corr'], origin='lower', aspect='auto',
                          vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1].set_title('OmniFold (truth space)', fontsize=10)
    axes[1].set_xlabel(true_label); axes[1].set_ylabel(true_label)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle(f'Truth-Space Correlation: {true_label}', fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, axes


def plot_comparison_frac_unc(metrics, true_label, save_path=None):
    """Plot per-bin fractional uncertainty bar chart."""
    n = len(metrics['w_frac'])
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - 0.18, metrics['w_frac']*100, 0.35, label='Wiener-SVD',
           color='steelblue', alpha=0.8)
    ax.bar(x + 0.18, metrics['of_frac']*100, 0.35, label='OmniFold',
           color='crimson', alpha=0.8)
    ax.set_xlabel('True bin index'); ax.set_ylabel('Frac. unc. [%]')
    ax.set_title(f'Per-bin uncertainty: {true_label}', fontsize=10)
    ax.legend(fontsize=9); ax.set_xticks(x)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_eigenspectrum(metrics, var_name, save_path=None):
    """Plot eigenvalue spectrum comparison."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    w_e = metrics['w_eig']; of_e = metrics['of_eig']
    ax.semilogy(range(1, len(w_e)+1), w_e/w_e[0], 'o-',
                color='steelblue', ms=5, lw=1.5, label='Wiener-SVD')
    ax.semilogy(range(1, len(of_e)+1), of_e/of_e[0], 's-',
                color='crimson', ms=5, lw=1.5, label='OmniFold')
    ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Normalized eigenvalue')
    ax.set_title(f'Covariance eigenspectrum: {var_name}', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, ls='--')
    ax.axhline(0.01, color='gray', ls=':', lw=1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


# =============================================================================
# Plotting: heatmap, uncertainty budget
# =============================================================================

def plot_heatmap(matrix, title, vcfg, axis_labels=None,
                 fmt=".2e", cmap="viridis", save_path=None):
    n = vcfg.n_bins
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, origin='lower', cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=11)
    xlbl, ylbl = axis_labels if axis_labels else [vcfg.xlabel] * 2
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    fsize = max(6, 9 - n // 3)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if not np.isnan(v):
                tc = 'white' if abs(v) > 0.5 * np.nanmax(np.abs(matrix)) else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=tc, fontsize=fsize)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_uncertainty_budget(vcfg, cv_total, cov_dict, title="Uncertainty Budget",
                            save_path=None):
    bins = vcfg.bins
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(5, 4))
    total_cov = np.zeros((vcfg.n_bins, vcfg.n_bins))
    for i, (label, cov) in enumerate(cov_dict.items()):
        total_cov += cov
        unc = np.sqrt(np.diag(cov).clip(0))
        frac = np.divide(unc, cv_total, out=np.zeros_like(unc), where=cv_total > 0)
        if np.all(frac == 0): continue
        ax.step(bins, np.append(frac*100, frac[-1]*100), where="post",
                label=label, lw=1.5, color=colors[i % len(colors)])
    tot_unc = np.sqrt(np.diag(total_cov).clip(0))
    tot_frac = np.divide(tot_unc, cv_total, out=np.zeros_like(tot_unc), where=cv_total > 0)
    ax.step(bins, np.append(tot_frac*100, tot_frac[-1]*100), where="post",
            label="Total", color="black", ls="--", lw=2)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(vcfg.xlabel); ax.set_ylabel(r"Uncertainty [%]")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(tot_frac.max()*150, 5))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, ls='--')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


# =============================================================================
# NPZ archiving
# =============================================================================

def save_cov_npz(filename, **arrays):
    np.savez_compressed(filename, **arrays)
    print(f"Saved {len(arrays)} arrays → {filename}")