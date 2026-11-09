"""
nue_xsec_helpers.py  (v4)
-------------------------
Helpers for nue_xsec.ipynb (Wiener-SVD cross-section extraction).

Changes over v3:
  - Binning follows nue_syst.ipynb §0.2 (6 KE bins, 5 cosθ bins); old SYST_*_BINS removed
  - N_targets counts nucleons (ρ·V·N_A = 4.88e31, technote Eq. 31) — v3 divided by M_Ar
  - Covariances read from the per-family NPZs written by nue_syst.ipynb §2.11,
    with shape checks so stale files (e.g. the old 7-bin detsys file) are skipped
  - Response r_aμ = ε_μ Δ_aμ built from sel_topo with POT-scaled model (technote Eq. 32)
  - FDT weights are pandas Series aligned to sel_topo (v3 truncated the truth weights)
  - χ², norm/shape decomposition, unfolded-space covariance propagation
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as scipy_chi2


# =============================================================================
# Physical constants (technote §5.1)
# =============================================================================

# FV from _fiducial_cut_tmp in make_nueCC_df.py (same as nue_syst.ipynb §2.8)
V_FV_R1 = 2 * 185.0 * 380.0 * 240.0   # z in (10,250), y in (-190,190), full x
V_FV_R2 = 185.0 * 290.0 * 200.0       # z in (250,450), y in (-190,100), x < 0
V_FV_R3 = 185.0 * 380.0 * 200.0       # z in (250,450), y in (-190,190), x > 0
V_FV = V_FV_R1 + V_FV_R2 + V_FV_R3    # 58,534,000 cm³

RHO_LAR    = 1.38434        # g/cm³
N_AVOGADRO = 6.02214076e23
N_TARGETS  = RHO_LAR * V_FV * N_AVOGADRO   # nucleons: 4.88e31

# Energy-averaged νe+ν̄e flux over the FV, per POT [cm⁻² POT⁻¹].
# Cross-check against the technote: 1797/0.992 ≈ 1811 true signal at 8.81e19 POT and
# σ_GENIE = 4.49e-39 cm²  →  Φ = 1811 / (4.88e31 · 4.49e-39) / 8.81e19 ≈ 9.4e-11 / POT.
# (The "5.43e10 cm⁻²" quoted in technote §5.1 does not reproduce those numbers.)
FLUX_PER_POT = 9.65e-11

# Back-compat aliases
LYNN_RHO, LYNN_V_FV, LYNN_N_TARGETS, LYNN_FLUX = RHO_LAR, V_FV, N_TARGETS, FLUX_PER_POT


def get_constants():
    """(N_targets [nucleons], Φ [cm⁻² / POT])"""
    return N_TARGETS, FLUX_PER_POT


# =============================================================================
# Binning — must match nue_syst.ipynb §0.2 (checked against the saved NPZ CVs)
# =============================================================================

KE_MIN, KE_MAX = 500.0, 3000.0
ANALYSIS_BINS = {
    'reco_ke':       np.array([500, 750, 1050, 1350, 1800, 2250, 3000], dtype=float),
    'reco_costheta': np.array([-1.0, 0.5, 0.75, 0.85, 0.925, 1.0], dtype=float),
}
ANALYSIS_WIDTHS = {   # display widths (handscan style) — only used for drawing
    'reco_ke':       np.array([5, 6, 6, 9, 9, 15], dtype=float),
    'reco_costheta': np.array([6, 6, 4, 3, 2], dtype=float),
}
TRUE_VAR = {'reco_ke': 'true_ke', 'reco_costheta': 'true_costheta'}

LYNN_ENERGY_BINS_GEV = np.array([0.5, 0.7, 0.95, 1.25, 1.7, 2.5])
LYNN_COS_BINS        = np.array([-1.0, 0.6, 0.75, 0.85, 0.925, 1.0])

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
    6: r"Other $\nu$", 7: r"Cosmic", 8: r"Offbeam", 9: r"Dirt $\nu$", -1: r"Unknown",
}
CAT_COLORS = {
    0: "#56B4E9", 1: "#0072B2", 2: "green", 3: "#009E73",
    4: "#E69F00", 5: "#CC79A7", 6: "#D55E00", 7: "#F0E442",
    8: "#BDBDBD", 9: "#D2691E", -1: "#7F7F7F",
}


# =============================================================================
# VariableConfig
# =============================================================================

class VariableConfig:
    def __init__(self, name, bins, xlabel, xlabel_reco=None, xlabel_true=None,
                 unit='', save_name=None, disp_widths=None):
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
        w = np.ones(self.n_bins) if disp_widths is None else np.asarray(disp_widths, float)
        self.disp_edges  = np.concatenate([[0.0], np.cumsum(w)])   # drawing coordinates
        self.disp_centers = 0.5 * (self.disp_edges[:-1] + self.disp_edges[1:])

    def edge_labels(self):
        return [f'{int(e)}' if e == int(e) else f'{e:g}' for e in self.bins]

    def set_xaxis(self, ax, label=None, fontsize=9):
        """Ticks at display edges, labelled with the real bin edges."""
        ax.set_xticks(self.disp_edges)
        ax.set_xticklabels(self.edge_labels(), fontsize=fontsize)
        ax.set_xlim(self.disp_edges[0], self.disp_edges[-1])
        if label is not None:
            ax.set_xlabel(label)


# =============================================================================
# Covariance loading (NPZs from nue_syst.ipynb §2.11)
# =============================================================================

COV_BLOCKS = ('cov_ms_ms', 'cov_bs_bs', 'cov_ms_bs', 'cov_bs_ms')


def _as_matrix(v):
    v = np.asarray(v)
    if v.ndim == 0:
        v = v.item()
        if isinstance(v, dict) and 'cov' in v:
            v = v['cov']
    return np.asarray(v, dtype=float)


def load_family_covs(syst_dir, var_name, n_bins, exclude=('total',), verbose=True):
    """Read every saved_syst/<var>/<family>_cov_matrices.npz.

    Returns
    -------
    covs : {family: {block: (n,n)}}
    ref  : {'ms', 'bs', 'true_signal', 'response'} from the first file read (the CVs
           nue_syst used — compare these with the notebook CVs to catch binning drift)
    skipped : {family: reason}
    """
    vdir = os.path.join(syst_dir, var_name)
    if not os.path.isdir(vdir):
        raise FileNotFoundError(f"{vdir} not found — run nue_syst.ipynb first")
    covs, ref, skipped = {}, {}, {}
    for f in sorted(os.listdir(vdir)):
        if not f.endswith('_cov_matrices.npz'):
            continue
        fam = f.replace('_cov_matrices.npz', '')
        if fam in exclude:
            continue
        d = np.load(os.path.join(vdir, f), allow_pickle=True)
        blocks = {k: _as_matrix(d[k]) for k in COV_BLOCKS if k in d.files}
        shapes = {b.shape for b in blocks.values()}
        if not blocks or shapes != {(n_bins, n_bins)}:
            skipped[fam] = f"shape {sorted(shapes)} ≠ ({n_bins},{n_bins}) — stale file?"
            continue
        for k in COV_BLOCKS:
            blocks.setdefault(k, np.zeros((n_bins, n_bins)))
        covs[fam] = blocks
        if not ref:
            ref = {k: np.asarray(d[k], float) for k in ('ms', 'bs', 'true_signal', 'response')
                   if k in d.files}
    if verbose:
        print(f"[{var_name}] families: {list(covs)}")
        for fam, why in skipped.items():
            print(f"  ⚠ skipped {fam}: {why}")
    return covs, ref, skipped


def full_pred_cov(blocks):
    """Covariance of the total prediction S+B (signal, background and their correlation)."""
    return sum(blocks[k] for k in COV_BLOCKS)


# =============================================================================
# Response matrix  r_aμ = ε_μ Δ_aμ   (technote Eq. 32)
# =============================================================================

def _clip(x, bins, eps=1e-8):
    return np.clip(np.asarray(x, dtype=float), bins[0], bins[-1] - eps)


def build_response(sel_topo, final_stage, reco_col, true_col, bins, pot_scale,
                   sig_col='is_sig', truth_df=None):
    """Response / migration / efficiency from the selection DataFrame.

    truth_df : true-signal table (one row per true signal ν, incl. ν lost before
               preselection; nh.load_true_signal).  It is the efficiency
               denominator — without it only preselected signal is counted and
               the efficiency misses the flash-match / FV / reconstruction losses.
    Under/overflow in both reco and true go to the first/last bin (as in nue_syst).
    The model (true signal) and the selected-signal reco CV are POT-scaled with
    pot_scale, so  response @ model == sig_reco  exactly.
    """
    sig = sel_topo[sel_topo[sig_col].astype(bool)]
    t_all = (truth_df if truth_df is not None else sig)[true_col].dropna()
    truth_raw = np.histogram(_clip(t_all, bins), bins=bins)[0].astype(float)

    m = sig[final_stage].astype(bool) & sig[reco_col].notna() & sig[true_col].notna()
    sel = sig[m]
    n2d = np.histogram2d(_clip(sel[reco_col], bins), _clip(sel[true_col], bins),
                         bins=[bins, bins])[0]                     # (reco a, true μ)
    n_sel_true = n2d.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        eff       = np.where(truth_raw > 0, n_sel_true / truth_raw, 0.0)
        response  = np.where(truth_raw[None, :] > 0, n2d / truth_raw[None, :], 0.0)
        migration = np.where(n_sel_true[None, :] > 0, n2d / n_sel_true[None, :], 0.0)
    return dict(response=response, migration=migration, efficiency=eff,
                reco_vs_true_raw=n2d, truth_raw=truth_raw,
                model=truth_raw * pot_scale, sig_reco=n2d.sum(axis=1) * pot_scale,
                n_sig_true=int(len(t_all)), n_sig_sel=int(len(sel)))


def mc_reco_by_category(sel_topo, final_stage, reco_col, bins, pot_scale,
                        weights=None, sig_col='is_sig'):
    """Selected MC reco histograms: {'signal': …, truth_cat c (non-signal): …}.
    `weights` (Series aligned to sel_topo) multiplies the POT scale (fake data)."""
    m = sel_topo[final_stage].astype(bool) & sel_topo[reco_col].notna()
    sel = sel_topo[m]
    w = np.full(len(sel), pot_scale) if weights is None else weights[m].to_numpy(float) * pot_scale
    x = _clip(sel[reco_col], bins)
    is_sig = sel[sig_col].astype(bool).to_numpy()
    out = {'signal': np.histogram(x[is_sig], bins=bins, weights=w[is_sig])[0]}
    cats = sel['truth_cat'].to_numpy()
    for c in sorted(pd.unique(cats[~is_sig])):
        k = (~is_sig) & (cats == c)
        out[int(c)] = np.histogram(x[k], bins=bins, weights=w[k])[0]
    return out


# =============================================================================
# Statistics helpers
# =============================================================================

def safe_inv(cov):
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov, hermitian=True)


def chi2_cov(obs, exp, cov):
    """χ² = (o−e)ᵀ V⁻¹ (o−e), ndf = n_bins, p-value."""
    d = np.asarray(obs, float) - np.asarray(exp, float)
    c2 = float(d @ safe_inv(cov) @ d)
    ndf = len(d)
    return c2, ndf, float(scipy_chi2.sf(c2, ndf))


def frac_cov(cov, cv):
    cv = np.asarray(cv, float)
    den = np.outer(cv, cv)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den != 0, cov / den, 0.0)


def corr_from_cov(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    return cov / np.outer(d, d)


def norm_frac(cov, cv):
    """Normalization fractional uncertainty sqrt(ΣV)/ΣN (the % in the technote legends)."""
    s = np.sum(cv)
    return float(np.sqrt(max(np.sum(cov), 0.0)) / s) if s > 0 else 0.0


def matrix_decomp(model, cov):
    """MicroBooNE norm / shape / mixed decomposition of a covariance (w.r.t. `model`)."""
    N = np.asarray(model, float)
    NT, Msum = N.sum(), cov.sum()
    if NT == 0:
        z = np.zeros_like(cov)
        return z, cov.copy(), z
    row, col = cov.sum(axis=1), cov.sum(axis=0)
    norm  = np.outer(N, N) * Msum / NT**2
    shape = cov - (np.outer(row, N) + np.outer(N, col)) / NT + norm
    mixed = cov - norm - shape
    return norm, shape, mixed


# =============================================================================
# Wiener-SVD wrapper
# =============================================================================

def unfold(WienerSVD_fn, response, model, measured, cov, C_type=2, Norm_type=0):
    """Run Wiener-SVD and attach the A_C-smeared model and its χ² (technote Eq. 55)."""
    uf = WienerSVD_fn(response, model, measured, cov, C_type, Norm_type)
    uf['model_smeared'] = uf['AddSmear'] @ model
    uf['chi2'], uf['ndf'], uf['p'] = chi2_cov(uf['unfold'], uf['model_smeared'], uf['UnfoldCov'])
    return uf


def propagate_cov(uf, cov_reco):
    """Push a reco-space covariance through the same linear unfolding (Eq. 53–54)."""
    Rot = uf['CovRotation']
    return Rot @ cov_reco @ Rot.T


def xsec_denominator(bin_widths, pot, n_targets=N_TARGETS, flux_per_pot=FLUX_PER_POT):
    """N_targets · Φ · Δx  (Eq. 30). Pass bin widths in the units of the dσ/dx axis."""
    return n_targets * flux_per_pot * pot * np.asarray(bin_widths, float)


# =============================================================================
# Fake-data tests
# =============================================================================

def apply_tilt_weights(values, alpha, lo, hi):
    """Linear tilt: w = 1 + α·(x − x_mid)/x_range (same as OmniFold MakePlots.py FDS)."""
    x_mean = 0.5 * (lo + hi)
    x_range = hi - lo
    if x_range <= 0:
        return np.ones_like(values)
    return np.clip(1.0 + alpha * (values - x_mean) / x_range, 0.0, 10.0)


def fdt_weights(sel_topo, pi0_scale=1.0, tilt=None, low_e=None, sig_col='is_sig'):
    """Per-event weights (Series aligned to sel_topo) used for BOTH the fake data and
    the alternative truth, so the two always stay consistent.

    pi0_scale : scale non-signal νμ CC π⁰ / NC π⁰ (truth_cat 2, 3) — background-dominant
    tilt      : (true_col, alpha, lo, hi) linear tilt of the signal — signal-dominant
    low_e     : (true_col, threshold, scale) scale signal below threshold
    """
    w = pd.Series(1.0, index=sel_topo.index)
    sig = sel_topo[sig_col].astype(bool)
    if pi0_scale != 1.0:
        m = (~sig) & sel_topo['truth_cat'].isin([2, 3])
        w.loc[m] = w.loc[m] * pi0_scale
    if tilt is not None:
        col, alpha, lo, hi = tilt
        v = sel_topo.loc[sig, col].clip(lo, hi).to_numpy(float)
        w.loc[sig] = w.loc[sig].to_numpy() * np.nan_to_num(apply_tilt_weights(v, alpha, lo, hi), nan=1.0)
    if low_e is not None:
        col, thr, scale = low_e
        m = sig & (sel_topo[col] < thr)
        w.loc[m] = w.loc[m] * scale
    return w


def run_fdt(sel_topo, final_stage, reco_col, true_col, bins, pot_scale, weights,
            resp, bkg_cv, extra_bkg, V_syst, WienerSVD_fn, C_type=2, Norm_type=0,
            sig_col='is_sig', truth_df=None, truth_w=None):
    """One fake-data test (technote §5.3.2, without the CCBC).

    fake data  D = Σ w·pot_scale over selected MC + unchanged dirt/off-beam
    measured   d = D − B_CV                    (nominal background subtracted)
    cov        V_syst + diag(D)                (syst + fake-data stat)
    """
    by_cat = mc_reco_by_category(sel_topo, final_stage, reco_col, bins, pot_scale,
                                 weights=weights, sig_col=sig_col)
    D_fake = sum(by_cat.values()) + extra_bkg
    n_cv = resp['sig_reco'] + bkg_cv
    if truth_df is not None:          # alternative truth over ALL true signal (same as resp['model'])
        t = truth_df[true_col].to_numpy(float)
        ws = (np.ones(len(t)) if truth_w is None else np.asarray(truth_w, dtype=float))
    else:
        sig_mask = sel_topo[sig_col].astype(bool)
        t = sel_topo.loc[sig_mask, true_col].to_numpy(float)
        ws = weights[sig_mask].to_numpy(float)
    ok = np.isfinite(t)
    alt_truth = np.histogram(_clip(t[ok], bins), bins=bins, weights=ws[ok] * pot_scale)[0]

    c2_pre = chi2_cov(D_fake, n_cv, V_syst + np.diag(np.clip(n_cv, 1e-9, None)))
    cov = V_syst + np.diag(np.clip(D_fake, 1e-9, None))
    uf = unfold(WienerSVD_fn, resp['response'], resp['model'], D_fake - bkg_cv, cov,
                C_type, Norm_type)
    ms_alt = uf['AddSmear'] @ alt_truth
    c2_alt = chi2_cov(uf['unfold'], ms_alt, uf['UnfoldCov'])
    return dict(unfold=uf, D_fake=D_fake, n_cv=n_cv, by_cat=by_cat, alt_truth=alt_truth,
                ms_nom=uf['model_smeared'], ms_alt=ms_alt,
                c2_rate=c2_pre[0], p_rate=c2_pre[2],
                c2_nom=uf['chi2'], p_nom=uf['p'], c2_mod=c2_alt[0], p_mod=c2_alt[2],
                ndf=uf['ndf'])


# =============================================================================
# CCBC (technote §4.2) — needs signal-region × sideband covariances (not produced yet)
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
    V_nC_inv = safe_inv(V_nC_tilde)
    delta_C = D_C - n_C_cv
    B_constr = B_S_cv + V_BS_nC @ V_nC_inv @ delta_C
    V_BS_constr = V_BS_BS - V_BS_nC @ V_nC_inv @ V_BS_nC.T
    V_phiS_BS_constr = V_phiS_BS - V_phiS_nC @ V_nC_inv @ V_BS_nC.T
    m_S = phi_S_cv + B_constr
    V_mS_mS = V_phiS_phiS + V_phiS_BS_constr + V_phiS_BS_constr.T + V_BS_constr
    return dict(B_constr=B_constr, V_BS_constr=V_BS_constr,
                V_phiS_BS_constr=V_phiS_BS_constr, m_S=m_S, V_mS_mS=V_mS_mS)


# =============================================================================
# Plotting
# =============================================================================

def plot_matrix(ax, M, vcfg, ylabel_cfg=None, cmap='viridis', vmin=None, vmax=None,
                fmt='{:.2f}', title='', cbar_label='', xlabel=None, ylabel=None,
                annotate=True, fontsize=8):
    """Bin-index heatmap with real bin edges as tick labels (x = columns, y = rows)."""
    ycfg = ylabel_cfg or vcfg
    n_r, n_c = M.shape
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   extent=[0, n_c, 0, n_r])
    cb = plt.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    ax.set_xticks(np.arange(n_c + 1)); ax.set_xticklabels(vcfg.edge_labels(), fontsize=8)
    ax.set_yticks(np.arange(n_r + 1)); ax.set_yticklabels(ycfg.edge_labels(), fontsize=8)
    ax.set_xlabel(xlabel or vcfg.xlabel); ax.set_ylabel(ylabel or ycfg.xlabel)
    ax.set_title(title, fontsize=11)
    if annotate:
        cm = plt.get_cmap(cmap)
        for i in range(n_r):
            for j in range(n_c):
                v = M[i, j]
                if np.isfinite(v):
                    r_, g_, b_, _ = cm(im.norm(v))
                    ax.text(j + 0.5, i + 0.5, fmt.format(v), ha='center', va='center',
                            fontsize=fontsize,
                            color='black' if 0.299*r_ + 0.587*g_ + 0.114*b_ > 0.5 else 'white')
    return im


def plot_designed_matrix(M, vcfg, ylabel, title, vmin=0, vmax=1, cmap='Blues', fmt='{:.2f}',
                         cbar_label='Probability', figsize=(7, 6)):
    """Same format as the old nue_syst.ipynb §6.5 response plot: rows = true bins, columns = reco
    bins, designed display widths on both axes, real bin edges as tick labels."""
    fig, ax = plt.subplots(figsize=figsize)
    de, c = vcfg.disp_edges, vcfg.disp_centers
    pm = ax.pcolormesh(de, de, M, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pm, ax=ax, label=cbar_label)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(c[j], c[i], fmt.format(M[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.5 * vmax else 'black')
    ax.set_xlabel(vcfg.xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    vcfg.set_xaxis(ax)
    ax.set_yticks(de)
    ax.set_yticklabels(vcfg.edge_labels(), fontsize=9)
    ax.set_ylim(de[0], de[-1])
    fig.tight_layout()
    return fig, ax


def step(ax, edges, y, **kw):
    y = np.asarray(y, float)
    return ax.step(edges, np.append(y, y[-1]), where='post', **kw)


# =============================================================================
# Wiener-SVD vs OmniFold comparison
# =============================================================================

def compute_comparison_metrics(w_cov, of_cov, w_mean, of_mean):
    """Compare two unfolded covariances (diagnostics only, no plotting)."""
    n_w, n_o = w_cov.shape[0], of_cov.shape[0]

    w_dg = np.sqrt(np.diag(w_cov).clip(1e-10))
    of_dg = np.sqrt(np.diag(of_cov).clip(1e-10))
    w_corr = w_cov / np.outer(w_dg, w_dg)
    of_corr = of_cov / np.outer(of_dg, of_dg)

    w_eig = np.linalg.eigvalsh(w_cov)
    of_eig = np.linalg.eigvalsh(of_cov)

    w_offdiag = w_corr[np.triu_indices(n_w, k=1)]
    of_offdiag = of_corr[np.triu_indices(n_o, k=1)]

    n = min(n_w, n_o)
    w_frac = w_dg[:n] / np.abs(w_mean[:n]).clip(1)
    of_frac = of_dg[:n] / np.abs(of_mean[:n]).clip(1)

    _, w_shape_cov, _ = matrix_decomp(w_mean[:n], w_cov[:n, :n])
    _, of_shape_cov, _ = matrix_decomp(of_mean[:n], of_cov[:n, :n])
    w_shape = np.sqrt(np.diag(w_shape_cov).clip(0)) / np.abs(w_mean[:n]).clip(1e-6) * 100
    of_shape = np.sqrt(np.diag(of_shape_cov).clip(0)) / np.abs(of_mean[:n]).clip(1e-6) * 100

    return dict(
        w_cond=np.linalg.cond(w_cov), of_cond=np.linalg.cond(of_cov),
        w_edof=w_eig.sum() / w_eig.max(), of_edof=of_eig.sum() / of_eig.max(),
        n_w=n_w, n_o=n_o,
        w_mean_offdiag=np.abs(w_offdiag).mean(), of_mean_offdiag=np.abs(of_offdiag).mean(),
        w_frac_neg=(w_offdiag < 0).mean(), of_frac_neg=(of_offdiag < 0).mean(),
        w_total_frac=norm_frac(w_cov, w_mean) * 100,
        of_total_frac=norm_frac(of_cov, of_mean) * 100,
        w_frac=w_frac, of_frac=of_frac,
        w_shape=w_shape, of_shape=of_shape,
        w_corr=w_corr, of_corr=of_corr,
        w_eig=np.sort(w_eig)[::-1], of_eig=np.sort(of_eig)[::-1],
    )


def print_comparison_metrics(metrics, var_name):
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
    print(f"  {'Norm. frac unc [%]':<26} {m['w_total_frac']:>12.1f} {m['of_total_frac']:>12.1f}")
    print(f"  {'─'*50}")
    print(f"  Per-bin frac unc [%]:")
    print(f"    Wiener-SVD: {np.round(m['w_frac']*100, 1)}")
    print(f"    OmniFold:   {np.round(m['of_frac']*100, 1)}")
    print(f"  Shape-only unc [%]:")
    print(f"    Wiener-SVD: {np.round(m['w_shape'], 1)}")
    print(f"    OmniFold:   {np.round(m['of_shape'], 1)}")


def plot_comparison_frac_unc(metrics, true_label, save_path=None):
    n = len(metrics['w_frac'])
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - 0.18, metrics['w_frac']*100, 0.35, label='Wiener-SVD', color='steelblue', alpha=0.8)
    ax.bar(x + 0.18, metrics['of_frac']*100, 0.35, label='OmniFold', color='crimson', alpha=0.8)
    ax.set_xlabel('True bin index'); ax.set_ylabel('Frac. unc. [%]')
    ax.set_title(f'Per-bin uncertainty: {true_label}', fontsize=10)
    ax.legend(fontsize=9); ax.set_xticks(x)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def plot_eigenspectrum(metrics, var_name, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    w_e = metrics['w_eig']; of_e = metrics['of_eig']
    ax.semilogy(range(1, len(w_e)+1), w_e/w_e[0], 'o-', color='steelblue', ms=5, lw=1.5, label='Wiener-SVD')
    ax.semilogy(range(1, len(of_e)+1), of_e/of_e[0], 's-', color='crimson', ms=5, lw=1.5, label='OmniFold')
    ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Normalized eigenvalue')
    ax.set_title(f'Covariance eigenspectrum: {var_name}', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, ls='--')
    ax.axhline(0.01, color='gray', ls=':', lw=1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


# =============================================================================
# NPZ archiving
# =============================================================================

def save_cov_npz(filename, **arrays):
    np.savez_compressed(filename, **arrays)
    print(f"Saved {len(arrays)} arrays → {filename}")
