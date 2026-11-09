"""
nue_xsec_helpers.py  (v5)
-------------------------
Helpers for nue_xsec.ipynb (Wiener-SVD cross-section extraction).

Changes over v4:
  - Serialized 2D binning (SerialBinning, PC2D): p in cosθ slices, one unique bin index per
    (slice, p) cell, so the 2D measurement is unfolded as one 1D vector and bin migration
    between slices is kept (MicroBooNE CC1p0π style). Columns reco_pc2d / true_pc2d hold the
    index (+0.5), so the 1D machinery (response, universes, Wiener-SVD) is reused unchanged.
  - Slice-aware plotting: serialized axes with slice separators, per-slice panels, per-slice
    covariance blocks and χ² (slice_chi2), sub-blocks of the mega matrices
  - Removed the unused KE-based NueCCBinning2D, LYNN_* constants and the unused OmniFold plots

v4: binning follows nue_syst.ipynb §0.2; N_targets in nucleons; per-family NPZ covariances;
    r_aμ = ε_μ Δ_aμ (technote Eq. 32); FDT weights aligned to sel_topo; χ² / norm-shape tools.
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


def get_constants():
    """(N_targets [nucleons], Φ [cm⁻² / POT])"""
    return N_TARGETS, FLUX_PER_POT


# =============================================================================
# Binning — must match nue_syst.ipynb §0.2 (checked against the saved NPZ CVs)
# =============================================================================

KE_MIN, KE_MAX = 500.0, 3000.0
ANALYSIS_BINS = {
    'reco_ke':       np.array([500, 700, 950, 1250, 1600, 2250, 3000], dtype=float),  
    'reco_costheta': np.array([-1.0, 0.5, 0.75, 0.85, 0.925, 1.0], dtype=float),
}
ANALYSIS_WIDTHS = {
    'reco_ke':       np.array([4, 5, 6, 7, 13, 15], dtype=float),
    'reco_costheta': np.array([6, 6, 4, 3, 2], dtype=float),
}
TRUE_VAR = {'reco_ke': 'true_ke', 'reco_costheta': 'true_costheta', 'reco_pc2d': 'true_pc2d'}


# =============================================================================
# Serialized 2D binning: p in cosθ slices, one unique bin index per (slice, p) cell
# =============================================================================

class SerialBinning:
    """Unique (serialized) bin index for a 2D measurement.

    The 2D phase space is cut into slices of `slice_var`; slice i has its own edges x_edges[i]
    in `x_var`. Global bin index = offset of the slice + local x bin, so the true → reco
    response, the universes and the unfolding all act on ONE vector and bin migration between
    slices is kept. Under/overflow go to the first/last slice and to the first/last x bin of
    that slice (same convention as the 1D binning).
    """

    def __init__(self, slice_edges, x_edges, slice_var='costheta', x_var='p',
                 slice_tex=r'$\cos\theta_e$', x_tex=r'$p_e$', x_unit='GeV/c', x_scale=1e-3):
        self.slice_edges = np.asarray(slice_edges, dtype=float)
        self.x_edges = [np.asarray(e, dtype=float) for e in x_edges]
        assert len(self.x_edges) == len(self.slice_edges) - 1, 'one x binning per slice'
        self.slice_var, self.x_var = slice_var, x_var
        self.slice_tex, self.x_tex = slice_tex, x_tex
        self.x_unit, self.x_scale = x_unit, x_scale       # x_scale: stored unit → displayed unit
        self.n_slices = len(self.x_edges)
        self.n_x = np.array([len(e) - 1 for e in self.x_edges])
        self.offsets = np.concatenate([[0], np.cumsum(self.n_x)])
        self.n_bins = int(self.offsets[-1])
        self.edges = np.arange(self.n_bins + 1, dtype=float)          # serialized axis
        self.slice_of = np.repeat(np.arange(self.n_slices), self.n_x)
        self.local_of = np.concatenate([np.arange(n) for n in self.n_x])

    def block(self, i):
        """Global index range of slice i (use as M[b, b])."""
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def index(self, s, x):
        """Global bin index per event; -1 where s or x is NaN. float64 cast first: float32
        values clipped at edge−1e-9 round back onto the edge."""
        s = np.asarray(s, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        out = np.full(s.shape, -1, dtype=int)
        ok = np.isfinite(s) & np.isfinite(x)
        se = self.slice_edges
        i = np.clip(np.searchsorted(se, s[ok], side='right') - 1, 0, self.n_slices - 1)
        j = np.empty_like(i)
        for k in range(self.n_slices):
            m = i == k
            e = self.x_edges[k]
            j[m] = np.clip(np.searchsorted(e, x[ok][m], side='right') - 1, 0, len(e) - 2)
        out[ok] = self.offsets[i] + j
        return out

    def value(self, s, x):
        """Column value for the 1D machinery: index + 0.5 (NaN where undefined)."""
        idx = self.index(s, x)
        return np.where(idx >= 0, idx + 0.5, np.nan)

    def widths(self, x_scale=None):
        """Δslice · Δx per global bin (x in displayed units unless x_scale is given)."""
        xs = self.x_scale if x_scale is None else x_scale
        ds = np.diff(self.slice_edges)
        return np.concatenate([ds[k] * np.diff(e) * xs for k, e in enumerate(self.x_edges)])

    def slice_label(self, i, tex=True):
        lo, hi = self.slice_edges[i], self.slice_edges[i + 1]
        name = self.slice_tex if tex else self.slice_var
        return f'{lo:g} ≤ {name} < {hi:g}'

    def x_disp_edges(self, i):
        return self.x_edges[i] * self.x_scale

    @property
    def x_label(self):
        return f'{self.x_tex} [{self.x_unit}]' if self.x_unit else self.x_tex

    def bin_table(self):
        """One row per global bin: slice range and x range (stored units)."""
        lo_s = self.slice_edges[:-1][self.slice_of]
        hi_s = self.slice_edges[1:][self.slice_of]
        lo_x = np.concatenate([e[:-1] for e in self.x_edges])
        hi_x = np.concatenate([e[1:] for e in self.x_edges])
        return pd.DataFrame({'bin': np.arange(self.n_bins), 'slice': self.slice_of,
                             f'{self.slice_var}_lo': lo_s, f'{self.slice_var}_hi': hi_s,
                             f'{self.x_var}_lo': lo_x, f'{self.x_var}_hi': hi_x})


def design_slice_edges(slice_vals, x_vals, weights, slice_edges, x_candidates, min_count):
    """Per-slice x edges: start from the candidate edges and merge from the high-x end until
    every bin holds ≥ min_count (weighted) events; a low-x remainder below min_count is merged
    into its neighbour. Overflow above the last edge counts in the last bin.
    Returns (list of edge arrays, list of per-bin counts)."""
    s = np.asarray(slice_vals, np.float64)
    x = np.clip(np.asarray(x_vals, np.float64), x_candidates[0], x_candidates[-1] - 1e-6)
    w = np.broadcast_to(np.asarray(weights, np.float64), s.shape)
    out_e, out_n = [], []
    for k in range(len(slice_edges) - 1):
        lo, hi = slice_edges[k], slice_edges[k + 1]
        m = (s >= lo) & ((s < hi) if k < len(slice_edges) - 2 else (s <= hi))
        n = np.histogram(x[m], bins=x_candidates, weights=w[m])[0]
        edges, counts, acc = [x_candidates[-1]], [], 0.0
        for b in range(len(n) - 1, -1, -1):           # high x → low x
            acc += n[b]
            if acc >= min_count:
                edges.append(x_candidates[b]); counts.append(acc); acc = 0.0
        if acc > 0 or edges[-1] != x_candidates[0]:   # remainder at low x
            if counts:
                edges[-1] = x_candidates[0]; counts[-1] += acc
            else:
                edges.append(x_candidates[0]); counts.append(acc)
        out_e.append(np.array(edges[::-1])); out_n.append(np.array(counts[::-1]))
    return out_e, out_n


# Frozen 2D binning. cosθ slices = 1D analysis bins; p candidates = 1D KE edges in MeV/c
# (p = KE + 0.511 MeV, negligible). Derived in nue_xsec.ipynb §1.1 with design_slice_edges
# (≥ PC2D_MIN_SIG selected signal per bin at 6.6e20 POT); §1.1 warns if the current selection
# would give different edges. nue_syst.ipynb uses this to build the 2D covariances.
PC2D_MIN_SIG = 50.0
PC2D_P_CANDIDATES = ANALYSIS_BINS['reco_ke'].copy()
PC2D = SerialBinning(
    ANALYSIS_BINS['reco_costheta'],
    [[500, 750, 3000],
     [500, 750, 1050, 3000],
     [500, 750, 1050, 1350, 3000],
     [500, 750, 1050, 1350, 1800, 3000],
     [500, 750, 1050, 1350, 1800, 2250, 3000]])


def add_serial_columns(df, binning=PC2D, name='pc2d',
                       reco=('reco_costheta', 'reco_p'), true=('true_costheta', 'true_p')):
    """Add reco_<name> / true_<name> (serialized index + 0.5) to a DataFrame, in place.
    Pairs whose columns are missing (e.g. no truth in data) are skipped."""
    for kind, (sc, xc) in (('reco', reco), ('true', true)):
        if sc in df.columns and xc in df.columns:
            df[f'{kind}_{name}'] = binning.value(df[sc].to_numpy(), df[xc].to_numpy())
    return df


def add_serial_to_reco(reco, binning=PC2D, name='pc2d', s_key='reco_costheta', x_key='reco_p'):
    """Same for the {var: array} dicts of data / off-beam / in-time (arrays must be row-aligned)."""
    s, x = np.asarray(reco.get(s_key, [])), np.asarray(reco.get(x_key, []))
    if len(s) != len(x):
        raise ValueError(f'{s_key} ({len(s)}) and {x_key} ({len(x)}) are not row-aligned')
    reco[f'reco_{name}'] = binning.value(s, x)
    return reco


# =============================================================================
# VariableConfig
# =============================================================================

class VariableConfig:
    """serial : SerialBinning → `bins` is the serialized index axis 0..n_bins and every axis
    drawn with set_xaxis / plot_matrix gets slice separators and slice labels."""
    def __init__(self, name, bins, xlabel, xlabel_reco=None, xlabel_true=None,
                 unit='', save_name=None, disp_widths=None, serial=None):
        self.name        = name
        self.serial      = serial
        if serial is not None:
            bins = serial.edges
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

    def set_xaxis(self, ax, label=None, fontsize=9, slice_labels=True):
        """Ticks at display edges, labelled with the real bin edges.
        Serialized variable: one tick per bin (bin index), slice separators (+ slice labels;
        slice_labels=False for a ratio panel under a main panel that already has them)."""
        if self.serial is not None:
            ax.set_xticks(self.disp_centers)
            ax.set_xticklabels([str(i) for i in range(self.n_bins)], fontsize=min(fontsize, 7))
            ax.set_xlim(self.disp_edges[0], self.disp_edges[-1])
            draw_slice_separators(ax, self.serial, self.disp_edges, axis='x', labels=slice_labels)
        else:
            ax.set_xticks(self.disp_edges)
            ax.set_xticklabels(self.edge_labels(), fontsize=fontsize)
            ax.set_xlim(self.disp_edges[0], self.disp_edges[-1])
        if label is not None:
            ax.set_xlabel(label)


def draw_slice_separators(ax, binning, disp_edges=None, axis='x', labels=True, color='gray',
                          inside=True):
    """Dashed lines between slices of a serialized axis; slice ranges written along the top
    (axis='x': inside the axes, or just above it with inside=False — matrices) or right edge
    (axis='y'). Labels of the first call on an Axes only."""
    e = binning.edges if disp_edges is None else np.asarray(disp_edges)
    cuts = e[binning.offsets]
    line = ax.axvline if axis == 'x' else ax.axhline
    for c in cuts[1:-1]:
        line(c, color=color, ls='--', lw=0.9, zorder=6)
    if not labels or getattr(ax, '_slice_labels_drawn', False):
        return
    ax._slice_labels_drawn = True
    import matplotlib.transforms as mtrans
    for i in range(binning.n_slices):
        mid = 0.5 * (cuts[i] + cuts[i + 1])
        lo, hi = binning.slice_edges[i], binning.slice_edges[i + 1]
        txt = f'[{lo:g}, {hi:g})'
        if axis == 'x':
            tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(mid, 0.985 if inside else 1.005, txt, transform=tr, ha='center',
                    va='top' if inside else 'bottom', fontsize=7, color='dimgray')
        else:
            tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.005, mid, txt, transform=tr, ha='left', va='center', fontsize=6,
                    color='dimgray', rotation=90)


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


def slice_chi2(obs, exp, cov, binning):
    """χ² per slice with that slice's covariance block only (correlations to the other slices
    omitted, as in the per-slice MicroBooNE panels) and over all bins with the full matrix.
    Returns {slice i: (χ², ndf, p), 'all': (χ², ndf, p)}."""
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    out = {i: chi2_cov(obs[binning.block(i)], exp[binning.block(i)],
                       cov[binning.block(i), binning.block(i)]) for i in range(binning.n_slices)}
    out['all'] = chi2_cov(obs, exp, cov)
    return out


def frac_cov(cov, cv):
    cv = np.asarray(cv, float)
    den = np.outer(cv, cv)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den != 0, cov / den, 0.0)


def corr_from_cov(cov):
    """Correlation matrix; scale-free (xsec covariances are ~1e-78 cm⁴, so no absolute floor)."""
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    den = np.outer(d, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, cov / np.where(den > 0, den, 1.0), 0.0)


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

def block_c_matrix(binning, C_type):
    """Smoothness matrix for a serialized vector: Matrix_C(n_slice, C_type) on each slice block,
    zero between slices, so the regularization never couples the last p bin of one cosθ slice
    to the first p bin of the next."""
    from analysis_village.unfolding.wienersvd import Matrix_C
    C = np.zeros((binning.n_bins, binning.n_bins))
    for i in range(binning.n_slices):
        b = binning.block(i)
        C[b, b] = Matrix_C(int(binning.n_x[i]), C_type)
    return C


def wiener_svd_c(Response, Signal, Measure, Covariance, C, Norm_type):
    """analysis_village.unfolding.wienersvd.WienerSVD with an explicit smoothness matrix C
    (same algorithm and outputs; WienerSVD itself only takes an integer C_type)."""
    m, n = Response.shape
    _, s_cov, Vh_cov = np.linalg.svd(Covariance)
    Q = np.diag(np.where(s_cov != 0, 1 / np.sqrt(np.where(s_cov != 0, s_cov, 1)), 0)) @ Vh_cov
    M_trans, R = Q @ Measure, Q @ Response
    C = np.asarray(C, float) @ np.diag(1.0 / np.asarray(Signal, float) ** Norm_type)
    C_inv = np.linalg.inv(C)
    Signal_mod = C @ Signal
    U, D, Vh = np.linalg.svd(R @ C_inv, full_matrices=False)
    D_t = np.zeros((n, m)); D_t[np.arange(min(n, m)), np.arange(min(n, m))] = D
    S_vec = Vh @ Signal_mod
    W = np.diag(S_vec ** 2 / (D ** 2 * S_vec ** 2 + 1))
    WF = D ** 2 * np.diag(W)
    rot = C_inv @ Vh.T @ W @ D_t @ U.T @ Q
    return {'unfold': rot @ Measure, 'AddSmear': C_inv @ Vh.T @ np.diag(WF) @ Vh @ C, 'WF': WF,
            'UnfoldCov': rot @ Covariance @ rot.T, 'CovRotation': rot}


def unfold(WienerSVD_fn, response, model, measured, cov, C_type=2, Norm_type=0, serial=None):
    """Run Wiener-SVD and attach the A_C-smeared model and its χ² (technote Eq. 55).
    serial : SerialBinning → block-diagonal smoothness matrix (regularize within slices only)."""
    if serial is None:
        uf = WienerSVD_fn(response, model, measured, cov, C_type, Norm_type)
    else:
        uf = wiener_svd_c(response, model, measured, cov, block_c_matrix(serial, C_type), Norm_type)
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
            sig_col='is_sig', truth_df=None, truth_w=None, serial=None):
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
                C_type, Norm_type, serial=serial)
    ms_alt = uf['AddSmear'] @ alt_truth
    c2_alt = chi2_cov(uf['unfold'], ms_alt, uf['UnfoldCov'])
    out = dict(unfold=uf, D_fake=D_fake, n_cv=n_cv, by_cat=by_cat, alt_truth=alt_truth,
               ms_nom=uf['model_smeared'], ms_alt=ms_alt,
               c2_rate=c2_pre[0], p_rate=c2_pre[2],
               c2_nom=uf['chi2'], p_nom=uf['p'], c2_mod=c2_alt[0], p_mod=c2_alt[2],
               ndf=uf['ndf'])
    if serial is not None:            # per-slice χ² (slice covariance block only)
        out['slice_nom'] = slice_chi2(uf['unfold'], uf['model_smeared'], uf['UnfoldCov'], serial)
        out['slice_mod'] = slice_chi2(uf['unfold'], ms_alt, uf['UnfoldCov'], serial)
    return out


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

def _matrix_axis(ax, cfg, n, axis):
    """Ticks of one matrix axis: real bin edges (1D) or bin index + slice separators (serial)."""
    set_t, set_l = (ax.set_xticks, ax.set_xticklabels) if axis == 'x' else (ax.set_yticks, ax.set_yticklabels)
    if cfg.serial is not None:
        set_t(np.arange(n) + 0.5); set_l([str(i) for i in range(n)], fontsize=6)
        draw_slice_separators(ax, cfg.serial, axis=axis, color='white', labels=axis == 'x', inside=False)
    else:
        set_t(np.arange(n + 1)); set_l(cfg.edge_labels(), fontsize=8)


def plot_matrix(ax, M, vcfg, ylabel_cfg=None, cmap='viridis', vmin=None, vmax=None,
                fmt='{:.2f}', title='', cbar_label='', xlabel=None, ylabel=None,
                annotate=None, fontsize=8):
    """Bin-index heatmap with real bin edges as tick labels (x = columns, y = rows).
    Serialized configs get bin-index ticks and slice separators; annotate=None → only if ≤ 12 bins."""
    ycfg = ylabel_cfg or vcfg
    n_r, n_c = M.shape
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   extent=[0, n_c, 0, n_r])
    cb = plt.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    _matrix_axis(ax, vcfg, n_c, 'x')
    _matrix_axis(ax, ycfg, n_r, 'y')
    ax.set_xlabel(xlabel or vcfg.xlabel); ax.set_ylabel(ylabel or ycfg.xlabel)
    ax.set_title(title, fontsize=11, pad=16 if vcfg.serial is not None else 6)   # room for slice labels
    if annotate is None:
        annotate = max(n_r, n_c) <= 12
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
    if vcfg.serial is not None:                      # 20×20: bin-index axes, no numbers
        fig, ax = plt.subplots(figsize=(figsize[0] * 1.25, figsize[1] * 1.2))
        plot_matrix(ax, M, vcfg, cmap=cmap, vmin=vmin, vmax=vmax, title=title,
                    cbar_label=cbar_label, ylabel=ylabel, annotate=False)
        fig.tight_layout()
        return fig, ax
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
# Serialized 2D: per-slice views (x = the physical variable) and sub-blocks
# =============================================================================

def chi2_text(c):
    c2, ndf, p = c
    return rf'$\chi^2$/dof = {c2:.1f}/{ndf}, $p$ = {p:.2f}'


def slice_grid(binning, ncols=3, panel=(5.0, 3.8), ratio=False):
    """Figure with one panel per slice (+ an optional ratio panel below each)."""
    nrows = int(np.ceil(binning.n_slices / ncols))
    fig = plt.figure(figsize=(panel[0] * ncols, panel[1] * nrows * (1.3 if ratio else 1.0)))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.28)
    axes = []
    for i in range(binning.n_slices):
        r, c = divmod(i, ncols)
        if ratio:
            inner = gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(inner[0]); ar = fig.add_subplot(inner[1], sharex=ax)
            plt.setp(ax.get_xticklabels(), visible=False)
            axes.append((ax, ar))
        else:
            axes.append(fig.add_subplot(gs[r, c]))
    return fig, axes


def plot_slices(binning, steps=(), points=None, bands=(), chi2=None, ylabel='', title='',
                ncols=3, ratio_to=None, legend_slice=0):
    """Fig. 16-style panels: slice i drawn against its own x edges (displayed units).

    steps  : [(values on the serialized bins, step kwargs incl. label), …]
    points : (values, yerr_inner, yerr_outer or None, kwargs) — e.g. stat. and stat.⊕syst.
    bands  : [(values, half-width, bar kwargs incl. label), …] — e.g. norm. syst. at 0
    chi2   : slice_chi2() output → written in each panel and in the suptitle ('all')
    ratio_to : values to divide by in a ratio panel under each slice (None → no ratio panel)
    """
    fig, axes = slice_grid(binning, ncols=ncols, ratio=ratio_to is not None)
    for i in range(binning.n_slices):
        ax, ar = axes[i] if ratio_to is not None else (axes[i], None)
        b, e = binning.block(i), binning.x_disp_edges(i)
        ctr = 0.5 * (e[1:] + e[:-1])
        for vals, hw, kw in bands:
            ax.bar(e[:-1], 2 * np.asarray(hw)[b], bottom=np.asarray(vals)[b] - np.asarray(hw)[b],
                   width=np.diff(e), align='edge', **kw)
        for vals, kw in steps:
            step(ax, e, np.asarray(vals)[b], **kw)
        top = max([np.max(np.asarray(v)[b]) for v, _ in steps] + [0])
        if points is not None:
            v, inner, outer, kw = points
            v = np.asarray(v)[b]
            if outer is not None:
                ax.errorbar(ctr, v, yerr=np.asarray(outer)[b], fmt='none', ecolor=kw.get('color', 'k'),
                            capsize=3, lw=1)
                top = max(top, np.max(v + np.asarray(outer)[b]))
            ax.errorbar(ctr, v, yerr=np.asarray(inner)[b], **{'fmt': 'ko', 'ms': 4, 'capsize': 0,
                                                              'lw': 2, **kw})
            top = max(top, np.max(v + np.asarray(inner)[b]))
        ax.set_xlim(e[0], e[-1]); ax.set_ylim(0, top * 1.45 if top > 0 else 1)
        ax.set_title(binning.slice_label(i), fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        if chi2 is not None:
            ax.text(0.97, 0.95, chi2_text(chi2[i]), transform=ax.transAxes, ha='right', va='top', fontsize=8)
        if i == legend_slice:
            ax.legend(fontsize=7, frameon=False, loc='upper left')
        xa = ar if ar is not None else ax
        xa.set_xlabel(binning.x_label, fontsize=9)
        if ar is not None and points is not None:
            ref = np.asarray(ratio_to)[b]
            with np.errstate(divide='ignore', invalid='ignore'):
                ar.errorbar(ctr, v / ref, yerr=np.asarray(outer if outer is not None else inner)[b] / ref,
                            fmt='ko', ms=3, capsize=2)
            ar.axhline(1, color='gray', ls='--', lw=1); ar.set_ylim(0.5, 1.5)
            ar.set_ylabel('Ratio', fontsize=8)
    sup = title + (f"   (all slices, full covariance: {chi2_text(chi2['all'])})" if chi2 is not None else '')
    fig.suptitle(sup, fontsize=12, y=1.0)
    return fig, axes


def plot_serial(ax, vcfg, steps=(), points=None, bands=(), chi2=None, ylabel='', logy=False):
    """Fig. 17-style: every slice on one bin-index axis; per-slice χ² under the slice labels.
    logy: d²σ/dp dcosθ spans orders of magnitude between the backward and forward slices."""
    e, c = vcfg.disp_edges, vcfg.disp_centers
    for vals, hw, kw in bands:
        ax.bar(e[:-1], 2 * np.asarray(hw), bottom=np.asarray(vals) - np.asarray(hw), width=np.diff(e),
               align='edge', **kw)
    for vals, kw in steps:
        step(ax, e, vals, **kw)
    top = max([np.max(v) for v, _ in steps] + [0])
    if points is not None:
        v, inner, outer, kw = points
        if outer is not None:
            ax.errorbar(c, v, yerr=outer, fmt='none', ecolor=kw.get('color', 'k'), capsize=3, lw=1)
            top = max(top, np.max(np.asarray(v) + outer))
        ax.errorbar(c, v, yerr=inner, **{'fmt': 'ko', 'ms': 4, 'capsize': 0, 'lw': 2, **kw})
        top = max(top, np.max(np.asarray(v) + inner))
    if logy:
        pos = [np.min(np.asarray(v)[np.asarray(v) > 0]) for v, _ in steps if np.any(np.asarray(v) > 0)]
        ax.set_yscale('log'); ax.set_ylim(min(pos) * 0.3 if pos else 1e-3, top * 8 if top > 0 else 1)
    else:
        ax.set_ylim(0, top * 1.5 if top > 0 else 1)
    ax.set_ylabel(ylabel)
    vcfg.set_xaxis(ax, vcfg.xlabel)
    b = vcfg.serial
    if chi2 is not None:
        import matplotlib.transforms as mtrans
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        for i in range(b.n_slices):
            mid = 0.5 * (e[b.offsets[i]] + e[b.offsets[i + 1]])
            c2, ndf, _ = chi2[i]
            ax.text(mid, 0.93, f'χ² {c2:.1f}/{ndf}', transform=tr, ha='center', va='top', fontsize=7,
                    color='dimgray')
    return ax


def plot_cov_slices(cov, cv, binning, vcfg, chi2=None, unit='', title=''):
    """Figs. 79–83 style: total covariance, fractional covariance and correlation.
    Returns (grouped figure over all slices, figure with one row per slice: that slice's block).
    The per-slice block is the matrix used for that slice's χ² (chi2 → written in the titles)."""
    cov, cv = np.asarray(cov, float), np.asarray(cv, float)
    mats = [(cov, f'Covariance {unit}', 'viridis', None, None),
            (frac_cov(cov, cv), 'Fractional covariance', 'viridis', None, None),
            (corr_from_cov(cov), 'Correlation', 'RdBu_r', -1, 1)]
    fig_all, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    for ax, (M, lab, cmap, lo, hi) in zip(axes, mats):
        plot_matrix(ax, M, vcfg, cmap=cmap, vmin=lo, vmax=hi, title=lab, annotate=False,
                    xlabel=vcfg.xlabel_true, ylabel=vcfg.xlabel_true)
    fig_all.suptitle(title + (f"   ({chi2_text(chi2['all'])})" if chi2 is not None else ''), y=1.03)
    fig_all.tight_layout()

    fig_sl, axs = plt.subplots(binning.n_slices, 3, figsize=(16, 3.9 * binning.n_slices))
    for i in range(binning.n_slices):
        b = binning.block(i)
        e = binning.x_disp_edges(i)
        lab = [f'{x:g}' for x in e]
        for j, (M, name, cmap, lo, hi) in enumerate(mats):
            ax = axs[i, j]
            Mb = M[b, b]
            im = ax.imshow(Mb, origin='lower', cmap=cmap, vmin=lo, vmax=hi, aspect='auto',
                           extent=[0, Mb.shape[1], 0, Mb.shape[0]])
            plt.colorbar(im, ax=ax)
            n = Mb.shape[0]
            ax.set_xticks(np.arange(n + 1)); ax.set_xticklabels(lab, fontsize=8)
            ax.set_yticks(np.arange(n + 1)); ax.set_yticklabels(lab, fontsize=8)
            ax.set_xlabel(binning.x_label, fontsize=9); ax.set_ylabel(binning.x_label, fontsize=9)
            fmt = '{:.2f}' if j == 2 else '{:.1e}' if j == 0 else '{:.3f}'
            cm = plt.get_cmap(cmap)
            for r in range(n):
                for c in range(n):
                    r_, g_, b_, _ = cm(im.norm(Mb[r, c]))
                    ax.text(c + 0.5, r + 0.5, fmt.format(Mb[r, c]), ha='center', va='center', fontsize=7,
                            color='black' if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 0.5 else 'white')
            t = f'{binning.slice_label(i)} — {name}'
            if chi2 is not None and j == 0:
                t += f'\n{chi2_text(chi2[i])}'
            ax.set_title(t, fontsize=9)
    fig_sl.tight_layout()
    return fig_all, fig_sl


def plot_block(ax, M, binning, i, j=None, cmap='viridis', vmin=None, vmax=None, title='',
               fmt='{:.2f}', annotate=True):
    """Fig. 111 style: block (slice i rows, slice j columns) of a serialized matrix.
    y = global bin ID of slice i, x = true variable of slice j (real edges)."""
    j = i if j is None else j
    bi, bj = binning.block(i), binning.block(j)
    Mb = np.asarray(M)[bi, bj]
    nr, nc = Mb.shape
    im = ax.imshow(Mb, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   extent=[0, nc, 0, nr])
    plt.colorbar(im, ax=ax)
    ex = binning.x_disp_edges(j)
    ax.set_xticks(np.arange(nc + 1)); ax.set_xticklabels([f'{x:g}' for x in ex], fontsize=8)
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.set_yticklabels([str(k) for k in range(bi.start, bi.stop)], fontsize=8)
    ax.set_xlabel(f'True {binning.x_label}', fontsize=9); ax.set_ylabel('Bin ID #', fontsize=9)
    ax.set_title(title or binning.slice_label(i), fontsize=9)
    if annotate:
        cm = plt.get_cmap(cmap)
        for r in range(nr):
            for c in range(nc):
                r_, g_, b_, _ = cm(im.norm(Mb[r, c]))
                ax.text(c + 0.5, r + 0.5, fmt.format(Mb[r, c]), ha='center', va='center', fontsize=7,
                        color='black' if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 0.5 else 'white')
    return im


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


# =============================================================================
# NPZ archiving
# =============================================================================

def save_cov_npz(filename, **arrays):
    np.savez_compressed(filename, **arrays)
    print(f"Saved {len(arrays)} arrays → {filename}")
