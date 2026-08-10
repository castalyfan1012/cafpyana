"""
nue_helpers.py
--------------
Plotting and utility helpers for the nueCC inclusive SPINE analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ============================================================
# chi2 / p-value utilities
# ============================================================

def chi2_pvalue(observed, expected, cov_matrix=None, stat_errors=None):
    from scipy.stats import chi2 as chi2_dist
    obs  = np.asarray(observed, dtype=float)
    exp  = np.asarray(expected, dtype=float)
    diff = obs - exp
    n    = len(diff)
    if cov_matrix is not None:
        cov = np.asarray(cov_matrix, dtype=float)
        reg = np.eye(n) * max(1e-10, 1e-6 * np.abs(np.diag(cov)).max())
        try:
            cov_inv = np.linalg.inv(cov + reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov + reg)
        chi2_val = float(diff @ cov_inv @ diff)
    else:
        if stat_errors is not None:
            sigma2 = np.asarray(stat_errors, dtype=float) ** 2
        else:
            sigma2 = np.where(exp > 0, exp, 1.0)
        chi2_val = float(np.sum(diff**2 / np.where(sigma2 > 0, sigma2, 1.0)))
    ndof = n
    pval = float(1.0 - chi2_dist.cdf(chi2_val, ndof))
    return chi2_val, ndof, pval


def chi2_text(observed, expected, cov_matrix=None, stat_errors=None, label=""):
    chi2_val, ndof, pval = chi2_pvalue(observed, expected,
                                        cov_matrix=cov_matrix,
                                        stat_errors=stat_errors)
    prefix = f"{label}" if label else ""
    return f"{prefix}$\\chi^2$/ndof = {chi2_val:.1f}/{ndof} (p = {pval:.2f})"


def annotate_chi2(ax, observed, expected, cov_matrix=None, stat_errors=None,
                  label="", loc="upper right", fontsize=10):
    chi2_val, ndof, pval = chi2_pvalue(observed, expected,
                                        cov_matrix=cov_matrix,
                                        stat_errors=stat_errors)
    text = chi2_text(observed, expected, cov_matrix=cov_matrix,
                     stat_errors=stat_errors, label=label)
    ha_map = {"upper right": ("right", 0.97), "upper left": ("left", 0.03),
              "lower right": ("right", 0.97), "lower left": ("left", 0.03)}
    va_map = {"upper right": ("top", 0.88),   "upper left": ("top", 0.88),
              "lower right": ("bottom", 0.05), "lower left": ("bottom", 0.05)}
    ha, x = ha_map.get(loc, ("right", 0.97))
    va, y = va_map.get(loc, ("top", 0.88))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.3))
    return chi2_val, ndof, pval


# ============================================================
# Colour / style palette
# ============================================================

CAT_COLORS = {
    0:  "#56B4E9",   # nueCC in FV (signal)
    1:  "#0072B2",   # nueCC out FV
    2:  "green",     # numuCC + pi0
    3:  "#009E73",   # NC pi0
    4:  "#E69F00",   # other numuCC
    5:  "#CC79A7",   # NC other
    6:  "#F0E442",   # cosmic
    7:  "#BDBDBD",   # offbeam  
    9:  "#D2691E",   # dirt nu   
   -1:  "#7F7F7F",   # unknown
}
CAT_LABELS = {
    0:  r"$\nu_e$ CC (FV)",
    1:  r"$\nu_e$ CC (out FV)",
    2:  r"$\nu_\mu$ CC $\pi^0$",
    3:  r"$\nu$ NC $\pi^0$",
    4:  r"Other $\nu_\mu$ CC",
    5:  r"Other $\nu$ NC",
    6:  r"Cosmic",
    7:  r"Offbeam",  
    9:  r"Dirt $\nu$", 
   -1:  r"Unknown",
}

PID_PHOTON   = 0
PID_ELECTRON = 1
PID_MUON     = 2
PID_PION     = 3
PID_PROTON   = 4


# ============================================================
# ParticleView / InteractionView wrappers
# ============================================================

class ParticleView:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getattr__(self, name: str):
        try:
            return self._df[name]
        except KeyError:
            raise AttributeError(f"No column '{name}' in ParticleView") from None

    def __getitem__(self, key):
        if isinstance(self._df.index, pd.MultiIndex):
            try:
                return self._df.xs(key, level=0)
            except KeyError:
                return self._df.loc[key]
        return self._df.loc[key]

    def __len__(self):         return len(self._df)
    def __repr__(self):        return f"ParticleView({len(self._df):,} rows)"
    @property
    def df(self):              return self._df
    @property
    def index(self):           return self._df.index
    @property
    def columns(self):         return self._df.columns
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)


class InteractionView:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getattr__(self, name: str):
        try:
            return self._df[name]
        except KeyError:
            raise AttributeError(f"No column '{name}' in InteractionView") from None

    def __getitem__(self, key): return self._df[key]
    def __len__(self):          return len(self._df)
    def __repr__(self):         return f"InteractionView({len(self._df):,} rows)"
    @property
    def df(self):               return self._df
    @property
    def index(self):            return self._df.index
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)


def reco_particles(evtdf)    -> ParticleView:    return ParticleView(evtdf["rec"]["dlp"]["particles"])
def true_particles(evtdf)    -> ParticleView:    return ParticleView(evtdf["rec"]["dlp_true"]["particles"])
def reco_interactions(evtdf) -> InteractionView: return InteractionView(evtdf["rec"]["dlp"])
def true_interactions(evtdf) -> InteractionView: return InteractionView(evtdf["rec"]["dlp_true"])


# ============================================================
# POT / index utilities
# ============================================================

def compute_pot_scale(histpotdf, target_pot: float = 6.6e20):
    total_pot = histpotdf["TotalPOT"].sum()
    return total_pot, target_pot / total_pot

def inter_levels(evtdf) -> list:
    return list(range(evtdf.index.nlevels - 1))

def collapse_to_interactions(evtdf):
    return evtdf.groupby(level=inter_levels(evtdf)).first()


# ============================================================
# Selection DataFrame I/O
# ============================================================

_SEL_HDF_KEY = "sel_0"

def build_and_save_selection(path, presel_inter, truth_cat_presel,
                              reco_ke, reco_cos, true_ke, true_cos,
                              stage_idx_dict, cut_names, pot_scale,
                              reco_p=None, true_p=None):
    import os, warnings
    sel = pd.DataFrame(index=presel_inter)
    sel["truth_cat"]    = truth_cat_presel.reindex(presel_inter).astype("int8")
    sel["is_sig"]       = (sel["truth_cat"] == 0)
    sel["reco_ke"]      = reco_ke.reindex(presel_inter).astype("float32")
    sel["reco_costheta"]= reco_cos.reindex(presel_inter).astype("float32")
    if reco_p is not None:
        sel["reco_p"]   = reco_p.reindex(presel_inter).astype("float32")
    if true_ke is not None:
        sel["true_ke"]  = true_ke.reindex(presel_inter).astype("float32")
    if true_cos is not None:
        sel["true_costheta"] = true_cos.reindex(presel_inter).astype("float32")
    if true_p is not None:
        sel["true_p"]   = true_p.reindex(presel_inter).astype("float32")
    for name in cut_names:
        if name in stage_idx_dict:
            sel[f"sel_{name}"] = presel_inter.isin(stage_idx_dict[name])
        else:
            warnings.warn(f"build_and_save_selection: stage {name!r} missing")
    sel["pot_scale"] = np.float32(pot_scale)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel.to_hdf(path, key=_SEL_HDF_KEY, mode="w", complevel=1, complib="blosc")
    size_mb = os.path.getsize(path) / 1024**2
    print(f"Saved selection df → {path}  shape={sel.shape}  size={size_mb:.1f} MB")
    return sel

def load_selection(path):
    return pd.read_hdf(path, key=_SEL_HDF_KEY)


# ============================================================
# Purity / efficiency
# ============================================================

def calc_pur_eff(sel_mask, sig_mask, weights=None):
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w = np.ones(len(sel_mask)) if weights is None else np.asarray(weights)
    n_sel      = w[sel_mask].sum()
    n_true_pos = w[sel_mask & sig_mask].sum()
    n_sig      = w[sig_mask].sum()
    pur = float(n_true_pos / n_sel) if n_sel > 0 else np.nan
    eff = float(n_true_pos / n_sig) if n_sig > 0 else np.nan
    return pur, eff

def pur_eff_cut_flow(cut_flow_masks, sig_mask, weights=None):
    return zip(*[calc_pur_eff(m, sig_mask, weights) for m in cut_flow_masks])

def pur_eff_binned(values, bins, sel_mask, sig_mask, weights=None):
    values   = np.asarray(values, dtype=float)
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w        = np.ones(len(values)) if weights is None else np.asarray(weights)
    bins     = np.asarray(bins)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    purs = np.full(len(centers), np.nan)
    effs = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin     = (values >= lo) & (values < hi)
        counts[i]  = w[in_bin & sig_mask].sum()
        purs[i], effs[i] = calc_pur_eff(sel_mask & in_bin, sig_mask & in_bin, w)
    return centers, purs, effs, counts


# ============================================================
# Stacked histogram
# ============================================================

def plot_stacked_hist(series_list, labels, colors, bins, weights=None,
                      xlabel="", ylabel="Events / bin", title="",
                      density=False, ax=None, data_series=None,
                      data_label="Data", pot_label="",
                      invert_stack_order=False,
                      show_counts=True, show_percentage=True,
                      chi2_info=None, **hist_kw):
    """
    Stacked MC histogram with optional data overlay.

    weights : None | scalar | list of arrays
        None    → unweighted
        scalar  → same weight for all categories
        list    → per-category weight arrays (required when categories have
                  different scales, e.g. MC + offbeam)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Normalise weights to a list of arrays
    if weights is None:
        _weights = [None] * len(series_list)
    elif np.isscalar(weights):
        _weights = [np.full(len(s), float(weights)) for s in series_list]
    elif isinstance(weights, (list, tuple)):
        if len(weights) != len(series_list):
            raise ValueError("Length of weights must match series_list.")
        _weights = [None if w is None else np.asarray(w, dtype=float)
                    for w in weights]
    else:
        raise TypeError("weights must be None, a scalar, or a list of arrays.")

    # Strip NaN keeping weights aligned
    clean = []
    for s, w in zip(series_list, _weights):
        s    = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        clean.append((s[mask], w[mask] if w is not None else None))

    bins_arr = np.asarray(bins)

    # Per-category totals for legend
    cat_totals = []
    for s, w in clean:
        h, _ = np.histogram(s, bins=bins_arr, weights=w)
        cat_totals.append(h.sum())
    grand_total = sum(cat_totals)

    # Annotated labels
    annotated_labels = []
    for lbl, tot in zip(labels, cat_totals):
        parts = [lbl]
        if show_counts or show_percentage:
            inner = []
            if show_counts:
                inner.append(f"{tot:.1f}")
            if show_percentage and grand_total > 0:
                inner.append(f"{tot/grand_total:.1%}")
            parts.append(f"({', '.join(inner)})")
        annotated_labels.append(" ".join(parts))

    if invert_stack_order:
        x_list   = [c[0] for c in reversed(clean)]
        w_list   = [c[1] for c in reversed(clean)]
        ann_lbl  = list(reversed(annotated_labels))
        col_list = list(reversed(colors))
    else:
        x_list   = [c[0] for c in clean]
        w_list   = [c[1] for c in clean]
        ann_lbl  = annotated_labels
        col_list = list(colors)

    weights_arg = None if all(w is None for w in w_list) else w_list

    ax.hist(x_list, bins=bins_arr, weights=weights_arg,
            label=ann_lbl, color=col_list,
            stacked=True, histtype="stepfilled", density=density, **hist_kw)

    if data_series is not None:
        d = np.asarray(data_series, dtype=float)
        d = d[~np.isnan(d)]
        counts, _ = np.histogram(d, bins=bins_arr)
        centers   = 0.5 * (bins_arr[:-1] + bins_arr[1:])
        ax.errorbar(centers, counts, yerr=np.sqrt(counts),
                    fmt="ko", markersize=4, label=data_label, zorder=10)

    handles, legend_labels = ax.get_legend_handles_labels()
    if invert_stack_order:
        ax.legend(handles, legend_labels, fontsize=9, ncol=1,
                  loc="upper right", frameon=True, framealpha=0.3, edgecolor="none")
    else:
        ax.legend(handles[::-1], legend_labels[::-1], fontsize=9, ncol=1,
                  loc="upper right", frameon=True, framealpha=0.3, edgecolor="none")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)
    if pot_label:
        ax.text(0.02, 0.98, pot_label, transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="gray")
        ax.text(0.02, 0.93, f"Total MC events: {grand_total:.0f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, color="gray")
    else:
        ax.text(0.02, 0.98, f"Total MC events: {grand_total:.0f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, color="gray")

    if chi2_info is not None:
        annotate_chi2(ax,
                      observed    = chi2_info['observed'],
                      expected    = chi2_info['expected'],
                      cov_matrix  = chi2_info.get('cov_matrix'),
                      stat_errors = chi2_info.get('stat_errors'),
                      label       = chi2_info.get('label', ''),
                      loc         = chi2_info.get('loc', 'upper left'),
                      fontsize    = chi2_info.get('fontsize', 10))

    return ax.get_figure(), ax


# ============================================================
# 2D resolution / bias plot
# ============================================================

def plot_hist2d_frac_err(x, y, xlabel='x', ylabel='y', title=None,
                         cmap='Blues', plot_line=True,
                         normalize=True, use_errorbar=False,
                         fit_curve=True, show_fit=False,
                         log_color=False, fontsize=13, **pltkwargs):
    from scipy.optimize import curve_fit
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import AutoMinorLocator

    if 'bins' not in pltkwargs:
        raise ValueError('bins must be provided as a keyword argument')
    bins        = np.asarray(pltkwargs.pop('bins'))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    def _crystal_ball(t, A, a, n, mu, sigma):
        t    = np.asarray(t, dtype=float)
        out  = np.empty_like(t)
        z    = (t - mu) / sigma
        g    = z > -a
        out[g]  = np.exp(-0.5 * z[g]**2)
        Ac      = (n / a)**n * np.exp(-0.5 * a**2)
        B       = n / a - a
        out[~g] = Ac * (B - z[~g])**(-n)
        return A * out

    bias = np.full(len(bin_centers), np.nan)
    err  = np.full(len(bin_centers), np.nan)

    for i in range(len(bins) - 1):
        in_range = (x > bins[i]) & (x < bins[i + 1])
        _xb, _yb = x[in_range], y[in_range]
        if normalize:
            keep = np.abs(_xb) > 1e-6
            _xb, _yb = _xb[keep], _yb[keep]
        if len(_xb) < 5:
            continue
        stat = (_yb - _xb) / _xb if normalize else (_yb - _xb)
        stat = stat.dropna()
        if len(stat) < 5:
            continue
        _mean = float(np.nanmean(stat))
        _std  = max(float(np.nanstd(stat)), 1e-9)
        if fit_curve:
            _half  = 5.0 * _std
            _fbins = np.linspace(_mean - _half, _mean + _half, 30)
            _h, _e = np.histogram(stat, bins=_fbins)
            _c     = 0.5 * (_e[:-1] + _e[1:])
            try:
                A0 = max(float(_h.max()), 1.0)
                popt, _ = curve_fit(
                    _crystal_ball, _c, _h,
                    p0     = [A0, 1.5, 2.0, _mean, _std],
                    bounds = ([0, 0.1, 1.01, _mean - 3*_std, 1e-9],
                              [20*A0+1, 5.0, 50.0, _mean + 3*_std, 5*_std]),
                    maxfev = 8000)
                bias[i] = popt[3]
                err[i]  = popt[4]
            except Exception:
                bias[i] = _mean
                err[i]  = _std
            if show_fit:
                _t  = np.linspace(_mean - _half, _mean + _half, 200)
                plt.figure()
                plt.hist(stat, bins=_fbins, density=True, alpha=0.6,
                         label=f'Raw (n={len(stat):,})')
                _cb  = _crystal_ball(_t, *popt)
                _cb /= (_cb.sum() * (_t[1] - _t[0]))
                plt.plot(_t, _cb, lw=2,
                         label=f'CB  μ={popt[3]:.3f}, σ={popt[4]:.3f}')
                plt.axvline(popt[3], ls='--', color='blue', label='μ')
                plt.axvline(popt[3] - popt[4], ls=':', color='green', label='μ±σ')
                plt.axvline(popt[3] + popt[4], ls=':', color='green')
                plt.title(f'Bin [{bins[i]:.3g}, {bins[i+1]:.3g}]')
                plt.legend(fontsize=8)
                plt.show()
        else:
            bias[i] = _mean
            err[i]  = _std

    fig = plt.figure(figsize=(6, 8))
    gs  = GridSpec(2, 2, height_ratios=[4,1], width_ratios=[20,1],
                   hspace=0.03, wspace=0.15)
    ax_main = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[1, 0])
    cax     = fig.add_subplot(gs[0, 1])
    plt.setp(ax_main.get_xticklabels(), visible=False)

    valid_mask = x.notna() & y.notna()
    _norm      = LogNorm() if log_color else None
    h = ax_main.hist2d(x[valid_mask].values, y[valid_mask].values,
                       bins=[bins, bins], cmap=cmap, norm=_norm, **pltkwargs)
    fig.colorbar(h[3], cax=cax, label='Counts')
    cax.yaxis.label.set_size(fontsize - 2)
    cax.tick_params(labelsize=fontsize - 3)

    if plot_line:
        ax_main.plot([bins[0], bins[-1]], [bins[0], bins[-1]],
                     ls='--', color='red', lw=1.2, zorder=5)
    ax_main.set_xlim(bins[0], bins[-1])
    ax_main.set_ylim(bins[0], bins[-1])
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_ylabel(ylabel, fontsize=fontsize)
    ax_main.tick_params(axis='both', labelsize=fontsize - 2)
    if title:
        ax_main.set_title(title, fontsize=fontsize)
    ax_main.text(0.03, 0.97, r'$\mathbf{SBND}$ Simulation',
                 transform=ax_main.transAxes, va='top', ha='left',
                 fontsize=fontsize - 2, color='gray')

    if use_errorbar:
        ax2.errorbar(bin_centers, bias, yerr=err, fmt='o', color='black', markersize=5)
    else:
        ax2.scatter(bin_centers, bias, color='black', s=25, label='Frac. bias', zorder=4)
        ax2.scatter(bin_centers, err,  color='green', s=40, label='Resolution (CB σ)',
                    marker='*', zorder=4)
        ax2.legend(fontsize=11, ncol=1, loc='upper right', framealpha=0.35)

    ax2.axhline(0, ls='--', color='black', lw=1)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_ylim(-0.05, 0.2)
    ax2.tick_params(axis='both', labelsize=fontsize - 2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(which='major', linestyle='-',  linewidth=0.6, alpha=0.45)
    ax2.grid(which='minor', linestyle='--', linewidth=0.3, alpha=0.25)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    plt.tight_layout()

    fig.canvas.draw()
    ax1_pos = ax_main.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax1_pos.y0, cax_pos.width, ax1_pos.height])
    return fig, (ax_main, ax2)


def plot_leading_e_resolution(true_series, reco_series, bins,
                               stage_idx=None, xlabel='True value',
                               ylabel='Reco value', normalize=True,
                               fit_curve=True, title=None, **kwargs):
    if stage_idx is not None:
        mask        = true_series.index.isin(stage_idx)
        true_series = true_series[mask]
        reco_series = reco_series[mask]
    valid = true_series.notna() & reco_series.notna()
    return plot_hist2d_frac_err(
        true_series[valid].astype(float), reco_series[valid].astype(float),
        xlabel=xlabel, ylabel=ylabel, title=title,
        normalize=normalize, fit_curve=fit_curve,
        bins=np.asarray(bins), **kwargs)


# ============================================================
# Threshold scan plots
# ============================================================

def threshold_plots(evtdf, cut_flow_result, cut_stage, metric_col,
                    bounds, xaxis_name, plot_title, threshold_range,
                    cut_dir='<', legend_loc='right', truth_categories=None,
                    fig_width=7, save_name=None, plot_folder="plots"):
    from nue_selection import PID_ELECTRON, classify_truth
    il         = inter_levels(evtdf)
    presel_idx = cut_flow_result[cut_stage]["inter_index"]
    cat = truth_categories if truth_categories is not None else classify_truth(evtdf, verbose=False)
    rp_df = reco_particles(evtdf)._df

    _MULTI = {'pid_scores': ('pid_scores','I1'), 'primary_scores': ('primary_scores','I1')}
    if isinstance(metric_col, str) and metric_col in _MULTI:
        col_key = _MULTI[metric_col]
    elif isinstance(metric_col, str):
        if metric_col in rp_df.columns:
            col_key = metric_col
        elif (metric_col, '') in rp_df.columns:
            col_key = (metric_col, '')
        else:
            matches = [c for c in rp_df.columns
                       if (isinstance(c, tuple) and c[0] == metric_col) or c == metric_col]
            if not matches:
                raise KeyError(f"'{metric_col}' not found in reco_particles.")
            col_key = matches[0]
    else:
        col_key = metric_col

    test = rp_df[col_key]
    if isinstance(test, pd.DataFrame):
        raise KeyError(f"col_key={col_key!r} resolves to DataFrame, not Series.")

    ke_col  = ('ke','')  if ('ke','')  in rp_df.columns else 'ke'
    pid_col = ('pid','') if ('pid','') in rp_df.columns else 'pid'

    in_presel  = rp_df.index.droplevel(-1).isin(presel_idx)
    ele_presel = rp_df[in_presel & (rp_df[pid_col] == PID_ELECTRON)]
    tmp = pd.DataFrame({'ke': ele_presel[ke_col].to_numpy(dtype=float),
                        'metric': ele_presel[col_key].to_numpy(dtype=float)},
                       index=ele_presel.index)

    def metric_at_max_ke(g): return g.loc[g['ke'].idxmax(), 'metric']
    leading_metric = tmp.groupby(level=il).apply(metric_at_max_ke)
    inter_index    = leading_metric.index
    metric_vals    = leading_metric.to_numpy(dtype=float).flatten()
    is_signal      = (cat.reindex(inter_index).fillna(-1).to_numpy().flatten() == 0)
    valid          = ~np.isnan(metric_vals) & (metric_vals != -1)
    metric_vals    = metric_vals[valid]
    is_signal      = is_signal[valid]

    if len(metric_vals) == 0:
        print(f"No valid events for '{metric_col}'."); return None, None

    xmin, xmax   = bounds
    bin_edges    = np.linspace(xmin, xmax, 51)
    signal_vals  = metric_vals[is_signal]
    bkg_vals     = metric_vals[~is_signal]

    fig = plt.figure(figsize=(fig_width, 8))
    gs  = fig.add_gridspec(4, 1, hspace=0.05)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.hist(bkg_vals,    bins=bin_edges, histtype='step', density=True,
             label=f'Background ({len(bkg_vals):,})',    color='orange', lw=1.5)
    ax1.hist(signal_vals, bins=bin_edges, histtype='step', density=True,
             label=r'Signal $\nu_e$ CC' + f' ({len(signal_vals):,})', color='blue', lw=1.5)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.set_title(plot_title, fontsize=16, pad=10)
    ax1.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)

    start, stop, step = threshold_range
    thresholds   = np.arange(start, stop + step/2, step)
    if len(thresholds) <= 1:
        thresholds = np.linspace(start, stop, 120)
    total_signal = is_signal.sum()
    keep_above   = (cut_dir == '>')

    purities, efficiencies, f1s = [], [], []
    for thresh in thresholds:
        passed       = (metric_vals > thresh) if keep_above else (metric_vals < thresh)
        n_passed     = passed.sum()
        n_sig_passed = (passed & is_signal).sum()
        pur = n_sig_passed / n_passed     if n_passed     > 0 else 0.0
        eff = n_sig_passed / total_signal if total_signal > 0 else 0.0
        f1  = 2*pur*eff / (pur+eff)       if (pur+eff)   > 0 else 0.0
        purities.append(pur); efficiencies.append(eff); f1s.append(f1)

    f1s        = np.array(f1s)
    opt_idx    = np.argmax(f1s)
    opt_thresh = float(thresholds[opt_idx])
    max_f1     = float(f1s[opt_idx])

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(thresholds, purities,     'b-',           label='Purity',     lw=1.5)
    ax2.plot(thresholds, efficiencies, 'r-',           label='Efficiency', lw=1.5)
    ax2.plot(thresholds, f1s,          color='purple', label='F1 Score',   lw=1.5)
    ax2.axvline(opt_thresh, color='purple', ls='--', lw=1.5)
    ax2.set_xlabel(xaxis_name, fontsize=14)
    ax2.set_ylabel('Performance', fontsize=14)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(0, 1)
    ax2.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)
    range_x = xmax - xmin
    text_x = opt_thresh + range_x*0.05 if cut_dir == '<' else opt_thresh - range_x*0.05
    ha     = 'left'                     if cut_dir == '<' else 'right'
    ax2.text(text_x, 0.05, f'Max F1 at {opt_thresh:.3f}', color='purple',
             va='bottom', ha=ha, fontsize=13,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                       edgecolor='none', alpha=0.5))
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    if save_name is not None:
        save_plot(save_name, fig=fig, folder_name=plot_folder)
    plt.show()

    print(f"\n{'─'*42}")
    print(f"  Variable  : {metric_col}")
    print(f"  Cut dir   : {cut_dir} {opt_thresh:.4f}")
    print(f"  Efficiency: {efficiencies[opt_idx]*100:.1f}%")
    print(f"  Purity    : {purities[opt_idx]*100:.1f}%")
    print(f"  Max F1    : {max_f1:.4f}")
    print(f"{'─'*42}\n")
    return opt_thresh, max_f1


def threshold_plots_cumulative(evtdf, cut_flow_result, cut_stage,
                                shower_cuts, truth_categories=None,
                                opt_thresholds_prev=None,
                                fig_width=7, plot_folder="plots"):
    from nue_selection import PID_ELECTRON, classify_truth
    il         = inter_levels(evtdf)
    presel_idx = cut_flow_result[cut_stage]["inter_index"]
    cat = truth_categories if truth_categories is not None else classify_truth(evtdf, verbose=False)
    rp_df = reco_particles(evtdf)._df

    _MULTI = {'pid_scores': ('pid_scores','I1'), 'primary_scores': ('primary_scores','I1')}
    def _resolve_col(mc):
        if isinstance(mc, str) and mc in _MULTI: return _MULTI[mc]
        if isinstance(mc, str):
            if mc in rp_df.columns: return mc
            if (mc,'') in rp_df.columns: return (mc,'')
            m = [c for c in rp_df.columns
                 if (isinstance(c,tuple) and c[0]==mc) or c==mc]
            if not m: raise KeyError(f"'{mc}' not found in reco_particles.")
            return m[0]
        return mc

    ke_col  = ('ke','')  if ('ke','')  in rp_df.columns else 'ke'
    pid_col = ('pid','') if ('pid','') in rp_df.columns else 'pid'
    in_presel  = rp_df.index.droplevel(-1).isin(presel_idx)
    ele_presel = rp_df[in_presel & (rp_df[pid_col] == PID_ELECTRON)]

    metric_frames = {}
    for (mc, *_) in shower_cuts:
        ck  = _resolve_col(mc)
        tmp = pd.DataFrame({'ke': ele_presel[ke_col].to_numpy(dtype=float),
                            'metric': ele_presel[ck].to_numpy(dtype=float)},
                           index=ele_presel.index)
        def _mke(g): return g.loc[g['ke'].idxmax(), 'metric']
        metric_frames[mc] = tmp.groupby(level=il).apply(_mke)

    master = pd.DataFrame(metric_frames)
    master['is_signal'] = (cat.reindex(master.index).fillna(-1) == 0).values

    surviving_mask = pd.Series(True, index=master.index)
    if opt_thresholds_prev:
        for mc, bounds, xaxis_name, trange, cut_dir, lloc, sname in shower_cuts:
            if mc in opt_thresholds_prev:
                thresh = opt_thresholds_prev[mc]
                surviving_mask &= (master[mc] > thresh if cut_dir == '>' else master[mc] < thresh)

    opt_thresholds = {}

    for mc, bounds, xaxis_name, trange, cut_dir, lloc, sname in shower_cuts:
        xmin, xmax   = bounds
        current      = master[surviving_mask]
        metric_vals  = current[mc].to_numpy(dtype=float)
        is_signal    = current['is_signal'].to_numpy()
        valid        = ~np.isnan(metric_vals)
        metric_vals  = metric_vals[valid]; is_signal = is_signal[valid]
        n_total  = len(metric_vals)
        n_signal = is_signal.sum()
        n_bkg    = (~is_signal).sum()

        fig = plt.figure(figsize=(fig_width, 8))
        gs  = fig.add_gridspec(4, 1, hspace=0.05)
        ax1 = fig.add_subplot(gs[:3, 0])
        ax1.hist(metric_vals[~is_signal], bins=np.linspace(xmin,xmax,51),
                 histtype='step', density=True,
                 label=f'Background ({n_bkg:,} remaining)', color='orange', lw=1.5)
        ax1.hist(metric_vals[is_signal],  bins=np.linspace(xmin,xmax,51),
                 histtype='step', density=True,
                 label=r'Signal $\nu_e$ CC' + f' ({n_signal:,} remaining)',
                 color='blue', lw=1.5)
        ax1.set_title(xaxis_name, fontsize=14, pad=8)
        ax1.set_ylabel('Probability Density', fontsize=13)
        ax1.legend(loc=f'upper {lloc}', framealpha=0.3, fontsize=11)
        ax1.set_xlim(xmin, xmax); ax1.tick_params(labelbottom=False)
        ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)

        start, stop, step = trange
        thresholds   = np.arange(start, stop+step/2, step)
        total_signal = is_signal.sum()
        keep_above   = (cut_dir == '>')
        purities, efficiencies, f1s = [], [], []
        for thresh in thresholds:
            passed       = (metric_vals > thresh) if keep_above else (metric_vals < thresh)
            n_passed     = passed.sum()
            n_sig_passed = (passed & is_signal).sum()
            pur = n_sig_passed/n_passed     if n_passed     > 0 else 0.0
            eff = n_sig_passed/total_signal if total_signal > 0 else 0.0
            f1  = 2*pur*eff/(pur+eff)       if (pur+eff)   > 0 else 0.0
            purities.append(pur); efficiencies.append(eff); f1s.append(f1)

        f1s = np.array(f1s); opt_idx = np.argmax(f1s)
        opt_thresh = float(thresholds[opt_idx]); max_f1 = float(f1s[opt_idx])

        ax2 = fig.add_subplot(gs[3, 0])
        ax2.plot(thresholds, purities,     'b-',           label='Purity',     lw=1.5)
        ax2.plot(thresholds, efficiencies, 'r-',           label='Efficiency', lw=1.5)
        ax2.plot(thresholds, f1s,          color='purple', label='F1 Score',   lw=1.5)
        ax2.axvline(opt_thresh, color='purple', ls='--', lw=1.5)
        ax2.set_xlabel(xaxis_name, fontsize=13); ax2.set_ylabel('Performance', fontsize=13)
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(0, 1)
        ax2.legend(loc=f'upper {lloc}', framealpha=0.3, fontsize=11)
        range_x = xmax - xmin
        text_x = opt_thresh+range_x*0.05 if cut_dir=='<' else opt_thresh-range_x*0.05
        ha     = 'left' if cut_dir == '<' else 'right'
        ax2.text(text_x, 0.05, f'Max F1 at {opt_thresh:.3f}', color='purple',
                 va='bottom', ha=ha, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='none', alpha=0.5))
        ax2.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        if sname:
            save_plot(sname + "_cumulative", fig=fig, folder_name=plot_folder)
        plt.show()

        print(f"\n{'─'*50}")
        print(f"  Variable   : {mc}")
        print(f"  Remaining  : {n_total:,} ({n_signal} signal, {n_bkg} bkg)")
        print(f"  Cut dir    : {cut_dir} {opt_thresh:.4f}")
        print(f"  Efficiency : {efficiencies[opt_idx]*100:.1f}%")
        print(f"  Purity     : {purities[opt_idx]*100:.1f}%")
        print(f"  Max F1     : {max_f1:.4f}")
        print(f"{'─'*50}\n")

        opt_thresholds[mc] = opt_thresh
        surviving_mask &= (master[mc] > opt_thresh if cut_dir == '>' else master[mc] < opt_thresh)

    print(f"\nFinal surviving interactions: {surviving_mask.sum():,}")
    return opt_thresholds


# ============================================================
# Efficiency / purity plot helpers
# ============================================================

def plot_eff_pur_vs_bin(bin_centers, purs, effs, counts_true=None,
                        xlabel="", title="", ax=None, ax2=None,
                        eff_color="blue", pur_color="orange"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if counts_true is not None:
            ax2 = ax.twinx()
    ax.plot(bin_centers, effs, "o-", color=eff_color, label="Efficiency")
    ax.plot(bin_centers, purs, "s--", color=pur_color, label="Purity")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title(title)
    ax.legend(fontsize=9)
    if counts_true is not None and ax2 is not None:
        ax2.bar(bin_centers, counts_true,
                width=np.diff(bin_centers).mean() * 0.8,
                alpha=0.2, color="gray", label="True signal")
        ax2.set_ylabel("True signal events", fontsize=10)
    return ax.get_figure(), ax


def create_purity_efficiency_table(purs, effs, labels, ax):
    ax.axis("off")
    pur_pct = [p*100 if (p is not None and not np.isnan(p)) else np.nan for p in purs]
    eff_pct = [e*100 if (e is not None and not np.isnan(e)) else np.nan for e in effs]
    results = pd.DataFrame({
        "Cut": list(labels),
        "Purity [%]":    [f"{p:.2f}" if not np.isnan(p) else "—" for p in pur_pct],
        "Efficiency [%]":[f"{e:.2f}" if not np.isnan(e) else "—" for e in eff_pct],
        "ΔEff [%]": ["—"] + [f"{eff_pct[i]-eff_pct[i-1]:.2f}" for i in range(1,len(eff_pct))],
        "ΔPur [%]": ["—"] + [f"{pur_pct[i]-pur_pct[i-1]:.2f}" for i in range(1,len(pur_pct))],
    })
    table = ax.table(cellText=results.values.tolist(),
                     colLabels=results.columns.tolist(),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10 if len(results) <= 10 else 9)
    n_rows    = len(results) + 1
    row_scale = max(1.05, 18.0 / n_rows)
    table.scale(1.0, row_scale)
    col_widths = {0: None, 1: 0.13, 2: 0.15, 3: 0.12, 4: 0.12}
    for (row, col), cell in table.get_celld().items():
        w = col_widths.get(col)
        if w is not None: cell.set_width(w)
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf0fb"); cell.set_edgecolor("#cccccc")
        else:
            cell.set_facecolor("#ffffff"); cell.set_edgecolor("#cccccc")
        if row > 0 and col in (3, 4):
            cell.set_text_props(color="#555555", style="italic")
    return ax, table


def plot_matrix_binned(vals, count_matrix, bin_values, bin_labels, cut_labels,
                       remove_dummy_col=False, xlabel="", title="",
                       cmap="Greens", vmin=0, vmax=1,
                       label_fontsize=11, cell_size=0.8):
    if remove_dummy_col:
        vals = vals[:, :-1]; count_matrix = count_matrix[:, :-1]; cut_labels = cut_labels[:-1]
    n_bins, n_cuts = vals.shape
    fig, ax = plt.subplots(figsize=(cell_size*n_bins*1.3, cell_size*n_cuts*1.3))
    im = ax.imshow(vals.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticks(np.arange(n_cuts))
    ax.set_yticklabels(cut_labels, fontsize=label_fontsize)
    for ci in range(n_cuts):
        for bi in range(n_bins):
            v = vals[bi, ci]; n = count_matrix[bi, ci]
            if not np.isnan(v):
                ax.text(bi, ci, f"{v:.3f}\n({int(n)})",
                        ha="center", va="center", fontsize=label_fontsize,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cut stage", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


# ============================================================
# PID fraction vs true KE
# ============================================================

PID_NAMES  = {0:"Photon",   1:"Electron", 2:"Muon",    3:"Pion",    4:"Proton"}
PID_COLORS = {0:"#CC79A7",  1:"#9B8EC4",  2:"#56C8C8", 3:"#88CCAA", 4:"#AADDAA"}

def plot_pid_fraction_vs_ke(true_ke, reco_pid, ke_bins=None,
                             true_pid_target_name="True Electron (Primaries)",
                             watermark="SBND Work in Progress",
                             highlight_pid=None, title_fontsize=12,
                             figsize=(6, 7), ax1=None, ax2=None):
    _own = (ax1 is None) or (ax2 is None)
    if _own:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios":[3,1.2],"hspace":0.06},
                                        sharex=True)
    else:
        fig = ax1.get_figure()

    if ke_bins is None:
        ke_bins = np.linspace(0, 200, 21)
    ke_bins  = np.asarray(ke_bins, dtype=float)
    true_ke  = np.asarray(true_ke,  dtype=float)
    reco_pid = np.asarray(reco_pid, dtype=int)
    valid    = ~np.isnan(true_ke) & (reco_pid >= 0)
    true_ke  = true_ke[valid]; reco_pid = reco_pid[valid]

    n_bins  = len(ke_bins) - 1
    pid_ids = sorted(PID_NAMES.keys())
    counts  = np.zeros((n_bins, len(pid_ids)), dtype=float)
    for j, pid in enumerate(pid_ids):
        counts[:, j], _ = np.histogram(true_ke[reco_pid == pid], bins=ke_bins)
    total_per_bin = counts.sum(axis=1, keepdims=True)
    fracs = np.where(total_per_bin > 0, counts / total_per_bin, 0.0)

    for j, pid in enumerate(pid_ids):
        ls = "-" if (highlight_pid is not None and pid == highlight_pid) else "--"
        ax1.step(ke_bins, np.append(fracs[:, j], fracs[-1, j]),
                 where="post", color=PID_COLORS[pid], ls=ls, lw=1.8,
                 label=PID_NAMES[pid], alpha=0.3)
    ax1.set_ylabel("Fraction predicted", fontsize=12)
    ax1.set_ylim(0, 1.05); ax1.set_xlim(ke_bins[0], ke_bins[-1])
    ax1.legend(fontsize=12, loc="best", framealpha=0.3)
    ax1.set_title(true_pid_target_name, fontsize=title_fontsize, fontweight="bold", pad=4)
    ax1.text(0.98, 0.98, watermark, transform=ax1.transAxes,
             ha="right", va="top", fontsize=12, color="gray")
    ax1.tick_params(labelbottom=False)

    for j, pid in enumerate(pid_ids):
        ls = "-" if (highlight_pid is not None and pid == highlight_pid) else "--"
        ax2.step(ke_bins, np.append(counts[:, j], counts[-1, j]),
                 where="post", color=PID_COLORS[pid], ls=ls, lw=1.8, alpha=0.3)
    ax2.set_yscale("log")
    ax2.set_ylabel("True particles", fontsize=12)
    ax2.set_xlabel("True KE [MeV]", fontsize=12)
    ax2.set_xlim(ke_bins[0], ke_bins[-1])
    ax2.tick_params(which="both", direction="in", top=True, right=True)
    if _own: fig.tight_layout()
    return fig, (ax1, ax2)


# ============================================================
# Extra selection variable plots (used in syst notebook)
# ============================================================

def plot_extra_selection_vars(mc_evtdf, data_evtdf, sel_topo,
                              mc_se_idx, data_se_idx, mc_il, data_il,
                              truth_cat_at_topo, pot_scale_to_data,
                              data_total_pot, cov_results, cv_results,
                              var_cfgs, chi2_text_fn, savefig_fn,
                              title=r'SBND $\nu_e$ CC Inclusive',
                              display_cats=None,
                              stage_label='After topology cut',
                              data_label='On-beam data',
                              plot_dir='plots_syst',
                              offbeam_shower_vars=None,
                              offbeam_weight=None,
                              dirt_evtdf=None,
                              dirt_se_idx=None,
                              dirt_il=None,
                              dirt_weight=None):
    import nue_selection as _ns
    if display_cats is None:
        display_cats = [0,1,2,3,4,5,6]

    def _mat(block):
        return block.get('cov', block) if isinstance(block, dict) else block

    EXTRA_VARS = {
        "pid_score":    (('pid_scores','I1'), "Leading electron PID score",
                          np.linspace(0,1,41), _ns.THRESH_PID_SCORE, '>'),
        "primary_score":(('primary_scores','I1'), "Leading electron primary score",
                          np.linspace(0,1,41), _ns.THRESH_PRIMARY_SCORE, '>'),
        "vertex_dist":  (('vertex_distance',''), "Vertex distance [cm]",
                          np.linspace(0,20,41), _ns.THRESH_VERTEX_DIST, '<'),
        "calo_ke":      (('calo_ke',''), "Leading electron calo KE [MeV]",
                          np.linspace(0,2000,41), None, None),
    }

    def _get_var(evtdf_in, col_key, stage_idx, il):
        rp_df = reco_particles(evtdf_in)._df
        ke_col  = ('ke','') if ('ke','') in rp_df.columns else 'ke'
        pid_col = ('pid','') if ('pid','') in rp_df.columns else 'pid'

        ele = rp_df[rp_df.index.droplevel(-1).isin(stage_idx) &
                    (rp_df[pid_col] == 1)]

        if ele.empty or col_key not in ele.columns:
            return pd.Series(dtype=float)

        lead_idx = ele[ke_col].groupby(level=il).idxmax().dropna()
        if lead_idx.empty:
            return pd.Series(dtype=float)

        lead_mi = pd.MultiIndex.from_tuples(lead_idx.values,
                                            names=ele.index.names)
        vals = ele.loc[lead_mi, col_key]
        vals.index = lead_idx.index
        return vals

    def _by_cat(var_series, cat):
        mask = (truth_cat_at_topo == cat)
        common = var_series.index.intersection(mask.index)
        return var_series.loc[common][mask.loc[common]].dropna().values

    covs_ke = cov_results['reco_ke']
    total_var_ke = sum(np.diag(_mat(covs_ke[s]['cov_ms_ms'])).clip(0)
                       for s in covs_ke)
    cv_ke = cv_results['reco_ke']
    frac_cv_ke = np.where(cv_ke['sig_cv'] > 0,
                          np.sqrt(total_var_ke.clip(0))/cv_ke['sig_cv'],
                          0.0)

    ke_cov_centers = var_cfgs[0].bin_centers

    for var_name, var_spec in EXTRA_VARS.items():
        col_key, xlabel, bins = var_spec[:3]
        cut_val = var_spec[3]
        cut_dir = var_spec[4]

        mc_var = _get_var(mc_evtdf, col_key, mc_se_idx, mc_il)
        data_var = _get_var(data_evtdf, col_key, data_se_idx, data_il)

        dirt_var = pd.Series(dtype=float)
        if dirt_evtdf is not None and dirt_se_idx is not None:
            dirt_var = _get_var(dirt_evtdf, col_key, dirt_se_idx, dirt_il)

        print(f"{var_name}: MC={mc_var.notna().sum():,}  "
              f"Data={data_var.notna().sum():,}  "
              f"Dirt={dirt_var.notna().sum():,}")

        if mc_var.empty and data_var.empty and dirt_var.empty:
            continue

        bins_arr = np.asarray(bins)
        centers = 0.5*(bins_arr[:-1]+bins_arr[1:])

        frac_disp = np.interp(centers,
                              ke_cov_centers,
                              frac_cv_ke,
                              left=frac_cv_ke[0],
                              right=frac_cv_ke[-1])

        offbeam_vals = None
        if offbeam_shower_vars is not None:
            offbeam_vals = offbeam_shower_vars.get(var_name, None)

        fig = plot_unblinding_var(
            mc_vals=mc_var,
            data_vals=data_var.dropna().values,
            truth_cat=truth_cat_at_topo,
            bins=bins_arr,
            xlabel=xlabel,
            stage_name=stage_label,
            pot_scale_data=pot_scale_to_data,
            data_pot=data_total_pot,
            frac_unc_per_bin=frac_disp,
            offbeam_vals=offbeam_vals,
            offbeam_weight=offbeam_weight,
            dirt_vals=dirt_var.dropna().values,
            dirt_weight=dirt_weight,
            display_cats=display_cats,
            savefig_fn=savefig_fn,
            filename=f'{var_name}_topo_syst_data',
            cut_val=cut_val,
            cut_dir=cut_dir,
            save_subdir='extra_selection_vars',
        )

    print(f'Extra variable plots -> {plot_dir}/')


# ============================================================
# Unblinding / sideband helpers
# ============================================================

def _get_interaction_var(evtdf, col_key, stage_idx, il):
    ri_df = reco_interactions(evtdf)._df
    if col_key not in ri_df.columns:
        for c in ri_df.columns:
            if isinstance(c, tuple) and all(k in c for k in col_key if k):
                col_key = c; break
    return ri_df[col_key].groupby(level=il).first().reindex(stage_idx).dropna()


def _get_leading_electron_var(evtdf, col_key, stage_idx, il):
    rp_df   = reco_particles(evtdf)._df
    ke_col  = ('ke','')  if ('ke','')  in rp_df.columns else 'ke'
    pid_col = ('pid','') if ('pid','') in rp_df.columns else 'pid'
    ele = rp_df[rp_df.index.droplevel(-1).isin(stage_idx) & (rp_df[pid_col] == PID_ELECTRON)]
    if ele.empty or col_key not in ele.columns:
        return pd.Series(dtype=float)
    lead_idx = ele[ke_col].groupby(level=il).idxmax().dropna()
    if lead_idx.empty: return pd.Series(dtype=float)
    lead_mi = pd.MultiIndex.from_tuples(lead_idx.values, names=ele.index.names)
    vals = ele.loc[lead_mi, col_key]; vals.index = lead_idx.index
    return vals


def _by_cat_generic(var_series, truth_cat_series, cat):
    mask   = (truth_cat_series == cat)
    common = var_series.index.intersection(mask.index)
    return var_series.loc[common][mask.loc[common]].dropna().values


def plot_unblinding_var(mc_vals, data_vals, truth_cat, bins, xlabel,
                        stage_name, pot_scale_data, data_pot,
                        frac_unc_per_bin=None,
                        offbeam_vals=None, offbeam_weight=None,
                        dirt_vals=None, dirt_weight=None,
                        display_cats=None, savefig_fn=None, filename=None,
                        cut_val=None, cut_dir=None, save_subdir='unblinding'):
    """
    Stacked MC + offbeam + data + syst band + ratio panel + chi2 annotation.

    Parameters
    ----------
    offbeam_vals   : array-like, optional — offbeam reco values at this stage
    offbeam_weight : float, optional — offbeam scale factor
    frac_unc_per_bin : array, optional — fractional syst unc per bin
    """
    if display_cats is None:
        display_cats = [0, 1, 2, 3, 4, 5, 6]

    bins_arr = np.asarray(bins, dtype=float)
    n_bins   = len(bins_arr) - 1
    centers  = 0.5 * (bins_arr[:-1] + bins_arr[1:])

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # Build series/weights including offbeam
    mc_cats = display_cats
    series_list  = [_by_cat_generic(mc_vals, truth_cat, c) for c in mc_cats]
    weights_list = [np.full(len(s), pot_scale_data) for s in series_list]
    labels_plot  = [CAT_LABELS[c] for c in mc_cats]
    colors_plot  = [CAT_COLORS[c] for c in mc_cats]

    # Histograms for additional components
    ob_hist   = np.zeros(n_bins)
    dirt_hist = np.zeros(n_bins)

    # Add offbeam
    if offbeam_vals is not None and offbeam_weight is not None:
        ob_clean = np.asarray(offbeam_vals, dtype=float)
        ob_clean = ob_clean[~np.isnan(ob_clean)]

        if len(ob_clean) > 0:
            series_list.append(ob_clean)
            weights_list.append(
                np.full(len(ob_clean), offbeam_weight)
            )
            labels_plot.append(CAT_LABELS[7])
            colors_plot.append(CAT_COLORS[7])

            ob_hist, _ = np.histogram(
                ob_clean,
                bins=bins_arr,
                weights=np.full(len(ob_clean), offbeam_weight)
            )


    # Add lowE dirt ν
    if dirt_vals is not None and dirt_weight is not None:
        dirt_clean = np.asarray(dirt_vals, dtype=float)
        dirt_clean = dirt_clean[~np.isnan(dirt_clean)]

        if len(dirt_clean) > 0:
            series_list.append(dirt_clean)
            weights_list.append(
                np.full(len(dirt_clean), dirt_weight)
            )
            labels_plot.append(CAT_LABELS[9])
            colors_plot.append(CAT_COLORS[9])

            dirt_hist, _ = np.histogram(
                dirt_clean,
                bins=bins_arr,
                weights=np.full(len(dirt_clean), dirt_weight)
            )
            
    plot_stacked_hist(
        series_list=series_list, labels=labels_plot, colors=colors_plot,
        bins=bins_arr, weights=weights_list, xlabel='',
        title=fr'SBND $\nu_e$ CC Inclusive — {stage_name}',
        pot_label=f'Data POT: {data_pot:.2e}',
        ax=ax_main, invert_stack_order=True,
        show_counts=True, show_percentage=True,
        data_series=data_vals, data_label='On-beam data')

    # Total MC (MC + offbeam) per bin
    total_mc = np.zeros(n_bins)
    for cat in display_cats:
        v, _ = np.histogram(_by_cat_generic(mc_vals, truth_cat, cat), bins=bins_arr,
                            weights=np.full(len(_by_cat_generic(mc_vals, truth_cat, cat)),
                                            pot_scale_data))
        total_mc += v
    total_mc += ob_hist
    total_mc += dirt_hist

    # Syst band
    frac = None
    if frac_unc_per_bin is not None:
        frac = np.asarray(frac_unc_per_bin, dtype=float)
        if len(frac) != n_bins:
            frac = np.interp(centers, np.linspace(bins_arr[0], bins_arr[-1], len(frac)), frac)
        unc = frac * total_mc
        ax_main.bar(bins_arr[:-1], 2*unc, bottom=total_mc-unc,
                    width=np.diff(bins_arr), align='edge',
                    alpha=0.25, color='gray', hatch='///', label='Syst. unc.', linewidth=0)

    if cut_val is not None:
        sym = '>' if cut_dir == '>' else '<'
        ax_main.axvline(cut_val, color='red', ls='--', lw=1.5, label=f'Cut: {sym}{cut_val}')

    handles, lbls = ax_main.get_legend_handles_labels()
    ax_main.legend(handles, lbls, fontsize=7, ncol=2, loc='upper right',
                   frameon=True, framealpha=0.3, edgecolor='none')

    has_data = data_vals is not None and len(data_vals) > 0
    if has_data:
        dc, _ = np.histogram(np.asarray(data_vals, dtype=float), bins=bins_arr)
        dcf   = dc.astype(float)
        cov   = np.diag(np.where(dcf > 0, dcf, 1.0))       # data stat
        cov  += np.diag(ob_hist + dirt_hist)
        if frac is not None:
            cov += np.diag((frac * total_mc)**2)             # syst

        c2, nd, pv = chi2_pvalue(dcf, total_mc, cov_matrix=cov)
        ds, ms = float(dcf.sum()), float(total_mc.sum())
        dp  = ds/ms if ms > 0 else 0
        se  = np.sqrt(ds)/ms if ms > 0 else 0
        sye = (np.sqrt(np.sum((frac*total_mc)**2))/ms
               if frac is not None and ms > 0 else 0)

        ax_main.text(0.02, 0.75,
                     f'$\\Sigma$ Data/Pred = {dp:.2f} $\\pm$ {se:.2f} (stat.) '
                     f'$\\pm$ {sye:.2f} (syst.)\n'
                     f'$\\chi^2$/ndf = {c2:.1f}/{nd}, p = {pv:.2f}',
                     transform=ax_main.transAxes, fontsize=9, color='gray', va='top')

        print(f"  {stage_name} / {xlabel}:")
        print(f"    Data/Pred = {dp:.3f} ± {se:.3f} (stat.) ± {sye:.3f} (syst.)")
        print(f"    chi2/ndf = {c2:.1f}/{nd}  p = {pv:.3f}")

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(total_mc > 0, dcf/total_mc, np.nan)
            rerr  = np.where(total_mc > 0, np.sqrt(dcf)/total_mc, np.nan)
        ax_ratio.errorbar(centers, ratio, yerr=rerr, fmt='ko', ms=3, zorder=5)
        ax_ratio.axhline(1.0, color='gray', ls='--', lw=1)
        if frac is not None:
            ax_ratio.fill_between(centers, 1-frac, 1+frac,
                                  alpha=0.2, color='gray', step='mid')
        ax_ratio.set_ylim(0, 2)
        ax_ratio.set_ylabel('Data/MC', fontsize=11)
    else:
        ax_ratio.set_ylabel('Data/MC', fontsize=11)
        ax_ratio.text(0.5, 0.5, 'No data', transform=ax_ratio.transAxes,
                      ha='center', va='center', fontsize=12, color='gray')

    ax_ratio.set_xlabel(xlabel, fontsize=12)
    fig.tight_layout()
    if savefig_fn and filename:
        savefig_fn(fig, filename, save_subdir)
    return fig


# ============================================================
# Misc
# ============================================================

def save_plot(name, fig=None, folder_name="plots", dpi=150):
    import os
    os.makedirs(folder_name, exist_ok=True)
    if fig is None: fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(f"{folder_name}/{name}.png", dpi=dpi, bbox_inches="tight")