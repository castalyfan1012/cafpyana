"""
nue_plotter.py
--------------
All plotting functions for the nueCC inclusive SPINE analysis.
Extracted from nue_helpers.py + new Lynn-style comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from nue_helpers import (
    CAT_COLORS, CAT_LABELS, PID_ELECTRON,
    reco_particles, reco_interactions,
    inter_levels, chi2_pvalue, chi2_text, annotate_chi2,
    _get_leading_electron_var, _by_cat_generic,
    PID_NAMES, PID_COLORS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Save helper
# ══════════════════════════════════════════════════════════════════════════════

def save_plot(name, fig=None, folder_name="plots", dpi=150):
    import os
    os.makedirs(folder_name, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(f"{folder_name}/{name}.png", dpi=dpi, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# 2D resolution / bias plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_hist2d_frac_err(x, y, xlabel='x', ylabel='y', title=None,
                         cmap='Blues', plot_line=True,
                         normalize=True, use_errorbar=False,
                         fit_curve=True, show_fit=False,
                         log_color=False, fontsize=13, **pltkwargs):
    from scipy.optimize import curve_fit

    if 'bins' not in pltkwargs:
        raise ValueError('bins must be provided as a keyword argument')
    bins = np.asarray(pltkwargs.pop('bins'))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    def _crystal_ball(t, A, a, n, mu, sigma):
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        z = (t - mu) / sigma
        g = z > -a
        out[g] = np.exp(-0.5 * z[g] ** 2)
        Ac = (n / a) ** n * np.exp(-0.5 * a ** 2)
        B = n / a - a
        out[~g] = Ac * (B - z[~g]) ** (-n)
        return A * out

    bias = np.full(len(bin_centers), np.nan)
    err = np.full(len(bin_centers), np.nan)

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
        _std = max(float(np.nanstd(stat)), 1e-9)
        if fit_curve:
            _half = 5.0 * _std
            _fbins = np.linspace(_mean - _half, _mean + _half, 30)
            _h, _e = np.histogram(stat, bins=_fbins)
            _c = 0.5 * (_e[:-1] + _e[1:])
            try:
                A0 = max(float(_h.max()), 1.0)
                popt, _ = curve_fit(
                    _crystal_ball, _c, _h,
                    p0=[A0, 1.5, 2.0, _mean, _std],
                    bounds=([0, 0.1, 1.01, _mean - 3 * _std, 1e-9],
                            [20 * A0 + 1, 5.0, 50.0, _mean + 3 * _std, 5 * _std]),
                    maxfev=8000)
                bias[i] = popt[3]
                err[i] = popt[4]
            except Exception:
                bias[i] = _mean
                err[i] = _std
            if show_fit:
                _t = np.linspace(_mean - _half, _mean + _half, 200)
                plt.figure()
                plt.hist(stat, bins=_fbins, density=True, alpha=0.6,
                         label=f'Raw (n={len(stat):,})')
                _cb = _crystal_ball(_t, *popt)
                _cb /= (_cb.sum() * (_t[1] - _t[0]))
                plt.plot(_t, _cb, lw=2,
                         label=f'CB  μ={popt[3]:.3f}, σ={popt[4]:.3f}')
                plt.axvline(popt[3], ls='--', color='blue', label='μ')
                plt.axvline(popt[3] - popt[4], ls=':', color='green', label='μ±σ')
                plt.axvline(popt[3] + popt[4], ls=':', color='green')
                plt.title(f'Bin [{bins[i]:.3g}, {bins[i + 1]:.3g}]')
                plt.legend(fontsize=8)
                plt.show()
        else:
            bias[i] = _mean
            err[i] = _std

    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[20, 1],
                  hspace=0.03, wspace=0.15)
    ax_main = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[0, 1])
    plt.setp(ax_main.get_xticklabels(), visible=False)

    valid_mask = x.notna() & y.notna()
    _norm = LogNorm() if log_color else None
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
        ax2.scatter(bin_centers, err, color='green', s=40, label='Resolution (CB σ)',
                    marker='*', zorder=4)
        ax2.legend(fontsize=11, ncol=1, loc='upper right', framealpha=0.35)

    ax2.axhline(0, ls='--', color='black', lw=1)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_ylim(-0.05, 0.2)
    ax2.tick_params(axis='both', labelsize=fontsize - 2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.45)
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
        mask = true_series.index.isin(stage_idx)
        true_series = true_series[mask]
        reco_series = reco_series[mask]
    valid = true_series.notna() & reco_series.notna()
    return plot_hist2d_frac_err(
        true_series[valid].astype(float), reco_series[valid].astype(float),
        xlabel=xlabel, ylabel=ylabel, title=title,
        normalize=normalize, fit_curve=fit_curve,
        bins=np.asarray(bins), **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Threshold scan plots
# ══════════════════════════════════════════════════════════════════════════════

def threshold_plots(evtdf, cut_flow_result, cut_stage, metric_col,
                    bounds, xaxis_name, plot_title, threshold_range,
                    cut_dir='<', legend_loc='right', truth_categories=None,
                    fig_width=7, save_name=None, plot_folder="plots"):
    from nue_selection import PID_ELECTRON, classify_truth
    il = inter_levels(evtdf)
    presel_idx = cut_flow_result[cut_stage]["inter_index"]
    cat = truth_categories if truth_categories is not None else classify_truth(evtdf, verbose=False)
    rp_df = reco_particles(evtdf)._df

    _MULTI = {'pid_scores': ('pid_scores', 'I1'), 'primary_scores': ('primary_scores', 'I1')}
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

    ke_col = ('ke', '') if ('ke', '') in rp_df.columns else 'ke'
    pid_col = ('pid', '') if ('pid', '') in rp_df.columns else 'pid'

    in_presel = rp_df.index.droplevel(-1).isin(presel_idx)
    ele_presel = rp_df[in_presel & (rp_df[pid_col] == PID_ELECTRON)]
    tmp = pd.DataFrame({'ke': ele_presel[ke_col].to_numpy(dtype=float),
                        'metric': ele_presel[col_key].to_numpy(dtype=float)},
                       index=ele_presel.index)

    def metric_at_max_ke(g):
        return g.loc[g['ke'].idxmax(), 'metric']

    leading_metric = tmp.groupby(level=il).apply(metric_at_max_ke)
    inter_index = leading_metric.index
    metric_vals = leading_metric.to_numpy(dtype=float).flatten()
    is_signal = (cat.reindex(inter_index).fillna(-1).to_numpy().flatten() == 0)
    valid = ~np.isnan(metric_vals) & (metric_vals != -1)
    metric_vals = metric_vals[valid]
    is_signal = is_signal[valid]

    if len(metric_vals) == 0:
        print(f"No valid events for '{metric_col}'.")
        return None, None

    xmin, xmax = bounds
    bin_edges = np.linspace(xmin, xmax, 51)
    signal_vals = metric_vals[is_signal]
    bkg_vals = metric_vals[~is_signal]

    fig = plt.figure(figsize=(fig_width, 8))
    gs = fig.add_gridspec(4, 1, hspace=0.05)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.hist(bkg_vals, bins=bin_edges, histtype='step', density=True,
             label=f'Background ({len(bkg_vals):,})', color='orange', lw=1.5)
    ax1.hist(signal_vals, bins=bin_edges, histtype='step', density=True,
             label=r'Signal $\nu_e$ CC' + f' ({len(signal_vals):,})', color='blue', lw=1.5)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.set_title(plot_title, fontsize=16, pad=10)
    ax1.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.tick_params(labelbottom=False)

    start, stop, step = threshold_range
    thresholds = np.arange(start, stop + step / 2, step)
    if len(thresholds) <= 1:
        thresholds = np.linspace(start, stop, 120)
    total_signal = is_signal.sum()
    keep_above = (cut_dir == '>')

    purities, efficiencies, f1s = [], [], []
    for thresh in thresholds:
        passed = (metric_vals > thresh) if keep_above else (metric_vals < thresh)
        n_passed = passed.sum()
        n_sig_passed = (passed & is_signal).sum()
        pur = n_sig_passed / n_passed if n_passed > 0 else 0.0
        eff = n_sig_passed / total_signal if total_signal > 0 else 0.0
        f1 = 2 * pur * eff / (pur + eff) if (pur + eff) > 0 else 0.0
        purities.append(pur)
        efficiencies.append(eff)
        f1s.append(f1)

    f1s = np.array(f1s)
    opt_idx = np.argmax(f1s)
    opt_thresh = float(thresholds[opt_idx])
    max_f1 = float(f1s[opt_idx])

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(thresholds, purities, 'b-', label='Purity', lw=1.5)
    ax2.plot(thresholds, efficiencies, 'r-', label='Efficiency', lw=1.5)
    ax2.plot(thresholds, f1s, color='purple', label='F1 Score', lw=1.5)
    ax2.axvline(opt_thresh, color='purple', ls='--', lw=1.5)
    ax2.set_xlabel(xaxis_name, fontsize=14)
    ax2.set_ylabel('Performance', fontsize=14)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(0, 1)
    ax2.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)
    range_x = xmax - xmin
    text_x = opt_thresh + range_x * 0.05 if cut_dir == '<' else opt_thresh - range_x * 0.05
    ha = 'left' if cut_dir == '<' else 'right'
    ax2.text(text_x, 0.05, f'Max F1 at {opt_thresh:.3f}', color='purple',
             va='bottom', ha=ha, fontsize=13,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                       edgecolor='none', alpha=0.5))
    plt.tight_layout()
    if save_name is not None:
        save_plot(save_name, fig=fig, folder_name=plot_folder)
    plt.show()

    print(f"\n{'─' * 42}")
    print(f"  Variable  : {metric_col}")
    print(f"  Cut dir   : {cut_dir} {opt_thresh:.4f}")
    print(f"  Efficiency: {efficiencies[opt_idx] * 100:.1f}%")
    print(f"  Purity    : {purities[opt_idx] * 100:.1f}%")
    print(f"  Max F1    : {max_f1:.4f}")
    print(f"{'─' * 42}\n")
    return opt_thresh, max_f1


# ══════════════════════════════════════════════════════════════════════════════
# Efficiency / purity helpers
# ══════════════════════════════════════════════════════════════════════════════

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
    pur_pct = [p * 100 if (p is not None and not np.isnan(p)) else np.nan for p in purs]
    eff_pct = [e * 100 if (e is not None and not np.isnan(e)) else np.nan for e in effs]
    results = pd.DataFrame({
        "Cut": list(labels),
        "Purity [%]": [f"{p:.2f}" if not np.isnan(p) else "—" for p in pur_pct],
        "Efficiency [%]": [f"{e:.2f}" if not np.isnan(e) else "—" for e in eff_pct],
        "ΔEff [%]": ["—"] + [f"{eff_pct[i] - eff_pct[i - 1]:.2f}" for i in range(1, len(eff_pct))],
        "ΔPur [%]": ["—"] + [f"{pur_pct[i] - pur_pct[i - 1]:.2f}" for i in range(1, len(pur_pct))],
    })
    table = ax.table(cellText=results.values.tolist(),
                     colLabels=results.columns.tolist(),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10 if len(results) <= 10 else 9)
    n_rows = len(results) + 1
    row_scale = max(1.05, 18.0 / n_rows)
    table.scale(1.0, row_scale)
    col_widths = {0: None, 1: 0.13, 2: 0.15, 3: 0.12, 4: 0.12}
    for (row, col), cell in table.get_celld().items():
        w = col_widths.get(col)
        if w is not None:
            cell.set_width(w)
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf0fb")
            cell.set_edgecolor("#cccccc")
        else:
            cell.set_facecolor("#ffffff")
            cell.set_edgecolor("#cccccc")
        if row > 0 and col in (3, 4):
            cell.set_text_props(color="#555555", style="italic")
    return ax, table


def plot_matrix_binned(vals, count_matrix, bin_values, bin_labels, cut_labels,
                       remove_dummy_col=False, xlabel="", title="",
                       cmap="Greens", vmin=0, vmax=1,
                       label_fontsize=11, cell_size=0.8):
    if remove_dummy_col:
        vals = vals[:, :-1]
        count_matrix = count_matrix[:, :-1]
        cut_labels = cut_labels[:-1]
    n_bins, n_cuts = vals.shape
    fig, ax = plt.subplots(figsize=(cell_size * n_bins * 1.3, cell_size * n_cuts * 1.3))
    im = ax.imshow(vals.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticks(np.arange(n_cuts))
    ax.set_yticklabels(cut_labels, fontsize=label_fontsize)
    for ci in range(n_cuts):
        for bi in range(n_bins):
            v = vals[bi, ci]
            n = count_matrix[bi, ci]
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


# ══════════════════════════════════════════════════════════════════════════════
# PID fraction vs true KE
# ══════════════════════════════════════════════════════════════════════════════

def plot_pid_fraction_vs_ke(true_ke, reco_pid, ke_bins=None,
                             true_pid_target_name="True Electron (Primaries)",
                             watermark="SBND Work in Progress",
                             highlight_pid=None, title_fontsize=12,
                             figsize=(6, 7), ax1=None, ax2=None):
    _own = (ax1 is None) or (ax2 is None)
    if _own:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.06},
                                        sharex=True)
    else:
        fig = ax1.get_figure()

    if ke_bins is None:
        ke_bins = np.linspace(0, 200, 21)
    ke_bins = np.asarray(ke_bins, dtype=float)
    true_ke = np.asarray(true_ke, dtype=float)
    reco_pid = np.asarray(reco_pid, dtype=int)
    valid = ~np.isnan(true_ke) & (reco_pid >= 0)
    true_ke = true_ke[valid]
    reco_pid = reco_pid[valid]

    n_bins = len(ke_bins) - 1
    pid_ids = sorted(PID_NAMES.keys())
    counts = np.zeros((n_bins, len(pid_ids)), dtype=float)
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
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(ke_bins[0], ke_bins[-1])
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
    if _own:
        fig.tight_layout()
    return fig, (ax1, ax2)


# ══════════════════════════════════════════════════════════════════════════════
# Stacked histogram
# ══════════════════════════════════════════════════════════════════════════════

def _best_legend_side(series_list, weights_list, bins_arr):
    bins_arr = np.asarray(bins_arr, dtype=float)
    total = np.zeros(len(bins_arr) - 1)
    for s, w in zip(series_list, weights_list):
        s = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        if w is not None:
            h, _ = np.histogram(s[mask], bins=bins_arr,
                                weights=np.asarray(w, dtype=float)[mask])
        else:
            h, _ = np.histogram(s[mask], bins=bins_arr)
        total += h
    n = len(total)
    if n == 0:
        return 'right'
    mid = n // 2
    left_max = total[:mid].max() if mid > 0 else 0
    right_max = total[mid:].max() if mid < n else 0
    return 'right' if right_max <= left_max else 'left'


def plot_stacked_hist(series_list, labels, colors, bins, weights=None,
                      xlabel="", ylabel="Events / bin", title="",
                      density=False, ax=None, data_series=None,
                      data_label="Data", pot_label="",
                      invert_stack_order=False,
                      show_counts=True, show_percentage=True,
                      chi2_info=None, **hist_kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if weights is None:
        _weights = [None] * len(series_list)
    elif np.isscalar(weights):
        _weights = [np.full(len(s), float(weights)) for s in series_list]
    elif isinstance(weights, (list, tuple)):
        _weights = [None if w is None else np.asarray(w, dtype=float) for w in weights]
    else:
        raise TypeError("weights must be None, a scalar, or a list of arrays.")

    clean = []
    for s, w in zip(series_list, _weights):
        s = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        clean.append((s[mask], w[mask] if w is not None else None))

    bins_arr = np.asarray(bins)

    cat_totals = []
    for s, w in clean:
        h, _ = np.histogram(s, bins=bins_arr, weights=w)
        cat_totals.append(h.sum())
    grand_total = sum(cat_totals)

    annotated_labels = []
    for lbl, tot in zip(labels, cat_totals):
        parts = [lbl]
        if show_counts or show_percentage:
            inner = []
            if show_counts:
                inner.append(f"{tot:.1f}")
            if show_percentage and grand_total > 0:
                inner.append(f"{tot / grand_total:.1%}")
            parts.append(f"({', '.join(inner)})")
        annotated_labels.append(" ".join(parts))

    n_stack = len(clean)
    if invert_stack_order:
        x_list = [c[0] for c in reversed(clean)]
        w_list = [c[1] for c in reversed(clean)]
        plot_lbl = list(reversed(annotated_labels))
        col_list = list(reversed(colors))
    else:
        x_list = [c[0] for c in clean]
        w_list = [c[1] for c in clean]
        plot_lbl = annotated_labels
        col_list = list(colors)

    weights_arg = None if all(w is None for w in w_list) else w_list

    ax.hist(x_list, bins=bins_arr, weights=weights_arg,
            label=plot_lbl, color=col_list,
            stacked=True, histtype="stepfilled", density=density, **hist_kw)

    if data_series is not None:
        d = np.asarray(data_series, dtype=float)
        d = d[~np.isnan(d)]
        counts, _ = np.histogram(d, bins=bins_arr)
        centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
        ax.errorbar(centers, counts, yerr=np.sqrt(counts),
                    fmt="ko", markersize=4, label=data_label, zorder=10)

    handles, legend_labels = ax.get_legend_handles_labels()
    if invert_stack_order and n_stack > 1:
        stack_h = handles[:n_stack][::-1]
        stack_l = legend_labels[:n_stack][::-1]
        rest_h = handles[n_stack:]
        rest_l = legend_labels[n_stack:]
        handles = stack_h + rest_h
        legend_labels = stack_l + rest_l

    side = _best_legend_side([c[0] for c in clean], [c[1] for c in clean], bins_arr)
    ax.legend(handles, legend_labels, fontsize=9, ncol=1,
              loc=f'upper {side}', frameon=True, framealpha=0.3, edgecolor="none")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)

    info_side = 'right' if side == 'left' else 'left'
    info_ha = 'right' if info_side == 'right' else 'left'
    info_x = 0.98 if info_side == 'right' else 0.02

    info_lines = []
    if pot_label:
        info_lines.append(pot_label)
    info_lines.append(f"Total MC events: {grand_total:.0f}")
    ax.text(info_x, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            va="top", ha=info_ha, fontsize=10, color="gray", linespacing=1.15)

    if chi2_info is not None:
        annotate_chi2(ax, **chi2_info)

    return ax.get_figure(), ax


# ══════════════════════════════════════════════════════════════════════════════
# Unblinding plot (ratio panel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_unblinding_var(mc_vals, data_vals, truth_cat, bins, xlabel,
                        stage_name, pot_scale_data, data_pot,
                        frac_unc_per_bin=None,
                        offbeam_vals=None, offbeam_weight=None,
                        dirt_vals=None, dirt_weight=None,
                        display_cats=None, savefig_fn=None, filename=None,
                        cut_val=None, cut_dir=None, save_subdir='unblinding'):
    if display_cats is None:
        display_cats = [0, 1, 2, 3, 4, 5, 6]

    bins_arr = np.asarray(bins, dtype=float)
    n_bins = len(bins_arr) - 1
    centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])

    ob_hist = np.zeros(n_bins)
    dirt_hist = np.zeros(n_bins)

    mc_hists, mc_labels, mc_colors, mc_totals = [], [], [], []
    for cat in display_cats:
        vals = _by_cat_generic(mc_vals, truth_cat, cat)
        h, _ = np.histogram(vals, bins=bins_arr,
                            weights=np.full(len(vals), pot_scale_data))
        mc_hists.append(h)
        mc_labels.append(CAT_LABELS[cat])
        mc_colors.append(CAT_COLORS[cat])
        mc_totals.append(h.sum())

    dirt_total, ob_total = 0.0, 0.0
    has_dirt, has_ob = False, False

    if dirt_vals is not None and dirt_weight is not None:
        dc = np.asarray(dirt_vals, dtype=float)
        dc = dc[~np.isnan(dc)]
        if len(dc) > 0:
            dirt_hist, _ = np.histogram(dc, bins=bins_arr,
                                        weights=np.full(len(dc), dirt_weight))
            dirt_total = dirt_hist.sum()
            has_dirt = True

    if offbeam_vals is not None and offbeam_weight is not None:
        oc = np.asarray(offbeam_vals, dtype=float)
        oc = oc[~np.isnan(oc)]
        if len(oc) > 0:
            ob_hist, _ = np.histogram(oc, bins=bins_arr,
                                      weights=np.full(len(oc), offbeam_weight))
            ob_total = ob_hist.sum()
            has_ob = True

    total_mc = sum(mc_hists) + dirt_hist + ob_hist
    grand_total = total_mc.sum()

    mid = n_bins // 2
    left_max = total_mc[:mid].max() if mid > 0 else 0
    right_max = total_mc[mid:].max() if mid < n_bins else 0
    side = 'right' if right_max <= left_max else 'left'

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 7.5),
        gridspec_kw={'height_ratios': [3.2, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    bw = np.diff(bins_arr)
    bottoms = np.zeros(n_bins)

    for h, col in zip(reversed(mc_hists), reversed(mc_colors)):
        ax_main.bar(bins_arr[:-1], h, width=bw, bottom=bottoms,
                    align='edge', color=col, edgecolor='none', linewidth=0)
        bottoms += h

    if has_dirt:
        ax_main.bar(bins_arr[:-1], dirt_hist, width=bw, bottom=bottoms,
                    align='edge', color=CAT_COLORS[9], edgecolor='none', linewidth=0)
        bottoms += dirt_hist

    if has_ob:
        ax_main.bar(bins_arr[:-1], ob_hist, width=bw, bottom=bottoms,
                    align='edge', color=CAT_COLORS[7], edgecolor='none', linewidth=0)
        bottoms += ob_hist

    frac = None
    if frac_unc_per_bin is not None:
        frac = np.asarray(frac_unc_per_bin, dtype=float)
        if len(frac) != n_bins:
            frac = np.interp(centers,
                             np.linspace(bins_arr[0], bins_arr[-1], len(frac)), frac)
        unc = frac * total_mc
        ax_main.bar(bins_arr[:-1], 2 * unc, bottom=total_mc - unc,
                    width=bw, align='edge', alpha=0.25, color='gray',
                    hatch='///', linewidth=0, zorder=5)

    if cut_val is not None:
        sym = '>' if cut_dir == '>' else '<'
        ax_main.axvline(cut_val, color='red', ls='--', lw=1.5)

    has_data = data_vals is not None and len(data_vals) > 0
    dc = None
    if has_data:
        d = np.asarray(data_vals, dtype=float)
        d = d[~np.isnan(d)]
        dc, _ = np.histogram(d, bins=bins_arr)
        ax_main.errorbar(centers, dc, yerr=np.sqrt(dc.clip(1)),
                         fmt="ko", markersize=4, zorder=10)

    # Legend
    legend_handles, legend_labels = [], []
    for lbl, col, tot in zip(mc_labels, mc_colors, mc_totals):
        pct = f"{tot / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=col, edgecolor='none'))
        legend_labels.append(f"{lbl} ({tot:.1f}, {pct})")

    if has_dirt:
        pct = f"{dirt_total / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=CAT_COLORS[9], edgecolor='none'))
        legend_labels.append(f"{CAT_LABELS[9]} ({dirt_total:.1f}, {pct})")
    if has_ob:
        pct = f"{ob_total / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=CAT_COLORS[7], edgecolor='none'))
        legend_labels.append(f"{CAT_LABELS[7]} ({ob_total:.1f}, {pct})")
    if has_data:
        legend_handles.append(mlines.Line2D([], [], color='black', marker='o',
                                            linestyle='None', markersize=4))
        legend_labels.append('On-beam data')
    if frac is not None:
        legend_handles.append(mpatches.Patch(facecolor='gray', edgecolor='black',
                                             alpha=0.25, hatch='///', linewidth=0))
        legend_labels.append('Syst. unc.')
    if cut_val is not None:
        legend_handles.append(mlines.Line2D([], [], color='red', ls='--', lw=1.5))
        legend_labels.append(f'Cut: {sym}{cut_val}')

    ax_main.legend(legend_handles, legend_labels, fontsize=7, ncol=2,
                   loc=f'upper {side}', frameon=True, framealpha=0.85, edgecolor='none')

    ymax = max(total_mc.max(), dc.max() if dc is not None else 0)
    ax_main.set_ylim(bottom=0, top=ymax * 1.55)
    ax_main.set_ylabel('Events / bin', fontsize=12)
    ax_main.set_title(fr'SBND $\nu_e$ CC Inclusive — {stage_name}', fontsize=12)

    info_side = 'right' if side == 'left' else 'left'
    info_ha = 'right' if info_side == 'right' else 'left'
    info_x = 0.98 if info_side == 'right' else 0.02

    header = [f'Data POT: {data_pot:.2e}', f'Total MC events: {grand_total:.0f}']
    ax_main.text(info_x, 0.98, "\n".join(header),
                 transform=ax_main.transAxes, fontsize=9, color='gray',
                 va='top', ha=info_ha, linespacing=1.15)

    if has_data:
        dcf = dc.astype(float)
        cov = np.diag(np.where(dcf > 0, dcf, 1.0))
        cov += np.diag(ob_hist + dirt_hist)
        if frac is not None:
            cov += np.diag((frac * total_mc) ** 2)
        c2, nd, pv = chi2_pvalue(dcf, total_mc, cov_matrix=cov)
        ds, ms = float(dcf.sum()), float(total_mc.sum())
        dp = ds / ms if ms > 0 else 0
        se = np.sqrt(ds) / ms if ms > 0 else 0
        sye = (np.sqrt(np.sum((frac * total_mc) ** 2)) / ms
               if frac is not None and ms > 0 else 0)
        stats_text = (f'$\\Sigma$ Data/Pred = {dp:.2f} $\\pm$ {se:.2f} (stat.) '
                      f'$\\pm$ {sye:.2f} (syst.)\n'
                      f'$\\chi^2$/ndf = {c2:.1f}/{nd}, p = {pv:.2f}')
        ax_ratio.text(info_x, 0.98, stats_text,
                      transform=ax_ratio.transAxes, fontsize=9, color='gray',
                      va='top', ha=info_ha, linespacing=1.15)
        print(f"  {stage_name} / {xlabel}:")
        print(f"    Data/Pred = {dp:.3f} ± {se:.3f} (stat.) ± {sye:.3f} (syst.)")
        print(f"    chi2/ndf = {c2:.1f}/{nd}  p = {pv:.3f}")

    # Ratio panel
    if has_data:
        dcf = dc.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(total_mc > 0, dcf / total_mc, np.nan)
            rerr = np.where(total_mc > 0, np.sqrt(dcf) / total_mc, np.nan)
        ax_ratio.errorbar(centers, ratio, yerr=rerr, fmt='ko', ms=3, zorder=5)
        ax_ratio.axhline(1.0, color='gray', ls='--', lw=1)
        if frac is not None:
            ax_ratio.fill_between(centers, 1 - frac, 1 + frac,
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


# ══════════════════════════════════════════════════════════════════════════════
# Differential slice plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_differential_slices(
        sel_df, stage_col,
        slice_var, plot_var,
        slice_edges, plot_bins,
        pot_scale,
        data_reco_dict,
        offbeam_reco_dict, offbeam_weight,
        dirt_reco_dict, dirt_weight,
        frac_unc_fn,
        title, xlabel,
        display_cats=None,
        savefig_fn=None, filename=None,
        ncols=3):
    if display_cats is None:
        display_cats = [0, 1, 2, 3, 4, 5, 6]

    full_cats = list(display_cats) + [9, 7]
    n_slices = len(slice_edges)
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    plot_bins = np.asarray(plot_bins, dtype=float)
    n_bins = len(plot_bins) - 1
    proxy = 'reco_costheta' if 'cos' in plot_var else 'reco_ke'
    mc_final = sel_df[sel_df[stage_col]].copy()

    for si, (s_lo, s_hi) in enumerate(slice_edges):
        ax = axes_flat[si]
        mc_slice = mc_final[(mc_final[slice_var] >= s_lo) & (mc_final[slice_var] < s_hi)]

        stack_series, stack_weights, stack_labels, stack_colors = [], [], [], []
        for cat in reversed(display_cats):
            vals = mc_slice.loc[mc_slice['truth_cat'] == cat, plot_var].dropna().values
            stack_series.append(vals)
            stack_weights.append(np.full(len(vals), pot_scale))
            stack_labels.append(CAT_LABELS[cat])
            stack_colors.append(CAT_COLORS[cat])

        sv = np.asarray(dirt_reco_dict.get(slice_var, []), dtype=float)
        pv = np.asarray(dirt_reco_dict.get(plot_var, []), dtype=float)
        valid = ~np.isnan(sv) & ~np.isnan(pv)
        mask = valid & (sv >= s_lo) & (sv < s_hi)
        stack_series.append(pv[mask])
        stack_weights.append(np.full(mask.sum(), dirt_weight))
        stack_labels.append(CAT_LABELS[9])
        stack_colors.append(CAT_COLORS[9])

        sv = np.asarray(offbeam_reco_dict.get(slice_var, []), dtype=float)
        pv = np.asarray(offbeam_reco_dict.get(plot_var, []), dtype=float)
        valid = ~np.isnan(sv) & ~np.isnan(pv)
        mask = valid & (sv >= s_lo) & (sv < s_hi)
        stack_series.append(pv[mask])
        stack_weights.append(np.full(mask.sum(), offbeam_weight))
        stack_labels.append(CAT_LABELS[7])
        stack_colors.append(CAT_COLORS[7])

        n_stack = len(stack_series)
        n_mc = len(display_cats)

        clean_s, clean_w = [], []
        for s, w in zip(stack_series, stack_weights):
            s = np.asarray(s, dtype=float)
            m = ~np.isnan(s)
            clean_s.append(s[m])
            clean_w.append(np.asarray(w, dtype=float)[m])

        total_sl = np.zeros(n_bins)
        for s, w in zip(clean_s, clean_w):
            h, _ = np.histogram(s, bins=plot_bins, weights=w)
            total_sl += h

        frac_sl = frac_unc_fn(plot_bins, proxy)
        unc_sl = frac_sl * total_sl

        ax.hist(clean_s, bins=plot_bins, weights=clean_w,
                label=stack_labels, color=stack_colors,
                stacked=True, histtype="stepfilled")

        bw = np.diff(plot_bins)
        ax.bar(plot_bins[:-1], 2 * unc_sl, bottom=total_sl - unc_sl,
               width=bw, align='edge', alpha=0.25, color='gray',
               hatch='///', linewidth=0)

        data_sv = np.asarray(data_reco_dict.get(slice_var, []), dtype=float)
        data_pv = np.asarray(data_reco_dict.get(plot_var, []), dtype=float)
        valid_d = ~np.isnan(data_sv) & ~np.isnan(data_pv)
        d_mask = valid_d & (data_sv >= s_lo) & (data_sv < s_hi)
        data_sl = data_pv[d_mask]
        if len(data_sl) > 0:
            dch, _ = np.histogram(data_sl, bins=plot_bins)
            ctrs = 0.5 * (plot_bins[:-1] + plot_bins[1:])
            ax.errorbar(ctrs, dch, yerr=np.sqrt(dch.clip(1)),
                        fmt='ko', ms=4, zorder=10)

        if slice_var == 'reco_ke':
            ax.set_title(f'{int(s_lo)}–{int(s_hi)} MeV', fontsize=11)
        else:
            ax.set_title(f'{s_lo:.2g} < cos θ < {s_hi:.2g}', fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(plot_bins[0], plot_bins[-1])

        if si == 0:
            handles_raw, labels_raw = ax.get_legend_handles_labels()
            mc_h = handles_raw[:n_mc][::-1]
            mc_l = labels_raw[:n_mc][::-1]
            ext_h = handles_raw[n_mc:n_stack]
            ext_l = labels_raw[n_mc:n_stack]
            rest_h = handles_raw[n_stack:]
            rest_l = labels_raw[n_stack:]
            ax.legend(mc_h + ext_h + rest_h, mc_l + ext_l + rest_l,
                      fontsize=7, ncol=2, loc='best',
                      frameon=True, framealpha=0.8, edgecolor='none')
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    for j in range(si + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if savefig_fn and filename:
        savefig_fn(fig, filename)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# Extra selection variable plots
# ══════════════════════════════════════════════════════════════════════════════

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
        display_cats = [0, 1, 2, 3, 4, 5, 6]

    def _mat(block):
        return block.get('cov', block) if isinstance(block, dict) else block

    EXTRA_VARS = {
        "pid_score": (('pid_scores', 'I1'), "Leading electron PID score",
                      np.linspace(0, 1, 41), _ns.THRESH_PID_SCORE, '>'),
        "primary_score": (('primary_scores', 'I1'), "Leading electron primary score",
                          np.linspace(0, 1, 41), _ns.THRESH_PRIMARY_SCORE, '>'),
        "vertex_dist": (('vertex_distance', ''), "Vertex distance [cm]",
                        np.linspace(0, 20, 41), _ns.THRESH_VERTEX_DIST, '<'),
        "calo_ke": (('calo_ke', ''), "Leading electron calo KE [MeV]",
                    np.linspace(0, 2000, 41), None, None),
    }

    def _get_var(evtdf_in, col_key, stage_idx, il):
        rp_df = reco_particles(evtdf_in)._df
        ke_col = ('ke', '') if ('ke', '') in rp_df.columns else 'ke'
        pid_col = ('pid', '') if ('pid', '') in rp_df.columns else 'pid'
        ele = rp_df[rp_df.index.droplevel(-1).isin(stage_idx) & (rp_df[pid_col] == 1)]
        if ele.empty or col_key not in ele.columns:
            return pd.Series(dtype=float)
        lead_idx = ele[ke_col].groupby(level=il).idxmax().dropna()
        if lead_idx.empty:
            return pd.Series(dtype=float)
        lead_mi = pd.MultiIndex.from_tuples(lead_idx.values, names=ele.index.names)
        vals = ele.loc[lead_mi, col_key]
        vals.index = lead_idx.index
        return vals

    covs_ke = cov_results['reco_ke']
    total_var_ke = sum(np.diag(_mat(covs_ke[s]['cov_ms_ms'])).clip(0) for s in covs_ke)
    cv_ke = cv_results['reco_ke']
    frac_cv_ke = np.where(cv_ke['sig_cv'] > 0,
                          np.sqrt(total_var_ke.clip(0)) / cv_ke['sig_cv'], 0.0)
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
              f"Data={data_var.notna().sum():,}  Dirt={dirt_var.notna().sum():,}")
        if mc_var.empty and data_var.empty and dirt_var.empty:
            continue

        bins_arr = np.asarray(bins)
        centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
        frac_disp = np.interp(centers, ke_cov_centers, frac_cv_ke,
                              left=frac_cv_ke[0], right=frac_cv_ke[-1])

        offbeam_vals = None
        if offbeam_shower_vars is not None:
            offbeam_vals = offbeam_shower_vars.get(var_name, None)

        plot_unblinding_var(
            mc_vals=mc_var, data_vals=data_var.dropna().values,
            truth_cat=truth_cat_at_topo, bins=bins_arr, xlabel=xlabel,
            stage_name=stage_label, pot_scale_data=pot_scale_to_data,
            data_pot=data_total_pot, frac_unc_per_bin=frac_disp,
            offbeam_vals=offbeam_vals, offbeam_weight=offbeam_weight,
            dirt_vals=dirt_var.dropna().values, dirt_weight=dirt_weight,
            display_cats=display_cats, savefig_fn=savefig_fn,
            filename=f'{var_name}_topo_syst_data',
            cut_val=cut_val, cut_dir=cut_dir,
            save_subdir='extra_selection_vars')

    print(f'Extra variable plots -> {plot_dir}/')


# ══════════════════════════════════════════════════════════════════════════════
# Covariance heatmap
# ══════════════════════════════════════════════════════════════════════════════

def format_heatmap_value(v):
    if np.isnan(v):
        return ""
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1:
        return f"{v:.4f}"
    elif av < 1e3:
        return f"{v:.2f}"
    mantissa, exponent = f"{v:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def plot_heatmap_custom(matrix, title, vcfg, axis_labels=None,
                        cmap='viridis', vmin=None, vmax=None):
    import re
    try:
        from analysis_village.unfolding.utils import bin_range_labels, get_text_color
        raw = bin_range_labels(vcfg.bins)
        t_lbls = [re.sub(r'(\d+)\.0+(?!\d)', r'\1', l) for l in raw]
        _get_tc = get_text_color
    except Exception:
        t_lbls = [f'{vcfg.bins[i]:.0f}–{vcfg.bins[i + 1]:.0f}'
                  for i in range(len(vcfg.bins) - 1)]
        _get_tc = lambda v: 'white'

    n = len(vcfg.bins) - 1
    unif = np.arange(n + 1, dtype=float)
    tick_pos = 0.5 * (unif[:-1] + unif[1:])
    extent = [unif[0], unif[-1], unif[0], unif[-1]]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    im = ax.imshow(matrix, extent=extent, origin='lower',
                   cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(t_lbls, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(t_lbls, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    xlbl, ylbl = axis_labels if axis_labels else [vcfg.var_plot_name] * 2
    ax.set_xlabel(xlbl, fontsize=11)
    ax.set_ylabel(ylbl, fontsize=11)
    ax.set_title(title, fontsize=12)

    fsize = max(5, 9 - n // 3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                try:
                    tc = _get_tc(v)
                except Exception:
                    tc = 'white'
                ax.text(tick_pos[j], tick_pos[i],
                        format_heatmap_value(v),
                        ha='center', va='center', color=tc, fontsize=fsize)
    fig.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# Fractional uncertainty per bin
# ══════════════════════════════════════════════════════════════════════════════

SRC_LABEL = {
    'flux': 'BNB Flux', 'genie': 'GENIE', 'extra_xsec': 'Extra Xsec',
    'g4': 'Geant4', 'mcstat': 'MCstat', 'pot': 'POT',
    'ntargets': r'$N_\mathrm{targets}$', 'detsys': 'Detector',
}
COLORS = {
    'flux': '#2196F3', 'genie': '#F44336', 'extra_xsec': '#FF9800',
    'g4': '#00BCD4', 'mcstat': '#4CAF50', 'pot': '#9C27B0',
    'ntargets': '#795548', 'detsys': '#8B0000',
}


def _mat(block):
    return block.get('cov', block) if isinstance(block, dict) else block


def plot_fracunc(vcfg, cv_arr, cov_results_var, block_key, categ,
                 title=r'SBND $\nu_e$ CC Inclusive'):
    import warnings
    fig, ax = plt.subplots(figsize=(8, 4))
    total_var = np.zeros(len(vcfg.bin_centers))
    for src, covs in cov_results_var.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            diag = np.diag(_mat(covs[block_key])).clip(0)
            frac = np.where(cv_arr > 0, np.sqrt(diag) / cv_arr, 0.0)
        xe = np.append(vcfg.bins[:-1], vcfg.bins[-1])
        ye = np.append(frac, frac[-1])
        ax.step(xe, ye, where='post', label=SRC_LABEL.get(src, src),
                color=COLORS.get(src, 'gray'), lw=1.8)
        total_var += diag
    tot = np.where(cv_arr > 0, np.sqrt(total_var) / cv_arr, 0.0)
    ax.step(np.append(vcfg.bins[:-1], vcfg.bins[-1]),
            np.append(tot, tot[-1]),
            where='post', label='Total', color='black', lw=2.2, ls='--')
    ax.set_xlabel(vcfg.var_plot_name, fontsize=12)
    ax.set_ylabel('Fractional uncertainty', fontsize=12)
    ax.set_title(f'{title} - {categ}', fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xlim(vcfg.bins[0], vcfg.bins[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.99, 0.97, vcfg.pot_label, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='gray')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW: Lynn-style variable-width binning comparison plots
# ══════════════════════════════════════════════════════════════════════════════

# Lynn's binning (first/last bins are underflow/overflow)
LYNN_KE_BINS_GEV = np.array([0.5, 0.7, 0.95, 1.25, 1.7, 2.5])   # GeV
LYNN_KE_BINS_MEV = LYNN_KE_BINS_GEV * 1000.0                      # MeV
LYNN_COS_BINS = np.array([-1.0, 0.6, 0.75, 0.85, 0.925, 1.0])


def plot_lynn_comparison(sel_topo, stage_col, var_name,
                         pot_scale_data, data_pot,
                         data_reco, offbeam_reco, offbeam_weight,
                         dirt_reco, dirt_weight,
                         frac_unc_per_bin=None,
                         display_cats=None,
                         savefig_fn=None, filename=None,
                         save_subdir='lynn_comparison',
                         watermark='SBND Analysis In Progress'):
    """
    Produce a stacked histogram + ratio panel using Lynn's variable-width
    binning. Plots events / bin (NOT per-unit-width) to match her style.

    First and last bins capture underflow/overflow respectively.

    Parameters
    ----------
    var_name : 'reco_ke' or 'reco_costheta'
    """
    if display_cats is None:
        display_cats = [0, 1, 2, 3, 4, 5, 6]

    # Pick the right binning
    if var_name == 'reco_ke':
        bins_arr = LYNN_KE_BINS_MEV.copy()
        xlabel = r'Electron Energy $E_{e^-}$ (GeV)'
        # We'll label the x-axis in GeV for comparison with Lynn
        x_scale = 1e-3  # MeV → GeV for display
    elif var_name == 'reco_costheta':
        bins_arr = LYNN_COS_BINS.copy()
        xlabel = r'Electron Direction, $\cos\theta$'
        x_scale = 1.0
    else:
        raise ValueError(f"var_name must be 'reco_ke' or 'reco_costheta', got {var_name}")

    n_bins = len(bins_arr) - 1
    disp_edges = bins_arr * x_scale   # display edges (GeV or unitless)
    centers = 0.5 * (disp_edges[:-1] + disp_edges[1:])

    # ── Clip values into [first edge, last edge] for underflow/overflow ───
    def _clip(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return np.clip(v, bins_arr[0], bins_arr[-1] - 1e-8)

    # ── Histogram each component ──────────────────────────────────────────
    mc_hists, mc_labels, mc_colors, mc_totals = [], [], [], []
    truth_cat = sel_topo.loc[sel_topo[stage_col], 'truth_cat']

    for cat in display_cats:
        m = sel_topo[stage_col] & (sel_topo['truth_cat'] == cat)
        vals = _clip(sel_topo.loc[m, var_name].dropna().values)
        h, _ = np.histogram(vals, bins=bins_arr,
                            weights=np.full(len(vals), pot_scale_data))
        mc_hists.append(h)
        mc_labels.append(CAT_LABELS[cat])
        mc_colors.append(CAT_COLORS[cat])
        mc_totals.append(h.sum())

    dirt_hist = np.zeros(n_bins)
    ob_hist = np.zeros(n_bins)
    dirt_total, ob_total = 0.0, 0.0
    has_dirt, has_ob = False, False

    if dirt_reco is not None and dirt_weight is not None:
        dv = _clip(dirt_reco.get(var_name, []))
        if len(dv) > 0:
            dirt_hist, _ = np.histogram(dv, bins=bins_arr,
                                        weights=np.full(len(dv), dirt_weight))
            dirt_total = dirt_hist.sum()
            has_dirt = True

    if offbeam_reco is not None and offbeam_weight is not None:
        ov = _clip(offbeam_reco.get(var_name, []))
        if len(ov) > 0:
            ob_hist, _ = np.histogram(ov, bins=bins_arr,
                                      weights=np.full(len(ov), offbeam_weight))
            ob_total = ob_hist.sum()
            has_ob = True

    total_mc = sum(mc_hists) + dirt_hist + ob_hist
    grand_total = total_mc.sum()

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={'height_ratios': [3.2, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    bw = np.diff(disp_edges)
    bottoms = np.zeros(n_bins)

    # Stack: cosmic at bottom → signal on top
    for h, col in zip(reversed(mc_hists), reversed(mc_colors)):
        ax_main.bar(disp_edges[:-1], h, width=bw, bottom=bottoms,
                    align='edge', color=col, edgecolor='none', linewidth=0)
        bottoms += h

    if has_dirt:
        ax_main.bar(disp_edges[:-1], dirt_hist, width=bw, bottom=bottoms,
                    align='edge', color=CAT_COLORS[9], edgecolor='none', linewidth=0)
        bottoms += dirt_hist

    if has_ob:
        ax_main.bar(disp_edges[:-1], ob_hist, width=bw, bottom=bottoms,
                    align='edge', color=CAT_COLORS[7], edgecolor='none', linewidth=0)
        bottoms += ob_hist

    # ── Syst band (hatched MC stat+syst) ──────────────────────────────────
    frac = None
    if frac_unc_per_bin is not None:
        frac = np.asarray(frac_unc_per_bin, dtype=float)
        if len(frac) != n_bins:
            frac = np.interp(centers,
                             np.linspace(disp_edges[0], disp_edges[-1], len(frac)),
                             frac)
        unc = frac * total_mc
        ax_main.bar(disp_edges[:-1], 2 * unc, bottom=total_mc - unc,
                    width=bw, align='edge', alpha=0.3, color='gray',
                    hatch='xxxx', linewidth=0, zorder=5,
                    label='MC stat.+syst.')

    # ── Data ──────────────────────────────────────────────────────────────
    has_data = data_reco is not None and len(data_reco.get(var_name, [])) > 0
    dc = None
    n_data = 0
    if has_data:
        dv = _clip(data_reco[var_name])
        dc, _ = np.histogram(dv, bins=bins_arr)
        n_data = int(dc.sum())
        ax_main.errorbar(centers, dc, yerr=np.sqrt(dc.clip(1)),
                         fmt='ko', markersize=5, zorder=10,
                         label=f'data ({n_data})')

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles, legend_labels_list = [], []

    # Data first (matches Lynn's style)
    if has_data:
        legend_handles.append(mlines.Line2D([], [], color='black', marker='o',
                                            linestyle='None', markersize=5))
        legend_labels_list.append(f'data ({n_data})')

    # MC categories
    for lbl, col, tot in zip(mc_labels, mc_colors, mc_totals):
        pct = f"{tot / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=col, edgecolor='none'))
        legend_labels_list.append(f"{lbl} ({pct})")

    if has_dirt:
        pct = f"{dirt_total / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=CAT_COLORS[9], edgecolor='none'))
        legend_labels_list.append(f"{CAT_LABELS[9]} ({pct})")
    if has_ob:
        pct = f"{ob_total / grand_total:.1%}" if grand_total > 0 else "0%"
        legend_handles.append(mpatches.Patch(facecolor=CAT_COLORS[7], edgecolor='none'))
        legend_labels_list.append(f"{CAT_LABELS[7]} ({pct})")
    if frac is not None:
        legend_handles.append(mpatches.Patch(facecolor='gray', edgecolor='black',
                                             alpha=0.3, hatch='xxxx', linewidth=0))
        legend_labels_list.append('MC stat.+syst.')

    ax_main.legend(legend_handles, legend_labels_list, fontsize=7, ncol=2,
                   loc='upper left', frameon=True, framealpha=0.85, edgecolor='none')

    # ── Annotation ────────────────────────────────────────────────────────
    if has_data and frac is not None:
        dcf = dc.astype(float)
        ds, ms = float(dcf.sum()), float(total_mc.sum())
        dp = ds / ms if ms > 0 else 0
        se = np.sqrt(ds) / ms if ms > 0 else 0
        sye = np.sqrt(np.sum((frac * total_mc) ** 2)) / ms if ms > 0 else 0

        cov = np.diag(np.where(dcf > 0, dcf, 1.0))
        cov += np.diag(ob_hist + dirt_hist)
        cov += np.diag((frac * total_mc) ** 2)
        c2, nd, pv = chi2_pvalue(dcf, total_mc, cov_matrix=cov)

        stats = (f'Σ Data/Pred = {dp:.2f} ± {se:.2f} (stat.) ± {sye:.2f} (syst.)\n'
                 f'$\\chi^2$/ndf = {c2:.1f}/{nd}, p = {pv:.2f}')
        ax_main.text(0.98, 0.55, stats, transform=ax_main.transAxes,
                     fontsize=9, color='gray', va='top', ha='right', linespacing=1.3)

    ymax = max(total_mc.max(), dc.max() if dc is not None else 0)
    ax_main.set_ylim(bottom=0, top=ymax * 1.6)
    ax_main.set_ylabel('Events', fontsize=12)

    # Header
    pot_str = f'{data_pot:.2e} POT' if data_pot else ''
    ax_main.text(0.01, 1.02, watermark, transform=ax_main.transAxes,
                 fontsize=10, va='bottom', ha='left', fontweight='bold')
    ax_main.text(0.99, 1.02, pot_str, transform=ax_main.transAxes,
                 fontsize=10, va='bottom', ha='right')

    # ── Tick labels at bin edges ──────────────────────────────────────────
    ax_ratio.set_xticks(disp_edges)
    ax_ratio.set_xticklabels([f'{e:.3g}' if abs(e) < 10 else f'{e:.0f}'
                              for e in disp_edges], fontsize=9)

    # ── Ratio panel ───────────────────────────────────────────────────────
    if has_data:
        dcf = dc.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(total_mc > 0, dcf / total_mc, np.nan)
            rerr = np.where(total_mc > 0, np.sqrt(dcf) / total_mc, np.nan)
        ax_ratio.errorbar(centers, ratio, yerr=rerr, fmt='ko', ms=3, zorder=5)
        ax_ratio.axhline(1.0, color='gray', ls='--', lw=1)
        if frac is not None:
            # Draw syst band on ratio panel
            for i in range(n_bins):
                ax_ratio.fill_between(
                    [disp_edges[i], disp_edges[i + 1]],
                    1 - frac[i], 1 + frac[i],
                    alpha=0.15, color='red', linewidth=0)
        ax_ratio.set_ylim(0, 2)
    ax_ratio.set_ylabel('Data/Pred', fontsize=11)
    ax_ratio.set_xlabel(xlabel, fontsize=12)
    ax_ratio.set_xlim(disp_edges[0], disp_edges[-1])

    fig.tight_layout()
    if savefig_fn and filename:
        savefig_fn(fig, filename, save_subdir)
    return fig