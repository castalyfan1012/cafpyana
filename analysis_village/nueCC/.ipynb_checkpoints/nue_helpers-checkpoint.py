"""
nue_helpers.py (UPDATED)
--------------
Plotting and utility helpers for the nueCC inclusive SPINE analysis.

Key changes in this version:
- Fixed `plot_stacked_hist` bug that caused "weights should have the same shape as x"
  when some categories have zero events (common after final selection).
  Now correctly passes `weights=None` (single None) when no weights are supplied.
- Everything else unchanged and fully backward-compatible.

Quick-reference (unchanged)
---------------
  rp = reco_particles(evtdf)
  tp = true_particles(evtdf)
  ri = reco_interactions(evtdf)
  ti = true_interactions(evtdf)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================
# Colour / style palette (unchanged)
# ============================================================
CAT_COLORS = {
    0: "#56B4E9",   # nueCC in FV (signal)
    1: "#0072B2",   # nueCC out FV
    2: "green",   # numuCC + pi0
    3: "#009E73",   # NC pi0
    4: "#E69F00",   # other numuCC
    5: "#CC79A7",   # NC other
    6: "#F0E442",   # cosmic
   -1: "#7F7F7F",   # unknown
}
CAT_LABELS = {
    0: r"$\nu_e$ CC (FV)",
    1: r"$\nu_e$ CC (out FV)",
    2: r"$\nu_\mu$ CC $\pi^0$",
    3: r"$\nu$ NC $\pi^0$",
    4: r"Other $\nu_\mu$ CC",
    5: r"Other $\nu$ NC",
    6: r"Cosmic",
   -1: r"Unknown",
}

PID_PHOTON = 0
PID_ELECTRON = 1
PID_MUON = 2
PID_PION = 3
PID_PROTON = 4

# ============================================================
# ParticleView / InteractionView wrappers (unchanged)
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

    def __len__(self): return len(self._df)
    def __repr__(self): return f"ParticleView({len(self._df):,} rows)"
    @property
    def df(self): return self._df
    @property
    def index(self): return self._df.index
    @property
    def columns(self): return self._df.columns
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)

class InteractionView:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getattr__(self, name: str):
        try:
            return self._df[name]
        except KeyError:
            raise AttributeError(f"No column '{name}' in InteractionView") from None

    def __getitem__(self, key):
        return self._df[key]

    def __len__(self): return len(self._df)
    def __repr__(self): return f"InteractionView({len(self._df):,} rows)"
    @property
    def df(self): return self._df
    @property
    def index(self): return self._df.index
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)

# Top-level constructors (unchanged)
def reco_particles(evtdf) -> ParticleView:
    return ParticleView(evtdf["rec"]["dlp"]["particles"])

def true_particles(evtdf) -> ParticleView:
    return ParticleView(evtdf["rec"]["dlp_true"]["particles"])

def reco_interactions(evtdf) -> InteractionView:
    return InteractionView(evtdf["rec"]["dlp"])

def true_interactions(evtdf) -> InteractionView:
    return InteractionView(evtdf["rec"]["dlp_true"])

# ============================================================
# POT helpers (unchanged)
# ============================================================
def compute_pot_scale(histpotdf, target_pot: float = 6.6e20):
    total_pot = histpotdf["TotalPOT"].sum()
    return total_pot, target_pot / total_pot

# ============================================================
# Index utilities (unchanged)
# ============================================================
def inter_levels(evtdf) -> list:
    return list(range(evtdf.index.nlevels - 1))

def collapse_to_interactions(evtdf):
    il = inter_levels(evtdf)
    return evtdf.groupby(level=il).first()


_SEL_HDF_KEY = "sel_0"
 
 
def build_and_save_selection(path, presel_inter, truth_cat_presel,
                              reco_ke, reco_cos, true_ke, true_cos,
                              stage_idx_dict, cut_names, pot_scale,
                            reco_p=None, true_p=None):
    """
    Build a compact interaction-level selection DataFrame and write it to HDF5.
 
    One row per interaction in presel_inter.  All reco/true observables and
    per-stage boolean flags are stored so that plotting cells never need evtdf.
 
    Parameters
    ----------
    path             : output path  (e.g. PLOT_DIR/selected_nuecc.df)
    presel_inter     : pd.MultiIndex — interactions in evtdf (FM+FV baseline)
    truth_cat_presel : pd.Series    — truth category aligned to presel_inter
    reco_ke          : pd.Series    — leading-electron reco KE (all presel)
    reco_cos         : pd.Series    — leading-electron reco costheta
    true_ke          : pd.Series    — leading-electron true KE
    true_cos         : pd.Series    — leading-electron true costheta
    stage_idx_dict   : dict  name → pd.Index  (interaction indices per stage)
    cut_names        : list[str]    — keys to save (e.g. ns.CUT_NAMES_MORE)
    pot_scale        : float        — POT scale factor (stored as metadata)
 
    Returns
    -------
    pd.DataFrame  (also written to path)
    """
    import os, warnings
    import numpy as np
    import pandas as pd
 
    sel = pd.DataFrame(index=presel_inter)
 
    # ── Truth ────────────────────────────────────────────────────────────────
    sel["truth_cat"]    = truth_cat_presel.reindex(presel_inter).astype("int8")
    sel["is_sig"]       = (sel["truth_cat"] == 0)
 
    # ── Reco observables ─────────────────────────────────────────────────────
    sel["reco_ke"]       = reco_ke.reindex(presel_inter).astype("float32")
    sel["reco_costheta"] = reco_cos.reindex(presel_inter).astype("float32")
    if reco_p is not None:                                  
        sel["reco_p"] = reco_p.reindex(presel_inter).astype("float32")

    if true_ke is not None:
        sel["true_ke"]       = true_ke.reindex(presel_inter).astype("float32")
    if true_cos is not None:
        sel["true_costheta"] = true_cos.reindex(presel_inter).astype("float32")
    if true_p is not None:                                 
        sel["true_p"] = true_p.reindex(presel_inter).astype("float32")
 
    # ── Per-stage boolean flags ───────────────────────────────────────────────
    # True = this interaction passed up to and including this cut stage
    for name in cut_names:
        if name in stage_idx_dict:
            sel[f"sel_{name}"] = presel_inter.isin(stage_idx_dict[name])
        else:
            warnings.warn(f"build_and_save_selection: stage {name!r} missing from stage_idx_dict")
 
    # ── Scalar metadata stored as a constant column ───────────────────────────
    sel["pot_scale"] = np.float32(pot_scale)
 
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel.to_hdf(path, key=_SEL_HDF_KEY, mode="w", complevel=1, complib="blosc")
 
    size_mb = os.path.getsize(path) / 1024**2
    print(f"Saved selection df → {path}")
    print(f"  shape={sel.shape}  size={size_mb:.1f} MB")
    return sel
 
 
def load_selection(path):
    """Load a selection df previously written by build_and_save_selection."""
    import pandas as pd
    return pd.read_hdf(path, key=_SEL_HDF_KEY)



    
# ============================================================
# Purity / efficiency (unchanged)
# ============================================================
def calc_pur_eff(sel_mask, sig_mask, weights=None):
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w = np.ones(len(sel_mask)) if weights is None else np.asarray(weights)
    n_sel = w[sel_mask].sum()
    n_true_pos = w[sel_mask & sig_mask].sum()
    n_sig = w[sig_mask].sum()
    pur = float(n_true_pos / n_sel) if n_sel > 0 else np.nan
    eff = float(n_true_pos / n_sig) if n_sig > 0 else np.nan
    return pur, eff

def pur_eff_cut_flow(cut_flow_masks, sig_mask, weights=None):
    purs, effs = [], []
    for mask in cut_flow_masks:
        p, e = calc_pur_eff(mask, sig_mask, weights)
        purs.append(p)
        effs.append(e)
    return purs, effs

def pur_eff_binned(values, bins, sel_mask, sig_mask, weights=None):
    values = np.asarray(values, dtype=float)
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w = np.ones(len(values)) if weights is None else np.asarray(weights)
    bins = np.asarray(bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    purs = np.full(len(centers), np.nan)
    effs = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (values >= lo) & (values < hi)
        counts[i] = w[in_bin & sig_mask].sum()
        purs[i], effs[i] = calc_pur_eff(sel_mask & in_bin, sig_mask & in_bin, w)
    return centers, purs, effs, counts

# ============================================================
# Plotting helpers (FIXED)
# ============================================================
def plot_stacked_hist(series_list, labels, colors, bins, weights=None,
                      xlabel="", ylabel="Events / bin", title="",
                      density=False, ax=None, data_series=None,
                      data_label="Data", pot_label="",
                      invert_stack_order=False,
                      show_counts=True,
                      show_percentage=True,
                      **hist_kw):
    """
    Stacked MC histogram with optional data overlay.

    New parameters vs old version
    ------------------------------
    invert_stack_order : bool
        If True, reverses both the stack draw order and the legend order,
        so the largest category sits at the bottom and is listed first —
        matching the SpineSpectra1D draw() style.
    show_counts : bool
        Append the weighted event count for each category to its legend label.
    show_percentage : bool
        Append the percentage of the total MC to each legend label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # ── Normalise weights ────────────────────────────────────────────────────
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

    # ── Strip NaN, keeping weights aligned ──────────────────────────────────
    clean = []
    for s, w in zip(series_list, _weights):
        s = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        clean.append((s[mask], w[mask] if w is not None else None))

    # ── Compute per-category totals for legend annotation ───────────────────
    bins_arr = np.asarray(bins)
    cat_totals = []
    for s, w in clean:
        counts, _ = np.histogram(s, bins=bins_arr,
                                 weights=w if w is not None else None)
        cat_totals.append(counts.sum())
    grand_total = sum(cat_totals)

    # ── Build annotated labels ───────────────────────────────────────────────
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

    # ── Optionally reverse stack order ──────────────────────────────────────
    if invert_stack_order:
        x_list  = [c[0] for c in reversed(clean)]
        w_list  = [c[1] for c in reversed(clean)]
        ann_lbl = list(reversed(annotated_labels))
        col_list = list(reversed(colors))
    else:
        x_list  = [c[0] for c in clean]
        w_list  = [c[1] for c in clean]
        ann_lbl = annotated_labels
        col_list = list(colors)

    weights_arg = None if all(w is None for w in w_list) else w_list

    ax.hist(
        x_list,
        bins=bins_arr,
        weights=weights_arg,
        label=ann_lbl,
        color=col_list,
        stacked=True,
        histtype="stepfilled",
        density=density,
        **hist_kw,
    )

    # ── Optional data overlay ────────────────────────────────────────────────
    if data_series is not None:
        d = np.asarray(data_series, dtype=float)
        d = d[~np.isnan(d)]
        counts, _ = np.histogram(d, bins=bins_arr)
        centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
        ax.errorbar(centers, counts, yerr=np.sqrt(counts),
                    fmt="ko", markersize=4, label=data_label, zorder=10)

    # ── Legend: invert handle order so top of stack = top of legend ─────────
    handles, legend_labels = ax.get_legend_handles_labels()
    if invert_stack_order:
        # Already reversed above — just display as-is so signal is first
        ax.legend(handles, legend_labels, fontsize=10, ncol=1, loc="best")
    else:
        # Reverse so the top-stacked (last drawn) category appears first
        ax.legend(handles[::-1], legend_labels[::-1], fontsize=10, ncol=1, loc="best")
    # ── Legend: invert handle order so top of stack = top of legend ─────────
    handles, legend_labels = ax.get_legend_handles_labels()
    if invert_stack_order:
        ax.legend(handles, legend_labels, fontsize=9, ncol=1,
                  loc="upper right", frameon=True,
                  framealpha=0.85, edgecolor="none")
    else:
        ax.legend(handles[::-1], legend_labels[::-1], fontsize=9, ncol=1,
                  loc="upper right", frameon=True,
                  framealpha=0.85, edgecolor="none")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)
    if pot_label:
        ax.text(0.02, 0.98, pot_label, transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="gray")
        ax.text(0.02, 0.93, f"Total MC events: {grand_total:.0f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color="gray")
    else:
        ax.text(0.02, 0.98, f"Total MC events: {grand_total:.0f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color="gray")
    return ax.get_figure(), ax


# ============================================================
# 2D resolution / bias plots  (true vs reco with marginals)
# ============================================================
def plot_hist2d_frac_err(x, y, xlabel='x', ylabel='y', title=None,
                         cmap='Blues', plot_line=True,
                         normalize=True, use_errorbar=False,
                         fit_curve=True, show_fit=False,
                         log_color=False,
                         fontsize=13,
                         **pltkwargs):
    from scipy.optimize import curve_fit
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import AutoMinorLocator

    if 'bins' not in pltkwargs:
        raise ValueError('bins must be provided as a keyword argument')
    bins        = np.asarray(pltkwargs.pop('bins'))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # ── Crystal Ball function ─────────────────────────────────────────────
    def _crystal_ball(t, A, a, n, mu, sigma):   # <-- added A
        t     = np.asarray(t, dtype=float)
        out   = np.empty_like(t)
        z     = (t - mu) / sigma
        gauss = z > -a
        out[gauss]  = np.exp(-0.5 * z[gauss]**2)
        Ac    = (n / a)**n * np.exp(-0.5 * a**2)
        B     = n / a - a
        out[~gauss] = Ac * (B - z[~gauss])**(-n)
        return A * out                           # <-- scaled by A

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
                # p0: alpha=1.5 (transition ~1.5σ from mean),
                #     n=2 (soft tail), mu=mean, sigma=std
                A0 = max(float(_h.max()), 1.0)
                popt, _ = curve_fit(
                    _crystal_ball, _c, _h,
                    p0     = [A0,   1.5,  2.0,  _mean,        _std          ],
                    bounds = ([0,   0.1,  1.01, _mean - 3*_std, 1e-9        ],
                              [20*A0+1, 5.0, 50.0, _mean + 3*_std, 5*_std   ]),
                    maxfev = 8000,
                )
                # popt = [A, alpha, n, mu, sigma]  — shift indices by 1
                bias[i] = popt[3]   # mu    (was popt[2])
                err[i]  = popt[4]   # sigma (was popt[3])
            except Exception:
                # fallback: plain mean/std if CB fit fails
                bias[i] = _mean
                err[i]  = _std

            if show_fit:
                _t = np.linspace(_mean - _half, _mean + _half, 200)
                plt.figure()
                plt.hist(stat, bins=_fbins, label=f'Raw (n={len(stat):,})',
                         density=True, alpha=0.6)
                # normalise CB curve for overlay
                # inside the show_fit block — update index references
                _cb  = _crystal_ball(_t, *popt)
                _cb /= (_cb.sum() * (_t[1] - _t[0]))
                plt.plot(_t, _cb, lw=2, label=(
                    f'Crystal Ball\n'
                    f'μ={popt[3]:.3f}, σ={popt[4]:.3f}\n'   # 3,4 not 2,3
                    f'α={popt[1]:.2f}, n={popt[2]:.2f}'     # 1,2 not 0,1
                ))
                plt.axvline(popt[3],             ls='--', color='blue',  label='μ')
                plt.axvline(popt[3] - popt[4],   ls=':',  color='green', label='μ±σ')
                plt.axvline(popt[3] + popt[4],   ls=':',  color='green')
                plt.title(f'Bin [{bins[i]:.3g}, {bins[i+1]:.3g}]')
                plt.legend(fontsize=8)
                plt.show()
        else:
            bias[i] = _mean
            err[i]  = _std

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6, 8))
    gs  = GridSpec(
        2, 2,
        height_ratios = [4, 1],
        width_ratios  = [20, 1],
        hspace        = 0.03,
        wspace        = 0.15,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[1, 0])
    cax     = fig.add_subplot(gs[0, 1])

    plt.setp(ax_main.get_xticklabels(), visible=False)

    valid_mask = x.notna() & y.notna()
    _norm      = LogNorm() if log_color else None

    h = ax_main.hist2d(
        x[valid_mask].values, y[valid_mask].values,
        bins=[bins, bins], cmap=cmap, norm=_norm,
        **pltkwargs
    )
    fig.colorbar(h[3], cax=cax, label='Counts')
    cax.yaxis.label.set_size(fontsize - 2)
    cax.tick_params(labelsize=fontsize - 3)

    if plot_line:
        ax_main.plot(
            [bins[0], bins[-1]], [bins[0], bins[-1]],
            ls='--', color='red', lw=1.2, zorder=5,
        )

    ax_main.set_xlim(bins[0], bins[-1])
    ax_main.set_ylim(bins[0], bins[-1])
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_ylabel(ylabel, fontsize=fontsize)
    ax_main.tick_params(axis='both', labelsize=fontsize - 2)
    if title:
        ax_main.set_title(title, fontsize=fontsize)

    # ── "SBND Simulation" watermark ───────────────────────────────────────
    ax_main.text(
        0.03, 0.97,
        r'$\mathbf{SBND}$ Simulation',
        transform  = ax_main.transAxes,
        va='top', ha='left',
        fontsize   = fontsize - 2,
        color      = 'gray',
    )
    if use_errorbar:
        ax2.errorbar(bin_centers, bias, yerr=err,
                     fmt='o', color='black', markersize=5)
    else:
        ax2.scatter(bin_centers, bias, color='black', s=25,
                    label='Fractional error', zorder=4)
        ax2.scatter(bin_centers, err, color='green', s=40,
                    label='Resolution (Crystal-Ball fit σ)', marker='*', zorder=4)
        ax2.legend(fontsize=11, ncol=1, loc='upper right', framealpha=0.35)

    ax2.axhline(0, ls='--', color='black', lw=1)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_ylabel('')
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

def plot_leading_e_resolution(true_series, reco_series,
                               bins, stage_idx=None,
                               xlabel='True value',
                               ylabel='Reco value',
                               normalize=True,
                               fit_curve=True,
                               title=None,
                               **kwargs):
    """
    Wrapper around plot_hist2d_frac_err for leading-electron
    true-vs-reco plots.

    Parameters
    ----------
    true_series : pd.Series  indexed by (ntuple, entry, interaction)
    reco_series : pd.Series  same index
    bins        : array-like
    stage_idx   : pd.Index, optional  — restrict to a specific cut stage
    normalize   : bool — fractional error if True
    fit_curve   : bool — Gaussian fit per bin
    """
    if stage_idx is not None:
        mask        = true_series.index.isin(stage_idx)
        true_series = true_series[mask]
        reco_series = reco_series[mask]

    valid       = true_series.notna() & reco_series.notna()
    x           = true_series[valid].astype(float)
    y           = reco_series[valid].astype(float)

    return plot_hist2d_frac_err(
        x, y,
        xlabel    = xlabel,
        ylabel    = ylabel,
        title     = title,
        normalize = normalize,
        fit_curve = fit_curve,
        bins      = np.asarray(bins),
        **kwargs,
    )
# ============================================================
# Threshold scan plots (shower-quality cut optimisation)
# ============================================================
def threshold_plots(evtdf, cut_flow_result, cut_stage, metric_col,
                    bounds, xaxis_name, plot_title,
                    threshold_range, cut_dir='<', legend_loc='right',
                    truth_categories=None, fig_width=7,
                    save_name=None, plot_folder="plots"):       

    from nue_selection import PID_ELECTRON, classify_truth

    il         = inter_levels(evtdf)
    presel_idx = cut_flow_result[cut_stage]["inter_index"]

    if truth_categories is None:
        cat = classify_truth(evtdf, verbose=False)
    else:
        cat = truth_categories

    rp    = reco_particles(evtdf)
    rp_df = rp._df

    # ── Resolve to a SCALAR column key (always one column, never a block) ─
    # pid_scores   → ('pid_scores',   'I1')   electron score
    # primary_scores → ('primary_scores', 'I1') electron primary score
    # everything else → plain string or ('col', '')
    _MULTI_COL_ELECTRON_IDX = {
        'pid_scores':     ('pid_scores',     'I1'),
        'primary_scores': ('primary_scores', 'I1'),
    }

    if isinstance(metric_col, str) and metric_col in _MULTI_COL_ELECTRON_IDX:
        col_key = _MULTI_COL_ELECTRON_IDX[metric_col]
    elif isinstance(metric_col, str):
        if metric_col in rp_df.columns:
            col_key = metric_col
        elif (metric_col, '') in rp_df.columns:
            col_key = (metric_col, '')
        else:
            matches = [c for c in rp_df.columns
                       if (isinstance(c, tuple) and c[0] == metric_col)
                       or c == metric_col]
            if not matches:
                raise KeyError(
                    f"'{metric_col}' not found in reco_particles.\n"
                    f"First 40 cols: {list(rp_df.columns)[:40]}"
                )
            col_key = matches[0]
    else:
        col_key = metric_col   # already a precise tuple

    # Verify it resolves to a single Series, not a DataFrame
    test = rp_df[col_key]
    if isinstance(test, pd.DataFrame):
        raise KeyError(
            f"col_key={col_key!r} still resolves to a DataFrame "
            f"with columns {list(test.columns)}. "
            f"Pass a more specific tuple, e.g. {[(col_key, c) for c in test.columns]}"
        )

    ke_col  = ('ke',  '') if ('ke',  '') in rp_df.columns else 'ke'
    pid_col = ('pid', '') if ('pid', '') in rp_df.columns else 'pid'

    # ── Electrons in pre-selected interactions ────────────────────────────
    inter_idx_of_particle = rp_df.index.droplevel(-1)
    in_presel  = inter_idx_of_particle.isin(presel_idx)
    ele_presel = rp_df[in_presel & (rp_df[pid_col] == PID_ELECTRON)]

    # ── Build flat tmp frame — both columns now guaranteed 1-D ───────────
    tmp = pd.DataFrame({
        'ke':     ele_presel[ke_col].to_numpy(dtype=float),
        'metric': ele_presel[col_key].to_numpy(dtype=float),
    }, index=ele_presel.index)

    # ── Leading electron per interaction: metric at max KE ────────────────
    def metric_at_max_ke(g):
        return g.loc[g['ke'].idxmax(), 'metric']

    leading_metric = tmp.groupby(level=il).apply(metric_at_max_ke)
    inter_index    = leading_metric.index

    # ── Metric & truth ────────────────────────────────────────────────────
    metric_vals = leading_metric.to_numpy(dtype=float).flatten()
    is_signal   = (cat.reindex(inter_index).fillna(-1).to_numpy().flatten() == 0)

    assert len(metric_vals) == len(is_signal), \
        f"BUG: metric={len(metric_vals)} vs signal={len(is_signal)}"

    valid       = ~np.isnan(metric_vals) & (metric_vals != -1)
    metric_vals = metric_vals[valid]
    is_signal   = is_signal[valid]

    if len(metric_vals) == 0:
        print(f"No valid events for '{metric_col}'. Skipping.")
        return None, None

    # ── Histogram ─────────────────────────────────────────────────────────
    xmin, xmax = bounds
    bin_edges  = np.linspace(xmin, xmax, 51)

    signal_vals     = metric_vals[is_signal]
    background_vals = metric_vals[~is_signal]

    fig = plt.figure(figsize=(fig_width, 8))
    gs  = fig.add_gridspec(4, 1, hspace=0.05)
    ax1 = fig.add_subplot(gs[:3, 0])

    ax1.hist(background_vals, bins=bin_edges, histtype='step',
             label=f'Background ({len(background_vals):,} events)',
             color='orange', density=True, linewidth=1.5)
    ax1.hist(signal_vals, bins=bin_edges, histtype='step',
             label=r'Signal $\nu_e$ CC' + f' ({len(signal_vals):,} events)',
             color='blue', density=True, linewidth=1.5)

    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.set_title(plot_title, fontsize=16, pad=10)
    ax1.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.tick_params(labelbottom=False)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', length=8, labelsize=12,
                    direction='in', width=1.2)
    ax1.tick_params(axis='both', which='minor', length=4,
                    direction='in', width=0.8)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # ── Threshold scan ────────────────────────────────────────────────────
    start, stop, step = threshold_range
    thresholds = np.arange(start, stop + step / 2, step)
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
        f1  = 2 * pur * eff / (pur + eff) if (pur + eff)  > 0 else 0.0

        purities.append(pur)
        efficiencies.append(eff)
        f1s.append(f1)

    f1s        = np.array(f1s)
    opt_idx    = np.argmax(f1s)
    opt_thresh = float(thresholds[opt_idx])
    max_f1     = float(f1s[opt_idx])

    # ── Performance panel ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(thresholds, purities,     'b-',           label='Purity',     linewidth=1.5)
    ax2.plot(thresholds, efficiencies, 'r-',           label='Efficiency', linewidth=1.5)
    ax2.plot(thresholds, f1s,          color='purple', label='F1 Score',   linewidth=1.5)
    ax2.axvline(opt_thresh, color='purple', linestyle='--', linewidth=1.5)

    ax2.set_xlabel(xaxis_name, fontsize=14)
    ax2.set_ylabel('Performance', fontsize=14)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(0, 1)
    ax2.legend(loc=f'upper {legend_loc}', framealpha=0.3, fontsize=12)

    range_x = xmax - xmin
    if cut_dir == '<':
        text_x, ha = opt_thresh + range_x * 0.05, 'left'
    else:
        text_x, ha = opt_thresh - range_x * 0.05, 'right'

    ax2.text(text_x, 0.05, f'Max F1 at {opt_thresh:.3f}',
             color='purple', va='bottom', ha=ha, fontsize=13,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                       edgecolor='none', alpha=0.5))

    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='major', length=6, labelsize=12,
                    direction='in', width=1)
    ax2.tick_params(axis='both', which='minor', length=3,
                    direction='in', width=0.8)
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

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
    """
    Sequential threshold scan: each cut is applied to survivors of all
    previous cuts. Histograms show the remaining population at each stage.

    Parameters
    ----------
    shower_cuts : list of tuples — same format as the notebook shower_cuts list:
        (metric_col, bounds, xaxis_name, threshold_range, cut_dir, legend_loc, save_name)
    opt_thresholds_prev : dict  col → threshold  (pre-applied before scanning)
        Pass previously determined thresholds to start from a partially-cut sample.

    Returns
    -------
    dict  col → optimal_threshold
    """
    from nue_selection import PID_ELECTRON, classify_truth

    il         = inter_levels(evtdf)
    presel_idx = cut_flow_result[cut_stage]["inter_index"]

    if truth_categories is None:
        cat = classify_truth(evtdf, verbose=False)
    else:
        cat = truth_categories

    rp    = reco_particles(evtdf)
    rp_df = rp._df

    _MULTI_COL_ELECTRON_IDX = {
        'pid_scores':     ('pid_scores',     'I1'),
        'primary_scores': ('primary_scores', 'I1'),
    }

    def _resolve_col(metric_col):
        if isinstance(metric_col, str) and metric_col in _MULTI_COL_ELECTRON_IDX:
            return _MULTI_COL_ELECTRON_IDX[metric_col]
        if isinstance(metric_col, str):
            if metric_col in rp_df.columns:
                return metric_col
            if (metric_col, '') in rp_df.columns:
                return (metric_col, '')
            matches = [c for c in rp_df.columns
                       if (isinstance(c, tuple) and c[0] == metric_col) or c == metric_col]
            if not matches:
                raise KeyError(f"'{metric_col}' not found in reco_particles.")
            return matches[0]
        return metric_col

    ke_col  = ('ke',  '') if ('ke',  '') in rp_df.columns else 'ke'
    pid_col = ('pid', '') if ('pid', '') in rp_df.columns else 'pid'

    # ── Build per-interaction metric frame for ALL shower variables at once ──
    inter_idx_of_particle = rp_df.index.droplevel(-1)
    in_presel  = inter_idx_of_particle.isin(presel_idx)
    ele_presel = rp_df[in_presel & (rp_df[pid_col] == PID_ELECTRON)]

    # Build a flat df: one row per interaction, one column per metric
    metric_frames = {}
    for (metric_col, *_) in shower_cuts:
        col_key = _resolve_col(metric_col)
        tmp = pd.DataFrame({
            'ke':     ele_presel[ke_col].to_numpy(dtype=float),
            'metric': ele_presel[col_key].to_numpy(dtype=float),
        }, index=ele_presel.index)
        def _max_ke_metric(g):
            return g.loc[g['ke'].idxmax(), 'metric']
        metric_frames[metric_col] = tmp.groupby(level=il).apply(_max_ke_metric)

    # Master frame: interactions × metrics
    master = pd.DataFrame(metric_frames)
    master['is_signal'] = (cat.reindex(master.index).fillna(-1) == 0).values

    # ── Apply any pre-existing thresholds to start from survivors ────────────
    surviving_mask = pd.Series(True, index=master.index)
    if opt_thresholds_prev:
        for metric_col, bounds, xaxis_name, trange, cut_dir, lloc, sname in shower_cuts:
            if metric_col in opt_thresholds_prev:
                thresh = opt_thresholds_prev[metric_col]
                col_vals = master[metric_col]
                if cut_dir == '>':
                    surviving_mask &= (col_vals > thresh)
                else:
                    surviving_mask &= (col_vals < thresh)

    opt_thresholds = {}

    for metric_col, bounds, xaxis_name, trange, cut_dir, lloc, sname in shower_cuts:
        col_key    = _resolve_col(metric_col)
        xmin, xmax = bounds

        # ── Current survivors (after all previous cuts) ───────────────────
        current    = master[surviving_mask]
        metric_vals= current[metric_col].to_numpy(dtype=float)
        is_signal  = current['is_signal'].to_numpy()

        valid       = ~np.isnan(metric_vals)
        metric_vals = metric_vals[valid]
        is_signal   = is_signal[valid]

        n_total     = len(metric_vals)
        n_signal    = is_signal.sum()
        n_bkg       = (~is_signal).sum()

        signal_vals     = metric_vals[is_signal]
        background_vals = metric_vals[~is_signal]

        # ── Plot ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(fig_width, 8))
        gs  = fig.add_gridspec(4, 1, hspace=0.05)
        ax1 = fig.add_subplot(gs[:3, 0])

        ax1.hist(background_vals, bins=np.linspace(xmin, xmax, 51),
                 histtype='step', density=True,
                 label=f'Background ({n_bkg:,} remaining)',
                 color='orange', linewidth=1.5)
        ax1.hist(signal_vals, bins=np.linspace(xmin, xmax, 51),
                 histtype='step', density=True,
                 label=r'Signal $\nu_e$CC' + f' ({n_signal:,} remaining)',
                 color='blue', linewidth=1.5)

        n_cuts_applied = sum(1 for c, *_ in shower_cuts
                             if c in opt_thresholds or
                             (opt_thresholds_prev and c in opt_thresholds_prev))
        ax1.set_title(
            f"{xaxis_name} — threshold scan\n"
            f"({n_total:,} interactions surviving previous cuts)",
            fontsize=14, pad=8
        )
        ax1.set_ylabel('Probability Density', fontsize=13)
        ax1.legend(loc=f'upper {lloc}', framealpha=0.3, fontsize=11)
        ax1.set_xlim(xmin, xmax)
        ax1.tick_params(labelbottom=False)
        ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)

        # ── Threshold scan on surviving population ────────────────────────
        start, stop, step = trange
        thresholds = np.arange(start, stop + step / 2, step)
        total_signal = is_signal.sum()
        keep_above   = (cut_dir == '>')

        purities, efficiencies, f1s = [], [], []
        for thresh in thresholds:
            passed       = (metric_vals > thresh) if keep_above else (metric_vals < thresh)
            n_passed     = passed.sum()
            n_sig_passed = (passed & is_signal).sum()
            pur = n_sig_passed / n_passed     if n_passed     > 0 else 0.0
            eff = n_sig_passed / total_signal if total_signal > 0 else 0.0
            f1  = 2 * pur * eff / (pur + eff) if (pur + eff)  > 0 else 0.0
            purities.append(pur); efficiencies.append(eff); f1s.append(f1)

        f1s        = np.array(f1s)
        opt_idx    = np.argmax(f1s)
        opt_thresh = float(thresholds[opt_idx])
        max_f1     = float(f1s[opt_idx])

        ax2 = fig.add_subplot(gs[3, 0])
        ax2.plot(thresholds, purities,     'b-',           label='Purity',     linewidth=1.5)
        ax2.plot(thresholds, efficiencies, 'r-',           label='Efficiency', linewidth=1.5)
        ax2.plot(thresholds, f1s,          color='purple', label='F1 Score',   linewidth=1.5)
        ax2.axvline(opt_thresh, color='purple', linestyle='--', linewidth=1.5)
        ax2.set_xlabel(xaxis_name, fontsize=13)
        ax2.set_ylabel('Performance', fontsize=13)
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(0, 1)
        ax2.legend(loc=f'upper {lloc}', framealpha=0.3, fontsize=11)
        range_x = xmax - xmin
        if cut_dir == '<':
            text_x, ha = opt_thresh + range_x * 0.05, 'left'
        else:
            text_x, ha = opt_thresh - range_x * 0.05, 'right'
        ax2.text(text_x, 0.05, f'Max F1 at {opt_thresh:.3f}',
                 color='purple', va='bottom', ha=ha, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='none', alpha=0.5))
        ax2.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        if sname:
            save_plot(sname + "_cumulative", fig=fig, folder_name=plot_folder)
        plt.show()

        print(f"\n{'─'*50}")
        print(f"  Variable   : {metric_col}")
        print(f"  Remaining  : {n_total:,} interactions ({n_signal} signal, {n_bkg} bkg)")
        print(f"  Cut dir    : {cut_dir} {opt_thresh:.4f}")
        print(f"  Efficiency : {efficiencies[opt_idx]*100:.1f}%  (of remaining signal)")
        print(f"  Purity     : {purities[opt_idx]*100:.1f}%")
        print(f"  Max F1     : {max_f1:.4f}")
        print(f"{'─'*50}\n")

        opt_thresholds[metric_col] = opt_thresh

        # ── Apply this cut to survivors for the next iteration ────────────
        col_vals = master[metric_col]
        if cut_dir == '>':
            surviving_mask &= (col_vals > opt_thresh)
        else:
            surviving_mask &= (col_vals < opt_thresh)

    print(f"\nFinal surviving interactions: {surviving_mask.sum():,}")
    return opt_thresholds

    
    
# The rest of the plotting helpers are unchanged
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
    """
    Create a clean purity/efficiency summary table.
    Vertical row height now automatically scales with the number of cuts
    so the table never looks compressed even with 10+ rows.
    """
    ax.axis("off")

    _labels = list(labels)
    pur_pct = [p * 100 if (p is not None and not np.isnan(p)) else np.nan for p in purs]
    eff_pct = [e * 100 if (e is not None and not np.isnan(e)) else np.nan for e in effs]

    results = pd.DataFrame({
        "Cut": _labels,
        "Purity [%]": [f"{p:.2f}" if not np.isnan(p) else "—" for p in pur_pct],
        "Efficiency [%]": [f"{e:.2f}" if not np.isnan(e) else "—" for e in eff_pct],
        "ΔEff [%]": ["—"] + [f"{eff_pct[i] - eff_pct[i-1]:.2f}" for i in range(1, len(eff_pct))],
        "ΔPur [%]": ["—"] + [f"{pur_pct[i] - pur_pct[i-1]:.2f}" for i in range(1, len(pur_pct))],
    })

    table = ax.table(
        cellText=results.values.tolist(),
        colLabels=results.columns.tolist(),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10 if len(results) <= 10 else 9)   # slightly smaller font for very long tables

    # ── Dynamic row height: scales automatically with number of rows ──
    n_rows = len(results) + 1                    # +1 for the header row
    # This formula keeps the table nicely filling the axes:
    #   ~10 rows → scale ≈ 1.8–2.0
    #   ~15 rows → scale ≈ 1.3
    #   20+ rows → scale ≈ 1.1 (still readable)
    row_scale = max(1.05, 18.0 / n_rows)
    table.scale(1.0, row_scale)

    # ── Column widths: Cut column auto, numeric columns fixed ─────────────
    col_widths = {0: None, 1: 0.13, 2: 0.15, 3: 0.12, 4: 0.12}

    for (row, col), cell in table.get_celld().items():
        w = col_widths.get(col)
        if w is not None:
            cell.set_width(w)

        # Header styling
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
        # Alternating body rows
        elif row % 2 == 0:
            cell.set_facecolor("#eaf0fb")
            cell.set_edgecolor("#cccccc")
        else:
            cell.set_facecolor("#ffffff")
            cell.set_edgecolor("#cccccc")

        # Delta columns in muted italic
        if row > 0 and col in (3, 4):
            cell.set_text_props(color="#555555", style="italic")

    return ax, table


def plot_matrix_binned(
    vals,
    count_matrix,
    bin_values,
    bin_labels,
    cut_labels,
    remove_dummy_col=False,
    xlabel="",
    title="",
    cmap="Greens",
    vmin=0,
    vmax=1,
    label_fontsize=11,
    cell_size=0.8,
):
    if remove_dummy_col:
        vals         = vals[:, :-1]
        count_matrix = count_matrix[:, :-1]
        cut_labels   = cut_labels[:-1]

    n_bins, n_cuts = vals.shape

    # Set figure size so that each cell is approximately square
    fig, ax = plt.subplots(
        figsize=(cell_size * n_bins * 1.3, cell_size * n_cuts* 1.3)
    )

    im = ax.imshow(vals.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

    # Axis ticks
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right",
                       fontsize=label_fontsize)
    ax.set_yticks(np.arange(n_cuts))
    ax.set_yticklabels(cut_labels, fontsize=label_fontsize)

    # Annotate cells
    for ci in range(n_cuts):
        for bi in range(n_bins):
            v = vals[bi, ci]
            n = count_matrix[bi, ci]
            if not np.isnan(v):
                text_color = "white" if v > 0.5 else "black"
                ax.text(
                    bi, ci,
                    f"{v:.3f}\n({int(n)})",
                    ha="center", va="center",
                    fontsize=label_fontsize,
                    color=text_color,
                )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cut stage", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, ax


def save_plot(name, fig=None, folder_name="plots", dpi=150):
    import os
    os.makedirs(folder_name, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(f"{folder_name}/{name}.png", dpi=dpi, bbox_inches="tight")
