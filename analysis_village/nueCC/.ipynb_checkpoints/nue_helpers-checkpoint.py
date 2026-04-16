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
        fig, ax = plt.subplots(figsize=(8, 5))

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
        ax.legend(handles, legend_labels, fontsize=9, ncol=1)
    else:
        # Reverse so the top-stacked (last drawn) category appears first
        ax.legend(handles[::-1], legend_labels[::-1], fontsize=9, ncol=1)

    # ── Grand total annotation ───────────────────────────────────────────────
    ax.text(0.98, 0.98,
            f"Total MC: {grand_total:.1f}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=9, color="gray")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)
    if pot_label:
        ax.text(0.02, 0.98, pot_label, transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color="gray")

    return ax.get_figure(), ax

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
    ax.axis("off")
    _labels = list(labels)
    pur_pct = [p * 100 if (p is not None and not np.isnan(p)) else np.nan for p in purs]
    eff_pct = [e * 100 if (e is not None and not np.isnan(e)) else np.nan for e in effs]

    results = pd.DataFrame({
        "Cut":         _labels,
        "Purity [%]":  [f"{p:.2f}" if not np.isnan(p) else "—" for p in pur_pct],
        "Efficiency [%]": [f"{e:.2f}" if not np.isnan(e) else "—" for e in eff_pct],
        "ΔEff [%]":    ["—"] + [f"{eff_pct[i] - eff_pct[i-1]:.2f}" for i in range(1, len(eff_pct))],
        "ΔPur [%]":    ["—"] + [f"{pur_pct[i] - pur_pct[i-1]:.2f}" for i in range(1, len(pur_pct))],
    })

    table = ax.table(
        cellText=results.values.tolist(),
        colLabels=results.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # ── Column widths: Cut column is auto; numeric columns are fixed ──────────
    col_widths = {0: None, 1: 0.13, 2: 0.15, 3: 0.12, 4: 0.12}
    n_rows = len(results) + 1  # +1 for header

    for (row, col), cell in table.get_celld().items():
        w = col_widths.get(col)
        if w is not None:
            cell.set_width(w)

        # Header row
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

        # Highlight delta columns in muted tones
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
