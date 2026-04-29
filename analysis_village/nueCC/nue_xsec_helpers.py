#!/usr/bin/env python3
"""
nue_xsec_helpers.py
-------------------
Helper utilities for the SBND nueCC inclusive cross-section analysis.

Provides
--------
VariableConfig           — binning + column specs per kinematic observable
                           Presets: electron_ke, electron_costheta, nu_energy_reco
get_int_category         — truth interaction classification (cats 0–7, matches
                           make_nueCC_df._build_truth_counts)
get_primary_reco_electron — interaction-level df of highest-KE primary reco e–
get_primary_true_electron — interaction-level df of highest-KE primary true e–
add_mcstat_weights       — append N Poisson-universe weight columns to evtdf
compute_mcstat_covariance — MCstat covariance dict for signal + background
plot_stacked_topology    — stacked histogram with nuint_categ breakdown
plot_heatmap             — coloured matrix heatmap (wraps cafpyana style)
plot_uncertainty_budget  — fractional uncertainty per source (bar chart)
save_cov_npz             — archive covariance arrays for future syst studies
                           (DEFINED here but not yet called in the notebook)

Integration with cafpyana
-------------------------
Column discovery uses _find_col / _find_col_ends from make_nueCC_df.
Unfolding maths (WienerSVD, get_smear_matrix, …) live in
  analysis_village.unfolding.utils   (re-exports from wienersvd.py)
and are imported directly in the notebook; only lightweight wrappers for
heatmap formatting are defined here.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm


# =============================================================================
# Constants  (must match make_nueCC_df.py / nue_selection.py)
# =============================================================================
BRANCH_TRUE            = "dlp_true"
ELECTRON_THRESHOLD_MEV = 75.0
MUON_THRESHOLD_MEV     = 50.0
PID_ELECTRON           = 1
PID_MUON               = 2


# =============================================================================
# Column-finding utilities  (duplicated from make_nueCC_df for standalone use)
# =============================================================================

def _find_col(df, suffix, branch_must_contain=None, branch_must_not_contain=None):
    """Return the first MultiIndex column whose last non-empty part == suffix."""
    for col in df.columns:
        parts = [p for p in (col if isinstance(col, tuple) else (col,)) if p != ""]
        if not parts or parts[-1] != suffix:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        if branch_must_not_contain and any(branch_must_not_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(
        f"suffix={suffix!r} branch_must_contain={branch_must_contain!r} "
        f"branch_must_not_contain={branch_must_not_contain!r} not found"
    )


def _find_col_ends(df, *suffix_parts, branch_must_contain=None):
    """Find column whose last N non-empty parts equal suffix_parts."""
    n      = len(suffix_parts)
    target = tuple(suffix_parts)
    for col in df.columns:
        parts = tuple(p for p in (col if isinstance(col, tuple) else (col,)) if p != "")
        if len(parts) < n or parts[-n:] != target:
            continue
        if branch_must_contain and not any(branch_must_contain in str(p) for p in col):
            continue
        return col
    raise KeyError(f"No column ending with {suffix_parts!r}")


def _safe(fn, *args, **kwargs):
    """Call fn; return None on KeyError / StopIteration."""
    try:
        return fn(*args, **kwargs)
    except (KeyError, StopIteration):
        return None


# =============================================================================
# Constants
# =============================================================================

# Truth category labels and colours — must match _build_truth_counts in
# make_nueCC_df.py and get_int_category below.
CAT_LABELS = {
    0: r"$\nu_e$ CC in FV (signal)",
    1: r"$\nu_e$ CC out of FV",
    2: r"$\nu_\mu$ CC $\pi^0$",
    3: r"NC $\pi^0$",
    4: r"Other $\nu_\mu$ CC",
    5: r"Other NC",
    6: r"Cosmic",
    7: "Other",
}
CAT_COLORS = {
    0: "#56B4E9",   # nueCC in FV (signal)
    1: "#0072B2",   # nueCC out FV
    2: "green",   # numuCC + pi0
    3: "#009E73",   # NC pi0
    4: "#E69F00",   # other numuCC
    5: "#CC79A7",   # NC other
    6: "#F0E442",   # cosmic
    7: "#7F7F7F",   # unknown
}
# Ordered for stacked-histogram plotting (signal on top)
CAT_PLOT_ORDER = [6, 5, 4, 3, 2, 1, 0]


# =============================================================================
# VariableConfig
# =============================================================================

class VariableConfig:
    """
    Binning + column specifications for a single kinematic variable.

    Parameters
    ----------
    var_save_name   : short identifier used in file names
    var_plot_name   : LaTeX string for axis labels (no $ wrapping needed)
    var_unit        : unit string (empty string if dimensionless)
    bins            : 1-D array of bin edges
    reco_col_suffix : suffix passed to _find_col to locate the reco column
                      in the particle-level evtdf (branch_must_not_contain
                      BRANCH_TRUE is applied automatically)
    true_col_suffix : suffix passed to _find_col for the truth column
                      (branch_must_contain BRANCH_TRUE is applied)
    scale_to_gev    : multiply KE/momentum columns by this factor
                      (SPINE stores MeV; bins may be in GeV → 1e-3)
    """

    def __init__(
        self,
        var_save_name,
        var_plot_name,
        var_unit,
        bins,
        reco_col_suffix,
        true_col_suffix,
        scale_to_gev=1.0,
    ):
        self.var_save_name   = var_save_name
        self.var_plot_name   = var_plot_name
        self.var_unit        = var_unit
        self.bins            = np.asarray(bins, dtype=float)
        self.bin_centers     = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.reco_col_suffix = reco_col_suffix
        self.true_col_suffix = true_col_suffix
        self.scale_to_gev    = scale_to_gev

        unit_suffix = f"~[{var_unit}]" if var_unit else ""
        base = r"$\mathrm{" + var_plot_name + unit_suffix + r"}$"
        reco = r"$\mathrm{" + var_plot_name + r"^{reco.}" + unit_suffix + r"}$"
        true = r"$\mathrm{" + var_plot_name + r"^{true}" + unit_suffix + r"}$"
        self.var_labels = [base, reco, true]

    # ── column-finding helpers ────────────────────────────────────────────────

    def reco_col(self, df):
        """Return the MultiIndex column tuple for the reco variable."""
        return _find_col(df, self.reco_col_suffix,
                         branch_must_not_contain=BRANCH_TRUE)

    def true_col(self, df):
        """Return the MultiIndex column tuple for the truth variable."""
        return _find_col(df, self.true_col_suffix,
                         branch_must_contain=BRANCH_TRUE)

    def get_reco_vals(self, df):
        return df[self.reco_col(df)] * self.scale_to_gev

    def get_true_vals(self, df):
        return df[self.true_col(df)] * self.scale_to_gev

    # ── presets ───────────────────────────────────────────────────────────────

    @classmethod
    def electron_ke(cls):
        """Primary electron kinetic energy  [GeV]."""
        return cls(
            var_save_name   = "electron_ke",
            var_plot_name   = r"T_e",
            var_unit        = "GeV",
            bins            = np.array([0.075, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]),
            reco_col_suffix = "ke",
            true_col_suffix = "ke",
            scale_to_gev    = 1e-3,   # SPINE stores MeV
        )

    @classmethod
    def electron_costheta(cls):
        """Primary electron direction cos θ w.r.t. beam (+z)."""
        return cls(
            var_save_name   = "electron_costheta",
            var_plot_name   = r"\cos\theta_e",
            var_unit        = "",
            bins            = np.linspace(-1.0, 1.0, 9),
            reco_col_suffix = "dir_z",    # SPINE: rec.dlp.particles.dir_z
            true_col_suffix = "dir_z",
            scale_to_gev    = 1.0,
        )

    @classmethod
    def nu_energy_reco(cls):
        """
        Reconstructed neutrino energy  E_ν^reco = E_e + E_had  [GeV].
        NOTE: This is an interaction-level quantity computed after
        get_primary_reco_electron; the column must be added to the
        interaction df before calling.  Use col suffix 'nu_energy_reco'
        which nue_xsec_helpers.build_nu_energy_reco() adds explicitly.
        """
        return cls(
            var_save_name   = "nu_energy_reco",
            var_plot_name   = r"E_\nu^{reco}",
            var_unit        = "GeV",
            bins            = np.array([0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.5]),
            reco_col_suffix = "nu_energy_reco",
            true_col_suffix = "nu_energy_true",
            scale_to_gev    = 1.0,
        )


# =============================================================================
# Truth categorisation
# =============================================================================

def get_int_category(evtdf):
    """
    Assign each interaction (first nlevels–1 index levels) a truth category.

    Category definitions (mirror _build_truth_counts in make_nueCC_df.py):
      0  nueCC in FV, exactly 1 primary e– KE > threshold  ← signal
      1  nueCC out FV
      2  numuCC + π⁰
      3  NC  + π⁰
      4  other numuCC (no π⁰)
      5  other NC
      6  not ν (cosmic / dirt)
      7  other in-FV ν

    Returns
    -------
    pd.Series  indexed by the interaction-level MultiIndex, dtype int8
    """
    il    = list(range(evtdf.index.nlevels - 1))
    inter = evtdf.groupby(level=il).first()
    idx   = inter.index

    def _b(s):
        return s.reindex(idx, fill_value=False).astype(bool)

    # ── interaction-level truth flags ─────────────────────────────────────────
    nu_col   = _safe(_find_col, evtdf, "nu_id",        branch_must_contain=BRANCH_TRUE)
    cc_col   = _safe(_find_col, evtdf, "current_type", branch_must_contain=BRANCH_TRUE)
    fv_t_col = _safe(_find_col, evtdf, "is_fiducial",  branch_must_contain=BRANCH_TRUE,
                                                        branch_must_not_contain=None)
    is_nu      = _b(inter[nu_col]   >= 0) if nu_col   else pd.Series(False, index=idx)
    is_cc      = _b(inter[cc_col]   == 0) if cc_col   else pd.Series(False, index=idx)
    is_fv_true = _b(inter[fv_t_col] == 1) if fv_t_col else pd.Series(False, index=idx)

    # ── particle-level truth aggregation ──────────────────────────────────────
    pid_col = _safe(_find_col, evtdf, "pid",              branch_must_contain=BRANCH_TRUE)
    pri_col = _safe(_find_col, evtdf, "is_primary",       branch_must_contain=BRANCH_TRUE)
    ke_col  = _safe(_find_col, evtdf, "ke",               branch_must_contain=BRANCH_TRUE)
    ppd_col = _safe(_find_col, evtdf, "parent_pdg_code",  branch_must_contain=BRANCH_TRUE)

    if pid_col and pri_col and ke_col:
        elec_mask = (
            (evtdf[pid_col] == PID_ELECTRON) &
            (evtdf[pri_col] == 1) &
            (evtdf[ke_col]  > ELECTRON_THRESHOLD_MEV)
        )
        has_elec = (
            elec_mask.groupby(level=il).sum().reindex(idx, fill_value=0) == 1
        )
        has_muon = (
            ((evtdf[pid_col] == PID_MUON) &
             (evtdf[pri_col] == 1) &
             (evtdf[ke_col]  > MUON_THRESHOLD_MEV))
            .groupby(level=il).any().reindex(idx, fill_value=False)
        )
    else:
        has_elec = pd.Series(False, index=idx)
        has_muon = pd.Series(False, index=idx)

    has_pi0 = (
        (evtdf[ppd_col].abs() == 111)
        .groupby(level=il).any().reindex(idx, fill_value=False)
    ) if ppd_col else pd.Series(False, index=idx)

    # ── assign categories ─────────────────────────────────────────────────────
    cat = pd.Series(7, index=idx, dtype=np.int8)
    cat[~is_nu]                                        = 6   # cosmic/not-ν
    nu = is_nu
    cat[nu &  is_cc & ~is_fv_true & has_elec]          = 1   # nueCC out FV
    cat[nu &  is_cc &  is_fv_true & has_elec]          = 0   # ← SIGNAL
    cat[nu &  is_cc & has_muon    & has_pi0 ]           = 2   # numuCC + π⁰
    cat[nu & ~is_cc & has_pi0]                          = 3   # NC π⁰
    cat[nu &  is_cc & has_muon    & ~has_pi0]           = 4   # other numuCC
    cat[nu & ~is_cc & ~has_pi0]                         = 5   # other NC

    return cat


# =============================================================================
# Primary-particle extraction
# =============================================================================

def _il(df):
    return list(range(df.index.nlevels - 1))


def get_primary_reco_electron(evtdf, ke_threshold=ELECTRON_THRESHOLD_MEV):
    """
    For each interaction, return the primary reco electron with highest KE.

    Returns
    -------
    pd.DataFrame  indexed by the interaction-level MultiIndex.
                  Columns = same as evtdf (full row of the winning particle).
    """
    il     = _il(evtdf)
    pid_c  = _find_col(evtdf, "pid",        branch_must_not_contain=BRANCH_TRUE)
    pri_c  = _find_col(evtdf, "is_primary", branch_must_not_contain=BRANCH_TRUE)
    ke_c   = _find_col(evtdf, "ke",         branch_must_not_contain=BRANCH_TRUE)

    mask = (
        (evtdf[pid_c] == PID_ELECTRON) &
        (evtdf[pri_c] == 1) &
        (evtdf[ke_c]  > ke_threshold)
    )
    edf = evtdf[mask]
    if edf.empty:
        warnings.warn("get_primary_reco_electron: no electrons found after mask")
        return edf

    # Extract KE as a plain Series before groupby — pandas rejects MultiIndex
    # tuple column keys in groupby.__getitem__
    ke_series = edf[ke_c]
    idx_max   = ke_series.groupby(level=il).idxmax().dropna()
    return edf.loc[idx_max].droplevel(-1)   # drop particle index → interaction level


def get_primary_true_electron(evtdf, ke_threshold=ELECTRON_THRESHOLD_MEV):
    """
    For each interaction, return the primary *true* electron with highest KE.
    Uses the dlp_true branch columns.

    Returns
    -------
    pd.DataFrame  indexed by the interaction-level MultiIndex.
    """
    il     = _il(evtdf)
    pid_c  = _find_col(evtdf, "pid",        branch_must_contain=BRANCH_TRUE)
    pri_c  = _find_col(evtdf, "is_primary", branch_must_contain=BRANCH_TRUE)
    ke_c   = _find_col(evtdf, "ke",         branch_must_contain=BRANCH_TRUE)

    mask = (
        (evtdf[pid_c] == PID_ELECTRON) &
        (evtdf[pri_c] == 1) &
        (evtdf[ke_c]  > ke_threshold)
    )
    edf = evtdf[mask]
    if edf.empty:
        warnings.warn("get_primary_true_electron: no true electrons found")
        return edf

    ke_series = edf[ke_c]
    idx_max   = ke_series.groupby(level=il).idxmax().dropna()
    return edf.loc[idx_max].droplevel(-1)


# =============================================================================
# Neutrino energy helpers
# =============================================================================

def build_nu_energy_reco(reco_e_df, evtdf):
    """
    Append 'nu_energy_reco' (calorimetric) column to reco_e_df.

        E_ν^reco  =  E_e  +  Σ (KE of all other primary hadrons)

    This is a simple calorimetric estimator; no binding-energy correction.

    Parameters
    ----------
    reco_e_df : interaction-level df from get_primary_reco_electron
    evtdf     : full particle-level df (for summing hadronic KE)

    Returns
    -------
    reco_e_df with a new float column ('nu_energy_reco') in MeV.
    """
    il    = _il(evtdf)
    ke_c  = _find_col(evtdf, "ke",         branch_must_not_contain=BRANCH_TRUE)
    pid_c = _find_col(evtdf, "pid",        branch_must_not_contain=BRANCH_TRUE)
    pri_c = _find_col(evtdf, "is_primary", branch_must_not_contain=BRANCH_TRUE)

    # Sum KE of all primary particles (electron energy is already in reco_e_df)
    primary_mask = evtdf[pri_c] == 1
    primary_ke   = evtdf[primary_mask].groupby(level=il)[ke_c].sum()

    # Electron total energy = KE + m_e (0.511 MeV)
    e_ke_c = _find_col(evtdf, "ke", branch_must_not_contain=BRANCH_TRUE)
    reco_e_df = reco_e_df.copy()
    reco_e_df["nu_energy_reco"] = (
        primary_ke.reindex(reco_e_df.index, fill_value=0.0) + 0.511
    )
    return reco_e_df


def build_nu_energy_true(true_e_df, evtdf):
    """
    Append 'nu_energy_true' (truth) column.  Uses the mcnu energy if available,
    otherwise sums true primary particle energies.
    """
    # Try to get nu energy from the merged MC neutrino record
    nu_e_col = _safe(_find_col, evtdf, "nu_energy_true", branch_must_contain="mcnu")
    if nu_e_col is None:
        # Fall back: sum true primary particle KE + lepton mass
        il    = _il(evtdf)
        ke_c  = _find_col(evtdf, "ke",         branch_must_contain=BRANCH_TRUE)
        pri_c = _find_col(evtdf, "is_primary", branch_must_contain=BRANCH_TRUE)
        primary_ke = (
            evtdf[evtdf[pri_c] == 1].groupby(level=il)[ke_c].sum()
        )
        true_e_df = true_e_df.copy()
        true_e_df["nu_energy_true"] = (
            primary_ke.reindex(true_e_df.index, fill_value=0.0) + 0.511
        )
    else:
        il = _il(evtdf)
        inter = evtdf.groupby(level=il).first()
        true_e_df = true_e_df.copy()
        true_e_df["nu_energy_true"] = inter[nu_e_col].reindex(true_e_df.index)
    return true_e_df


# =============================================================================
# MC statistical uncertainty
# =============================================================================

def add_mcstat_weights(evtdf, n_universes=100, seed_base=42):
    """
    Append Poisson-resampled universe weight columns to *evtdf*.

    Each event gets a unique deterministic seed derived from its index values,
    ensuring reproducibility without requiring run/subrun/evt from hdr_df.

    Columns added
    -------------
    ('MCstat', 'univ_N', '', '', '') for N in range(n_universes)

    Returns
    -------
    evtdf with new MCstat columns appended.

    Notes
    -----
    The mean of Poisson(1) universes should equal 1 (verified in the notebook).
    Average deviation from 1 should be < 1/√N_universes ~ 1% for N=100.
    """
    rng = np.random.default_rng(seed_base)

    # Build deterministic per-event seeds from index tuple
    idx_arr  = np.array([hash(str(ix)) & 0xFFFFFFFF for ix in evtdf.index])
    univ_seeds = rng.integers(0, 2**31, size=n_universes)

    # Pre-allocate weights array  (shape: n_universes × n_events)
    weights = np.ones((n_universes, len(evtdf)), dtype=np.float32)

    for u, useed in enumerate(tqdm(univ_seeds, desc="MCstat universes", leave=False)):
        combined = (idx_arr + useed) % (2**32)
        # Vectorised Poisson sampling with per-event seeds
        weights[u] = np.array([
            np.random.default_rng(int(s)).poisson(1.0)
            for s in combined
        ], dtype=np.float32)

    # Build MultiIndex columns and attach
    mcstat_cols = pd.MultiIndex.from_tuples(
        [("MCstat", f"univ_{u}", "", "", "") for u in range(n_universes)]
    )
    wdf = pd.DataFrame(weights.T, index=evtdf.index, columns=mcstat_cols)
    return pd.concat([evtdf, wdf], axis=1)


def compute_mcstat_covariance(
    sel_signal_vals,      # 1-D array: reco variable values for selected signal
    sel_signal_weights,   # 1-D array: pot_weight for each selected signal event
    sel_bkg_dfs,          # list of (var_vals, pot_weights) for each bkg category
    var_cfg,              # VariableConfig
    evtdf_signal,         # slice of evtdf for selected signal (has MCstat cols)
    evtdf_bkg_list,       # list of evtdf slices for each bkg category
    n_universes=100,
    scale_factor=1.0,
    eps=1e-8,
):
    """
    Compute the MCstat covariance matrix for (signal – background) in reco space.

    Follows the universe-based approach: for each Poisson universe, compute
    the (signal + background variation) histogram and accumulate
        C_ij += (univ_i – cv_i)(univ_j – cv_j) / N_universes

    Parameters
    ----------
    sel_signal_vals      : reco variable for selected signal events (after clipping)
    sel_signal_weights   : POT weights for the selected signal events
    sel_bkg_dfs          : list of (bkg_vals, bkg_pot_weights) tuples
    var_cfg              : VariableConfig
    evtdf_signal         : evtdf rows for selected signal (contains MCstat cols)
    evtdf_bkg_list       : evtdf rows for each bkg category (contains MCstat cols)
    n_universes          : number of Poisson universes
    scale_factor         : XSEC_UNIT (or 1 for event-rate covariance)

    Returns
    -------
    dict with keys:
        Covariance      : absolute covariance matrix  (n_bins × n_bins)
        Covariance_Frac : fractional covariance matrix
        Correlation     : correlation matrix
        cv_events       : CV histogram (signal only, scaled)
        univ_events     : list of per-universe histograms
    """
    bins = var_cfg.bins
    eps_b = var_cfg.bins[0]
    eps_e = var_cfg.bins[-1] - eps

    # CV signal histogram
    signal_cv, _ = np.histogram(
        np.clip(sel_signal_vals, eps_b, eps_e),
        bins=bins,
        weights=sel_signal_weights,
    )
    signal_cv = signal_cv * scale_factor

    n_bins = len(bins) - 1
    Cov      = np.zeros((n_bins, n_bins))
    CovFrac  = np.zeros((n_bins, n_bins))
    univ_events = []

    for u in tqdm(range(n_universes), desc="MCstat Cov", leave=False):
        ucol = ("MCstat", f"univ_{u}", "", "", "")

        # Signal universe histogram
        sig_wgt  = evtdf_signal[ucol].values * sel_signal_weights
        sig_univ, _ = np.histogram(
            np.clip(sel_signal_vals, eps_b, eps_e),
            bins=bins, weights=sig_wgt,
        )
        sig_univ = sig_univ * scale_factor

        # Background universe contribution  Δbkg = bkg_univ – bkg_cv
        for (bkg_vals, bkg_pot_wgt), bkg_edf in zip(sel_bkg_dfs, evtdf_bkg_list):
            if bkg_edf is None or len(bkg_edf) == 0:
                continue
            bwgt = bkg_edf[ucol].fillna(1.0).values * bkg_pot_wgt
            bkg_univ, _ = np.histogram(
                np.clip(bkg_vals, eps_b, eps_e), bins=bins, weights=bwgt,
            )
            bkg_cv, _ = np.histogram(
                np.clip(bkg_vals, eps_b, eps_e), bins=bins, weights=bkg_pot_wgt,
            )
            sig_univ += (bkg_univ - bkg_cv) * scale_factor

        univ_events.append(sig_univ)

        dv = sig_univ - signal_cv
        with np.errstate(divide="ignore", invalid="ignore"):
            dvf = np.where(signal_cv != 0, dv / signal_cv, 0.0)
        Cov     += np.outer(dv,  dv)
        CovFrac += np.outer(dvf, dvf)

    Cov     /= n_universes
    CovFrac /= n_universes

    # Correlation
    diag = np.sqrt(np.diag(Cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        Corr = Cov / np.outer(np.where(diag > 0, diag, 1),
                               np.where(diag > 0, diag, 1))

    return dict(
        Covariance      = Cov,
        Covariance_Frac = CovFrac,
        Correlation     = Corr,
        cv_events       = signal_cv,
        univ_events     = univ_events,
    )


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_stacked_topology(
    evtdf_int,       # interaction-level df with 'nuint_categ' column
    var_vals,        # pd.Series of reco variable values (interaction-indexed)
    pot_weights,     # pd.Series of pot_weight (interaction-indexed)
    var_cfg,
    cat_order=None,
    title="",
    ylim_scale=1.3,
    data_vals=None,  # if provided, draw as black points (fake-data mode)
    data_label="Data",
    save_path=None,
):
    """
    Stacked histogram of nuint_categ truth categories vs. a reco variable.

    Parameters
    ----------
    evtdf_int   : interaction-level df containing 'nuint_categ'
    var_vals    : reco variable aligned with evtdf_int.index
    pot_weights : POT weight aligned with evtdf_int.index
    var_cfg     : VariableConfig
    cat_order   : list of category ints (bottom→top); defaults to CAT_PLOT_ORDER
    data_vals   : optional 1-D array (same binning) to draw as fake/real data
    """
    if cat_order is None:
        cat_order = CAT_PLOT_ORDER

    bins = var_cfg.bins
    eps  = 1e-8

    var_per_cat  = []
    wgt_per_cat  = []
    labels_used  = []
    colors_used  = []

    for cat in cat_order:
        mask = evtdf_int["nuint_categ"] == cat
        var_per_cat.append(
            np.clip(var_vals[mask].values, bins[0], bins[-1] - eps)
        )
        wgt_per_cat.append(pot_weights[mask].values)
        labels_used.append(CAT_LABELS.get(cat, str(cat)))
        colors_used.append(CAT_COLORS.get(cat, "black"))

    fig, ax = plt.subplots(figsize=(8, 6))
    mc_stack, _, _ = ax.hist(
        var_per_cat,
        bins=bins,
        weights=wgt_per_cat,
        stacked=True,
        color=colors_used,
        label=labels_used,
        edgecolor="none",
        histtype="stepfilled",
    )

    # Fraction labels in legend
    total_mc = np.array([np.sum(h) for h in mc_stack])
    total    = total_mc[-1] if len(total_mc) else 1.0
    indiv    = np.diff(np.concatenate([[0.], total_mc]))
    frac_labels = [
        f"{l} ({100*f/total:.1f}%)"
        for l, f in zip(labels_used[::-1], indiv[::-1])
    ]

    if data_vals is not None:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.errorbar(
            bin_centers, data_vals, yerr=np.sqrt(np.abs(data_vals)),
            fmt="ko", capsize=3, label=data_label,
        )
        frac_labels.append(data_label)

    ax.legend(frac_labels[::-1] if data_vals is not None else frac_labels,
              loc="upper left", fontsize=9, frameon=False, ncol=2)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(var_cfg.var_labels[1], fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_ylim(0, ylim_scale * (data_vals.max() if data_vals is not None
                                  else mc_stack[-1].max()))
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax, mc_stack


def plot_heatmap(
    matrix,
    title,
    var_cfg,
    axis_labels=None,    # [x_label, y_label] or None → use var_cfg
    fmt=".2f",
    cmap="viridis",
    save_path=None,
):
    """
    Coloured matrix heatmap with bin-range tick labels (cafpyana style).
    Uses bin_range_labels from analysis_village.unfolding.utils.
    """
    from analysis_village.unfolding.utils import bin_range_labels, get_text_color

    bins         = var_cfg.bins
    n            = len(bins) - 1
    unif_bin     = np.linspace(0, n, n + 1)
    extent       = [unif_bin[0], unif_bin[-1], unif_bin[0], unif_bin[-1]]
    tick_pos     = 0.5 * (unif_bin[:-1] + unif_bin[1:])
    tick_labels  = bin_range_labels(bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, extent=extent, origin="lower", cmap=cmap,
                   aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels)

    xlbl, ylbl = (axis_labels if axis_labels else [var_cfg.var_labels[2],
                                                    var_cfg.var_labels[1]])
    ax.set_xlabel(xlbl, fontsize=11)
    ax.set_ylabel(ylbl, fontsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(tick_pos[j], tick_pos[i], f"{v:{fmt}}",
                        ha="center", va="center",
                        color=get_text_color(v), fontsize=9)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_uncertainty_budget(
    var_cfg,
    cv_total,          # CV (signal + background), 1-D array
    cov_dict,          # dict {label: cov_matrix}
    title="Uncertainty Budget",
    save_path=None,
):
    """
    Fractional uncertainty per source as step-function lines.
    Total uncertainty drawn in black dashed.

    Parameters
    ----------
    cv_total   : signal + background histogram (denominator for fractional unc.)
    cov_dict   : {label: (n_bins × n_bins) covariance matrix}
    """
    bins   = var_cfg.bins
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 5))
    total_cov = np.zeros((len(cv_total), len(cv_total)))

    for i, (label, cov) in enumerate(cov_dict.items()):
        total_cov += cov
        unc  = np.sqrt(np.diag(cov))
        frac = np.divide(unc, cv_total, out=np.zeros_like(unc),
                         where=cv_total > 0)
        if np.all(frac == 0):
            continue
        ax.step(bins, np.append(frac, frac[-1]), where="post",
                label=label, lw=1.8, color=colors[i % len(colors)])

    tot_unc  = np.sqrt(np.diag(total_cov))
    tot_frac = np.divide(tot_unc, cv_total, out=np.zeros_like(tot_unc),
                         where=cv_total > 0)
    ax.step(bins, np.append(tot_frac, tot_frac[-1]), where="post",
            label="Total", color="black", ls="--", lw=2.5)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(var_cfg.var_labels[0], fontsize=12)
    ax.set_ylabel(r"Fractional unc. $\sigma / N$", fontsize=12)
    ax.set_title(f"{title}: {var_cfg.var_plot_name}", fontsize=13)
    ax.set_ylim(0, max(tot_frac.max() * 1.5, 0.05))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.grid(alpha=0.2, ls="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_unfolded_result(
    var_cfg,
    unfold_result,     # dict returned by WienerSVD
    model,             # 1-D: truth signal (scaled by XSEC_UNIT)
    reco_signal=None,  # optional: reco measured (for overlay)
    title="Unfolded Result",
    save_path=None,
):
    """
    Unfolded data points with error bars vs. Ac × truth model.
    Includes χ²/ndf annotation.
    """
    from scipy.stats import chi2 as scipy_chi2

    bins        = var_cfg.bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    unfold_val  = unfold_result["unfold"]
    unfold_cov  = unfold_result["UnfoldCov"]
    unfold_err  = np.sqrt(np.diag(unfold_cov))
    model_smear = unfold_result["AddSmear"] @ model

    chi2_val = float((unfold_val - model_smear) @
                     np.linalg.inv(unfold_cov) @
                     (unfold_val - model_smear))
    ndf   = len(unfold_val)
    pval  = 1 - scipy_chi2.cdf(chi2_val, ndf)

    fig, ax = plt.subplots(figsize=(8, 6))

    h_data = ax.errorbar(
        bin_centers, unfold_val, yerr=unfold_err,
        fmt="ko", capsize=3, elinewidth=1.5, markersize=5,
        label="Unfolded",
    )
    h_truth, = ax.step(
        bins, np.append(model_smear, model_smear[-1]),
        where="post", color="crimson", lw=2,
        label=r"$A_c \times$ Truth",
    )
    handles = [h_data, h_truth]
    labels  = [
        "Unfolded",
        r"$A_c \times$ Truth",
    ]

    if reco_signal is not None:
        h_reco, = ax.plot(bin_centers, reco_signal, "o",
                          color="steelblue", alpha=0.7, ms=5,
                          label="Reco. signal")
        handles.append(h_reco)
        labels.append("Reco. signal")

    chi2_str = (
        r"$\chi^2/\mathrm{ndf} = %.2f / %d$  $(p = %.2f)$"
        % (chi2_val, ndf, pval)
    )
    ax.text(0.05, 0.95, chi2_str, transform=ax.transAxes,
            fontsize=12, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", alpha=0.6, ec="none"))

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(var_cfg.var_labels[0], fontsize=12)
    ax.set_ylabel(r"Flux-avg. $\sigma$ [cm$^2$/Ar]", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(handles, labels, frameon=False, loc="upper right")
    ax.grid(alpha=0.2, ls="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# =============================================================================
# NPZ archiving  (for future systematic-weight studies — not yet called)
# =============================================================================

def save_cov_npz(
    filename,
    response=None,
    true_signal=None,
    measured_signal=None,
    measured_background=None,
    cov_ms_ms=None,
    cov_bs_bs=None,
    cov_ms_bs=None,
    cov_bs_ms=None,
    **extra,
):
    """
    Save all covariance and histogram arrays needed for Wiener-SVD unfolding
    into a compressed NumPy archive (.npz).

    Naming convention mirrors the cohpi analysis (ccpi0ana_sbnd.toml).

    Parameters
    ----------
    filename            : output path (e.g. 'mcstat_syst_cov_matrices.npz')
    response            : (n_reco × n_true) response matrix
    true_signal         : 1-D true signal histogram (XSEC_UNIT scaled)
    measured_signal     : 1-D reco signal histogram  (ms)
    measured_background : 1-D reco background histogram (bs)
    cov_ms_ms           : signal×signal covariance dict {'cov': array}
    cov_bs_bs           : background×background covariance dict
    cov_ms_bs           : signal×background covariance dict
    cov_bs_ms           : background×signal covariance dict
    **extra             : additional arrays saved verbatim

    Usage (future, when syst weights are available)
    -----------------------------------------------
    # In the covariance-computation stage:
    save_cov_npz(
        "mcstat_syst_cov_matrices.npz",
        response   = Response,
        true_signal = nevts_signal_truth * XSEC_UNIT,
        measured_signal = nevts_signal_sel_reco * XSEC_UNIT,
        measured_background = nevts_bkg_reco * XSEC_UNIT,
        cov_ms_ms  = {"cov": ret_mcstat["Covariance"]},
        cov_bs_bs  = {"cov": ret_mcstat_bkg["Covariance"]},
    )

    # In the unfolding notebook:
    cov_data = np.load("mcstat_syst_cov_matrices.npz", allow_pickle=True)
    cov_total = cov_data["cov_ms_ms"].item()["cov"] + ...
    """
    arrays = {}
    if response            is not None: arrays["response"]             = response
    if true_signal         is not None: arrays["true_signal"]          = true_signal
    if measured_signal     is not None: arrays["ms"]                   = measured_signal
    if measured_background is not None: arrays["bs"]                   = measured_background
    if cov_ms_ms           is not None: arrays["cov_ms_ms"]            = np.array(cov_ms_ms)
    if cov_bs_bs           is not None: arrays["cov_bs_bs"]            = np.array(cov_bs_bs)
    if cov_ms_bs           is not None: arrays["cov_ms_bs"]            = np.array(cov_ms_bs)
    if cov_bs_ms           is not None: arrays["cov_bs_ms"]            = np.array(cov_bs_ms)
    arrays.update(extra)
    np.savez_compressed(filename, **arrays)
    print(f"Saved {len(arrays)} arrays → {filename}")