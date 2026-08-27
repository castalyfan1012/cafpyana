"""
build_detsys_covariances.py
---------------------------
Build detector-variation covariance matrices for the nueCC analysis.

This version reads from .df files produced by run_df_maker.py with
nueCC_detvar.py config (data-mode loader). It applies the SAME selection
code (nue_selection.shower_qual_cuts) as the nominal MC and data, ensuring
identical cuts.

Method: unisim (shape-only, no matched CV)
  1. Load sel_topo pickle for nominal histograms
  2. Load det-var .df files, apply shower_qual_cuts, extract reco observables
  3. Shape-normalise det-var to nominal, compute covariance
  4. Save NPZ compatible with the notebook's cov_results format

Usage:
  python3 build_detsys_covariances.py
  python3 build_detsys_covariances.py --vars wiremodxtheta
  python3 build_detsys_covariances.py --check-nominal
  python3 build_detsys_covariances.py --check-detvar wiremodxtheta
"""

import os, sys, argparse, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Must be importable from the nueCC analysis directory ──────────────────────
NUECC_DIR = "/home/castalyf/cafpyana/analysis_village/nueCC"
for p in [NUECC_DIR, "/home/castalyf/cafpyana", "/home/castalyf/cafpyana/pyanalib"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import nue_selection as nue_sel
import nue_helpers as nh

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vars", nargs="+", default=None)
parser.add_argument("--no-plots", action="store_true")
parser.add_argument("--check-nominal", action="store_true")
parser.add_argument("--check-detvar", type=str, default=None,
                    help="Run diagnostics on a det-var .df file and exit")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
SEL_TOPO_PKL = f"{NUECC_DIR}/nuecc_dfs/selected_nuecc_qual.pkl"
SYST_DIR     = f"{NUECC_DIR}/saved_syst"
PLOT_DIR     = f"{NUECC_DIR}/plots_syst/detsys"
FINAL_STAGE  = "sel_vertex_distance"
FINAL_CUT    = "vertex_distance"       # key in shower_qual_cuts output

# ── Det-var .df file paths (produced by run_df_maker.py + nueCC_detvar.py) ───
DETVAR_DF_BASE = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/detvar/detvar_dfs"
DETVAR_DFS = {
    "wiremodxtheta": f"{DETVAR_DF_BASE}/wiremodxtheta/detvar_wiremodxtheta.df",
    "wiremodyz":     f"{DETVAR_DF_BASE}/wiremodyz/detvar_wiremodyz.df",
}
vars_to_run = args.vars or list(DETVAR_DFS.keys())

# ── Binning — MUST match nue_syst_v2.ipynb Cell 4 ────────────────────────────
RECO_BINS = np.linspace(0, 2000, 8)   # 7 bins for reco_ke
COS_BINS  = np.linspace(-1, 1, 11)    # 10 bins for reco_costheta
_m_e      = 0.511  # MeV/c²

VARIABLES = {
    "reco_ke":       {"bins": RECO_BINS, "xlabel": r"Reco leading-$e^-$ KE [MeV]"},
    "reco_costheta": {"bins": COS_BINS,  "xlabel": r"Reco leading-$e^-$ $\cos\theta$"},
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Load and select from a det-var .df file ───────────────────────────────────

def load_detvar_and_select(df_path, verbose=True):
    """
    Load a det-var .df (data-mode), apply shower_qual_cuts, extract
    reco_ke and reco_costheta for the leading electron at each cut stage.

    Uses the SAME selection code as the nominal MC and data notebooks.

    Returns
    -------
    dict with 'reco_ke', 'reco_costheta', 'reco_p' arrays at final selection,
    plus 'cut_flow' dict and 'evtdf' for diagnostics.
    """
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Det-var .df not found: {df_path}")

    if verbose:
        print(f"  Loading: {df_path}")

    evtdf = pd.read_hdf(df_path, key="evt_0")
    if verbose:
        print(f"  Shape: {evtdf.shape}")
        print(f"  Index: {evtdf.index.names} (nlevels={evtdf.index.nlevels})")

    # Apply the SAME selection as the nominal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut_flow = nue_sel.shower_qual_cuts(evtdf)

    il = list(range(evtdf.index.nlevels - 1))

    if verbose:
        print(f"  Cut flow:")
        for stage, d in cut_flow.items():
            print(f"    {stage:<30s}: {len(d['inter_index']):>8,}")

    # Extract reco observables at final stage
    final_idx = cut_flow[FINAL_CUT]["inter_index"]

    if len(final_idx) == 0:
        if verbose:
            print(f"  No events at final stage — returning empty")
        return {
            "reco_ke": np.array([]),
            "reco_costheta": np.array([]),
            "reco_p": np.array([]),
            "cut_flow": cut_flow,
            "n_final": 0,
        }

    # Use the same extraction as the notebook's _extract_reco
    rp = evtdf["rec"]["dlp"]["particles"]
    pid_c = ('pid', '') if ('pid', '') in rp.columns else 'pid'
    cal_c = ('calo_ke', '') if ('calo_ke', '') in rp.columns else None
    ke_c  = ('ke', '') if ('ke', '') in rp.columns else 'ke'

    # Select electron candidates at final stage
    ele = rp[rp.index.droplevel(-1).isin(final_idx) & (rp[pid_c] == 1)]
    if ele.empty:
        if verbose:
            print(f"  No electron particles at final stage")
        return {
            "reco_ke": np.array([]),
            "reco_costheta": np.array([]),
            "reco_p": np.array([]),
            "cut_flow": cut_flow,
            "n_final": 0,
        }

    # Leading electron by KE
    ke_src = ele[cal_c] if cal_c is not None and cal_c in ele.columns else ele[ke_c]
    lead_idx = ke_src.groupby(level=il).idxmax().dropna()

    if lead_idx.empty:
        return {
            "reco_ke": np.array([]),
            "reco_costheta": np.array([]),
            "reco_p": np.array([]),
            "cut_flow": cut_flow,
            "n_final": 0,
        }

    lmi = pd.MultiIndex.from_tuples(lead_idx.values, names=ele.index.names)
    ld = ele.loc[lmi]
    ld.index = lead_idx.index

    reco_ke = ld[cal_c if cal_c is not None and cal_c in ld.columns else ke_c].dropna()
    reco_p  = np.sqrt(((reco_ke + _m_e)**2 - _m_e**2).clip(0))

    # cos(theta) from start_dir
    try:
        dx = [c for c in rp.columns if 'start_dir' in str(c) and 'I0' in str(c)][0]
        dy = [c for c in rp.columns if 'start_dir' in str(c) and 'I1' in str(c)][0]
        dz = [c for c in rp.columns if 'start_dir' in str(c) and 'I2' in str(c)][0]
        dd = ele.loc[lmi, [dx, dy, dz]]
        dd.index = lead_idx.index
        pm = np.sqrt(dd[dx]**2 + dd[dy]**2 + dd[dz]**2).replace(0, np.nan)
        reco_cos = (dd[dz] / pm).dropna()
    except (IndexError, KeyError):
        reco_cos = pd.Series(np.nan, index=reco_ke.index)

    result = {
        "reco_ke":       reco_ke.values.astype(float),
        "reco_costheta": reco_cos.values.astype(float),
        "reco_p":        reco_p.values.astype(float),
        "cut_flow":      cut_flow,
        "n_final":       len(final_idx),
    }

    if verbose:
        print(f"  Final: {result['n_final']} interactions, "
              f"{len(result['reco_ke'])} electrons")
        if len(result['reco_ke']) > 0:
            print(f"  reco_ke  : {result['reco_ke'].mean():.1f} ± "
                  f"{result['reco_ke'].std():.1f} MeV")
            print(f"  reco_cos : {result['reco_costheta'].mean():.3f} ± "
                  f"{result['reco_costheta'].std():.3f}")

    del evtdf
    gc.collect()
    return result


# ── Nominal loading ──────────────────────────────────────────────────────────

def load_nominal():
    print(f"Loading nominal: {SEL_TOPO_PKL}")
    sel = pd.read_pickle(SEL_TOPO_PKL)
    print(f"  Shape: {sel.shape}")
    return sel


def check_nominal(sel):
    ps = float(sel["pot_scale"].iloc[0])
    print(f"\n── Nominal diagnostics ──")
    print(f"  pot_scale        : {ps:.4f}")
    print(f"  n_passing final  : {sel[FINAL_STAGE].sum():,}")
    print(f"  n_signal passing : {(sel['is_sig'] & sel[FINAL_STAGE]).sum():,}")
    print(f"  n_bkg passing    : {(sel[FINAL_STAGE] & sel['truth_cat'].between(1,6)).sum():,}")
    passed = sel[sel[FINAL_STAGE]]
    for var, cfg in VARIABLES.items():
        v = passed[var].dropna()
        print(f"\n  {var}: n={len(v):,}  mean={v.mean():.1f}  std={v.std():.1f}")


def nominal_hists(sel, pot_scale):
    passed = sel[sel[FINAL_STAGE]]
    sig_mask = passed["is_sig"]
    bkg_mask = passed["truth_cat"].between(1, 6)
    eps = 1e-8
    hists = {}
    for var, cfg in VARIABLES.items():
        bins = cfg["bins"]
        vals = passed[var].clip(bins[0], bins[-1] - eps)
        sig_cv, _ = np.histogram(vals[sig_mask].dropna(), bins=bins,
                                 weights=np.full(sig_mask.sum(), pot_scale))
        bkg_cv, _ = np.histogram(vals[bkg_mask].dropna(), bins=bins,
                                 weights=np.full(bkg_mask.sum(), pot_scale))
        hists[var] = {"sig_cv": sig_cv, "bkg_cv": bkg_cv, "total_cv": sig_cv + bkg_cv}
        print(f"  Nominal {var}: sig={sig_cv.sum():.1f}  bkg={bkg_cv.sum():.1f}")
    return hists


# ── Histogram + covariance ───────────────────────────────────────────────────

def detvar_hist(vals, bins, nom_total):
    eps = 1e-8
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    v = np.clip(v, bins[0], bins[-1] - eps)
    h, _ = np.histogram(v, bins=bins)
    h = h.astype(float)
    if h.sum() > 0 and nom_total.sum() > 0:
        h = h * (nom_total.sum() / h.sum())
    return h


def plot_comparison(var, h_nom, h_var, det_var, save_path, xlabel):
    bins = VARIABLES[var]["bins"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].step(bins, np.append(h_nom, h_nom[-1]),
                 where="post", color="black", linewidth=2, label="Nominal MC")
    axes[0].step(bins, np.append(h_var, h_var[-1]),
                 where="post", color="red", linewidth=1.5,
                 linestyle="--", label=f"Det-var: {det_var}")
    axes[0].set_ylabel("Weighted events")
    axes[0].legend()
    axes[0].set_title(f"Det-sys comparison: {var} — {det_var}")
    frac = (h_var - h_nom) / np.where(h_nom > 0, h_nom, 1)
    axes[1].step(bins, np.append(frac, frac[-1]),
                 where="post", color="red", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=1)
    for y in [0.05, -0.05]:
        axes[1].axhline(y, color="gray", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("(var - nom) / nom")
    axes[1].set_ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sel = load_nominal()

    if args.check_nominal:
        check_nominal(sel)
        return

    if args.check_detvar:
        var_name = args.check_detvar
        if var_name not in DETVAR_DFS:
            print(f"Unknown variation: {var_name}")
            print(f"Available: {list(DETVAR_DFS.keys())}")
            return
        print(f"\n── Det-var diagnostics: {var_name} ──")
        result = load_detvar_and_select(DETVAR_DFS[var_name], verbose=True)
        return

    pot_scale = float(sel["pot_scale"].iloc[0])
    print(f"  pot_scale = {pot_scale:.4f}")
    nom = nominal_hists(sel, pot_scale)

    total_cov = {}
    for var in VARIABLES:
        nb = len(VARIABLES[var]["bins"]) - 1
        total_cov[var] = {k: np.zeros((nb, nb))
                          for k in ["cov_ms_ms", "cov_bs_bs", "cov_ms_bs", "cov_bs_ms"]}

    for det_var in vars_to_run:
        if det_var not in DETVAR_DFS:
            print(f"\nSkipping {det_var}: no .df path defined")
            continue

        print(f"\n{'='*60}\nProcessing: {det_var}\n{'='*60}")
        try:
            result = load_detvar_and_select(DETVAR_DFS[det_var], verbose=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if result["n_final"] == 0:
            print(f"  No events for {det_var}, skipping")
            continue

        for var, cfg in VARIABLES.items():
            bins = cfg["bins"]
            nh_var = nom[var]
            vals = result[var]

            if len(vals) == 0:
                continue

            h_var = detvar_hist(vals, bins, nh_var["total_cv"])
            frac_shift = (h_var - nh_var["total_cv"]) / nh_var["total_cv"].clip(1e-6)

            delta_s = frac_shift * nh_var["sig_cv"]
            delta_b = frac_shift * nh_var["bkg_cv"]
            total_cov[var]["cov_ms_ms"] += np.outer(delta_s, delta_s)
            total_cov[var]["cov_bs_bs"] += np.outer(delta_b, delta_b)
            total_cov[var]["cov_ms_bs"] += np.outer(delta_s, delta_b)
            total_cov[var]["cov_bs_ms"] += np.outer(delta_b, delta_s)

            centers = 0.5 * (bins[:-1] + bins[1:])
            print(f"\n  {var} fractional shift ({det_var}):")
            for c, f in zip(centers, frac_shift):
                bar = ("+" if f >= 0 else "-") * min(int(abs(f) * 30), 30)
                print(f"    {c:8.1f}: {f:+.4f}  {bar}")

            if not args.no_plots:
                plot_comparison(var, nh_var["total_cv"], h_var, det_var,
                                os.path.join(PLOT_DIR, f"detsys_{det_var}_{var}.png"),
                                cfg["xlabel"])

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSaving\n{'='*60}")
    for var, cfg in VARIABLES.items():
        nh_var = nom[var]
        tc = total_cov[var]
        bins = cfg["bins"]

        diag_sig = np.sqrt(np.diag(tc["cov_ms_ms"]).clip(0))
        frac_sig = diag_sig / nh_var["sig_cv"].clip(1e-6)
        diag_bkg = np.sqrt(np.diag(tc["cov_bs_bs"]).clip(0))
        frac_bkg = diag_bkg / nh_var["bkg_cv"].clip(1e-6)

        centers = 0.5 * (bins[:-1] + bins[1:])
        print(f"\n  {var} det-sys fractional uncertainty:")
        print(f"    {'bin':>10}  {'signal':>8}  {'bkg':>8}")
        for c, fs, fb in zip(centers, frac_sig, frac_bkg):
            print(f"    {c:10.1f}  {fs:7.4f}  {fb:7.4f}")

        out_dir = os.path.join(SYST_DIR, var)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detsys_cov_matrices.npz")
        np.savez(out_path,
                 ms=nh_var["sig_cv"], bs=nh_var["bkg_cv"],
                 cov_ms_ms=tc["cov_ms_ms"], cov_bs_bs=tc["cov_bs_bs"],
                 cov_ms_bs=tc["cov_ms_bs"], cov_bs_ms=tc["cov_bs_ms"],
                 bins=bins, frac_unc_sig=frac_sig, frac_unc_bkg=frac_bkg,
                 variations=np.array(vars_to_run))
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()