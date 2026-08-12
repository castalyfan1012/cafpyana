"""
build_detsys_covariances.py
---------------------------
Build detector-variation covariance matrices for the nueCC analysis.

Method: unisim (shape-only, no matched CV available)
  Each wiremod variation is a single ±1σ universe.
  1. Load sel_topo pickle (flat columns: reco_ke, reco_costheta, etc.)
  2. Read det-var flat CAF → reco_ke, reco_costheta via direct branch access
  3. Shape-normalise det-var total to nominal total (isolate shape effect)
  4. Compute fractional shift → apply to sig and bkg separately
  5. Save NPZ compatible with the notebook's cov_results format

Inputs:
  - sel_topo pickle  (selected_nuecc_qual.pkl — from the syst notebook)
  - Det-var flat CAFs (Jacob's wiremod files, accessed via xrootd)

Outputs (to SYST_DIR/{var_name}/):
  - detsys_cov_matrices.npz   per-variable, loadable by the notebook

Usage:
  python3 build_detsys_covariances.py
  python3 build_detsys_covariances.py --vars wiremodxtheta
  python3 build_detsys_covariances.py --check-nominal
  python3 build_detsys_covariances.py --no-plots
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detsys_histdf import make_nuecc_detsys_seldf

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vars", nargs="+", default=None,
                    help="Which variations to run (default: all available)")
parser.add_argument("--no-plots", action="store_true")
parser.add_argument("--check-nominal", action="store_true",
                    help="Run nominal diagnostics and exit")
args = parser.parse_args()

# ── Paths — must match the syst notebook ──────────────────────────────────────
SEL_TOPO_PKL = (
    "/home/castalyf/cafpyana/analysis_village/nueCC/"
    "nuecc_dfs/selected_nuecc_qual.pkl"
)
SYST_DIR = "/home/castalyf/cafpyana/analysis_village/nueCC/saved_syst"
PLOT_DIR = "/home/castalyf/cafpyana/analysis_village/nueCC/plots_syst/detsys"
FINAL_STAGE = "sel_vertex_distance"

# ── Det-var flat CAF xrootd paths ─────────────────────────────────────────────
DET_VAR_CAFS = {
    "wiremodxtheta": (
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/"
        "users/jzettle/sbnd_wiremod_x_thetaxw/"
        "combined_wiremod_x_thetaxw.flat.caf.root"
    ),
    "wiremodyz": (
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/"
        "users/jzettle/sbnd_wiremod_y_z/"
        "combined_wiremod_y_z.flat.caf.root"
    ),
}

vars_to_run = args.vars or list(DET_VAR_CAFS.keys())

# ── Binning — MUST match nue_syst_v2.ipynb Cell 4 exactly ────────────────────
RECO_BINS = np.linspace(0, 2000, 8)   # 7 bins for reco_ke
COS_BINS  = np.linspace(-1, 1, 11)    # 10 bins for reco_costheta

VARIABLES = {
    "reco_ke":       {"bins": RECO_BINS, "xlabel": r"Reco leading-$e^-$ KE [MeV]"},
    "reco_costheta": {"bins": COS_BINS,  "xlabel": r"Reco leading-$e^-$ $\cos\theta$"},
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Load nominal from sel_topo ────────────────────────────────────────────────

def load_nominal():
    print(f"Loading nominal: {SEL_TOPO_PKL}")
    sel = pd.read_pickle(SEL_TOPO_PKL)
    print(f"  Shape: {sel.shape}, Columns: {list(sel.columns)}")
    return sel


def check_nominal(sel):
    ps = float(sel["pot_scale"].iloc[0])
    print(f"\n── Nominal diagnostics ──")
    print(f"  pot_scale         : {ps:.4f}")
    print(f"  n_total           : {len(sel):,}")
    print(f"  n_passing final   : {sel[FINAL_STAGE].sum():,}")
    print(f"  n_signal passing  : {(sel['is_sig'] & sel[FINAL_STAGE]).sum():,}")
    print(f"  n_bkg passing     : {(sel[FINAL_STAGE] & sel['truth_cat'].between(1,6)).sum():,}")
    passed = sel[sel[FINAL_STAGE]]
    for var, cfg in VARIABLES.items():
        v = passed[var].dropna()
        bins = cfg["bins"]
        h_all, _ = np.histogram(v.clip(bins[0], bins[-1]-1e-8), bins=bins,
                                weights=np.full(len(v), ps))
        print(f"\n  {var} (n={len(v):,}, weighted total={h_all.sum():.1f}):")
        centers = 0.5 * (bins[:-1] + bins[1:])
        for c, h in zip(centers, h_all):
            print(f"    {c:8.1f}: {h:8.1f}")


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
        total_cv = sig_cv + bkg_cv
        hists[var] = {"sig_cv": sig_cv, "bkg_cv": bkg_cv, "total_cv": total_cv}
        print(f"  Nominal {var}: sig={sig_cv.sum():.1f}  bkg={bkg_cv.sum():.1f}  "
              f"total={total_cv.sum():.1f}")
    return hists


# ── Det-var processing ────────────────────────────────────────────────────────

def detvar_total_hist(seldf, var, bins, nom_total):
    eps = 1e-8
    vals = seldf[var].clip(bins[0], bins[-1] - eps).dropna()
    h, _ = np.histogram(vals, bins=bins)
    h = h.astype(float)
    if h.sum() > 0 and nom_total.sum() > 0:
        h = h * (nom_total.sum() / h.sum())
    return h


def unisim_cov_from_frac(frac_shift, cv_arr):
    delta = frac_shift * cv_arr
    return np.outer(delta, delta)


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
    axes[1].axhline( 0.05, color="gray", linewidth=0.8, linestyle=":")
    axes[1].axhline(-0.05, color="gray", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("(var - nom) / nom")
    axes[1].set_ylim(-0.3, 0.3)
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

    pot_scale = float(sel["pot_scale"].iloc[0])
    print(f"  pot_scale = {pot_scale:.4f}")
    nom = nominal_hists(sel, pot_scale)

    # Accumulate total det-sys covariance per variable
    total_cov = {}
    for var in VARIABLES:
        nb = len(VARIABLES[var]["bins"]) - 1
        total_cov[var] = {k: np.zeros((nb, nb))
                          for k in ["cov_ms_ms","cov_bs_bs","cov_ms_bs","cov_bs_ms"]}

    for det_var in vars_to_run:
        if det_var not in DET_VAR_CAFS:
            print(f"\nSkipping {det_var}: no CAF path"); continue

        print(f"\n{'='*60}\nProcessing: {det_var}\n{'='*60}")
        try:
            seldf = make_nuecc_detsys_seldf(DET_VAR_CAFS[det_var], verbose=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if seldf.empty:
            print(f"  Empty df for {det_var}, skipping"); continue

        for var, cfg in VARIABLES.items():
            bins = cfg["bins"]
            nh = nom[var]

            h_var = detvar_total_hist(seldf, var, bins, nh["total_cv"])
            frac_shift = (h_var - nh["total_cv"]) / nh["total_cv"].clip(1e-6)

            # Apply same fractional shift to signal and bkg separately
            delta_s = frac_shift * nh["sig_cv"]
            delta_b = frac_shift * nh["bkg_cv"]

            total_cov[var]["cov_ms_ms"] += np.outer(delta_s, delta_s)
            total_cov[var]["cov_bs_bs"] += np.outer(delta_b, delta_b)
            total_cov[var]["cov_ms_bs"] += np.outer(delta_s, delta_b)
            total_cov[var]["cov_bs_ms"] += np.outer(delta_b, delta_s)

            centers = 0.5 * (bins[:-1] + bins[1:])
            print(f"\n  {var} fractional shift ({det_var}):")
            for c, f in zip(centers, frac_shift):
                bar = ("+" if f >= 0 else "-") * min(int(abs(f)*30), 30)
                print(f"    {c:8.1f}: {f:+.4f}  {bar}")

            if not args.no_plots:
                plot_comparison(var, nh["total_cv"], h_var, det_var,
                                os.path.join(PLOT_DIR, f"detsys_{det_var}_{var}.png"),
                                cfg["xlabel"])

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSaving det-sys covariance matrices\n{'='*60}")
    for var, cfg in VARIABLES.items():
        nh = nom[var]; tc = total_cov[var]; bins = cfg["bins"]
        diag_sig = np.sqrt(np.diag(tc["cov_ms_ms"]).clip(0))
        frac_sig = diag_sig / nh["sig_cv"].clip(1e-6)
        diag_bkg = np.sqrt(np.diag(tc["cov_bs_bs"]).clip(0))
        frac_bkg = diag_bkg / nh["bkg_cv"].clip(1e-6)

        centers = 0.5 * (bins[:-1] + bins[1:])
        print(f"\n  {var} det-sys fractional uncertainty:")
        print(f"    {'bin':>10}  {'signal':>8}  {'bkg':>8}")
        for c, fs, fb in zip(centers, frac_sig, frac_bkg):
            print(f"    {c:10.1f}  {fs:7.4f}  {fb:7.4f}")

        out_dir = os.path.join(SYST_DIR, var)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detsys_cov_matrices.npz")
        np.savez(out_path,
                 ms=nh["sig_cv"], bs=nh["bkg_cv"],
                 cov_ms_ms=tc["cov_ms_ms"], cov_bs_bs=tc["cov_bs_bs"],
                 cov_ms_bs=tc["cov_ms_bs"], cov_bs_ms=tc["cov_bs_ms"],
                 bins=bins, frac_unc_sig=frac_sig, frac_unc_bkg=frac_bkg,
                 variations=np.array(vars_to_run))
        print(f"  Saved: {out_path}")

    print("\nDone. See notebook integration instructions below.")
    print("""
Add after Cell 8 in nue_syst_v2.ipynb:

    # ── Load det-sys covariance ──
    for vcfg in VAR_CFGS:
        detsys_path = f'{SYST_DIR}/{vcfg.name}/detsys_cov_matrices.npz'
        if os.path.exists(detsys_path):
            d = np.load(detsys_path)
            cov_results[vcfg.name]['detsys'] = {
                'cov_ms_ms': d['cov_ms_ms'],
                'cov_bs_bs': d['cov_bs_bs'],
                'cov_ms_bs': d['cov_ms_bs'],
                'cov_bs_ms': d['cov_bs_ms'],
            }
            print(f'  [{vcfg.name}] detsys: mean sig frac unc = '
                  f'{d["frac_unc_sig"].mean()*100:.2f}%')

    SRC_LABEL['detsys'] = 'Detector'
    COLORS['detsys']    = '#8B0000'
""")


if __name__ == "__main__":
    main()