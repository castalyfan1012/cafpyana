"""
build_detsys_covariances.py
---------------------------
Build detector-variation covariance matrices for the nueCC histogram study.

Method: unisim
  Each variation is a single ±1σ universe.
  Cov[i,j] = (N_var[i] - N_nom[i]) * (N_var[j] - N_nom[j])
  Total det-sys covariance = sum over all variations.

Inputs:
  - Nominal selected MC df  (mc1e20_nueCC_v2.df, already exists)
  - Det-var flat CAFs       (read via make_nuecc_detsys_seldf)

Outputs (to config.save_dir):
  - detsys_cov_{var}_{variable}.npz    per-variation covariance
  - detsys_cov_total_{variable}.npz    summed covariance → load in notebook

Usage:
  python3 build_detsys_covariances.py [--vars wiremodxtheta wiremodyz]
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/castalyf/cafpyana/analysis_village/nueCC/detvar")
from make_detsys_histdf import make_nuecc_detsys_seldf
from nuecc_detsys_config import NueCCDetsysConfig, DET_VARS_ALL

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vars", nargs="+", default=None,
                    help="Which variations to run (default: all in DET_VAR_CAFS)")
parser.add_argument("--final-stage", default="sel_vertex_distance",
                    help="Nominal selection cut stage column name")
parser.add_argument("--no-plots", action="store_true")
args = parser.parse_args()

cfg = NueCCDetsysConfig()

# ── Det-var CAF paths — extend when PDS/SCE files become available ────────────
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
    # "pmtgain": "root://...",
    # "pmtqe":   "root://...",
    # "nosce":   "root://...",
}

vars_to_run = args.vars or list(DET_VAR_CAFS.keys())

# ── Binning — must match nue_syst_v2.ipynb ────────────────────────────────────
BINS = {
    "reco_p":        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    "reco_costheta": np.linspace(-1, 1, 11),
}
XLABEL = {
    "reco_p":        "Reco electron momentum [MeV/c]",
    "reco_costheta": r"Reco $\cos\theta_e$",
}
VARIABLES = list(BINS.keys())

os.makedirs(cfg.save_dir,  exist_ok=True)
os.makedirs(cfg.plot_dir,  exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_nominal(final_stage):
    print(f"Loading nominal MC: {cfg.nominal_df_file}")
    df = pd.read_hdf(cfg.nominal_df_file, key="evt_0")
    if final_stage in df.columns:
        sel = df[df["is_sig"] & df[final_stage]]
    else:
        warnings.warn(f"'{final_stage}' not found — using is_sig only")
        sel = df[df["is_sig"]]
    print(f"  Nominal selected: {len(sel):,} signal events")
    return sel


def nominal_hists(sel, pot_scale):
    """POT-weighted nominal histograms."""
    hists = {}
    w = sel.get("pot_scale", pd.Series(pot_scale, index=sel.index))
    for var in VARIABLES:
        if var not in sel.columns:
            print(f"  WARNING: '{var}' not in nominal df"); continue
        h, _ = np.histogram(sel[var].dropna(), bins=BINS[var], weights=w)
        hists[var] = h
        print(f"  Nominal {var}: {h.sum():.1f} weighted events across bins")
    return hists


def detvar_hists(seldf, nom_hists):
    """
    Raw (unweighted) det-var histograms, then shape-normalised to match
    nominal total so we isolate the SHAPE effect of the detector variation.
    POT normalisation is replaced with actual ratio once Jacob confirms exposure.
    """
    hists = {}
    for var in VARIABLES:
        if var not in seldf.columns: continue
        h, _ = np.histogram(seldf[var].dropna(), bins=BINS[var])
        h = h.astype(float)
        # Shape normalisation
        if h.sum() > 0 and nom_hists[var].sum() > 0:
            h = h * (nom_hists[var].sum() / h.sum())
        hists[var] = h
    return hists


def unisim_cov(h_nom, h_var):
    delta = h_var - h_nom
    return np.outer(delta, delta)


def plot_comparison(var, h_nom, h_var, det_var, save_path):
    bins = BINS[var]
    centers = 0.5 * (bins[:-1] + bins[1:])
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

    ratio = h_var / np.where(h_nom > 0, h_nom, 1)
    delta = h_var - h_nom
    frac  = delta / np.where(h_nom > 0, h_nom, 1)
    axes[1].step(bins, np.append(frac, frac[-1]),
                 where="post", color="red", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline( 0.05, color="gray", linewidth=0.8, linestyle=":")
    axes[1].axhline(-0.05, color="gray", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel(XLABEL[var])
    axes[1].set_ylabel("(var - nom) / nom")
    axes[1].set_ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sel_nom  = load_nominal(args.final_stage)
    pot_scale = float(sel_nom["pot_scale"].iloc[0]) \
                if "pot_scale" in sel_nom.columns else 1.0
    print(f"  pot_scale = {pot_scale:.6f}")
    nom_hist = nominal_hists(sel_nom, pot_scale)

    total_cov = {var: np.zeros((len(BINS[var])-1, len(BINS[var])-1))
                 for var in VARIABLES}

    for det_var in vars_to_run:
        if det_var not in DET_VAR_CAFS:
            print(f"\nSkipping {det_var}: no CAF path defined"); continue

        print(f"\n{'='*60}\nProcessing: {det_var}\n{'='*60}")
        try:
            seldf = make_nuecc_detsys_seldf(DET_VAR_CAFS[det_var], verbose=True)
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        if seldf.empty:
            print(f"  Empty df for {det_var}, skipping"); continue

        dv_hist = detvar_hists(seldf, nom_hist)

        for var in VARIABLES:
            if var not in nom_hist or var not in dv_hist: continue
            h_nom = nom_hist[var]
            h_var = dv_hist[var]
            cov   = unisim_cov(h_nom, h_var)
            total_cov[var] += cov

            # Per-bin fractional shift
            bins    = BINS[var]
            centers = 0.5 * (bins[:-1] + bins[1:])
            frac    = (h_var - h_nom) / h_nom.clip(1e-6)
            print(f"\n  {var} fractional shift (det-var − nominal) / nominal:")
            for c, f in zip(centers, frac):
                bar = ("+" if f >= 0 else "-") * min(int(abs(f)*30), 30)
                print(f"    {c:8.1f}: {f:+.4f}  {bar}")

            # Save per-variation covariance
            out = f"{cfg.save_dir}/detsys_cov_{det_var}_{var}.npz"
            np.savez(out, cov=cov, h_nom=h_nom, h_var=h_var, bins=bins)
            print(f"  Saved: {out}")

            # Comparison plot
            if not args.no_plots:
                plot_comparison(
                    var, h_nom, h_var, det_var,
                    f"{cfg.plot_dir}/detsys_compare_{det_var}_{var}.png"
                )

    # ── Save total covariance ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nTotal det-sys covariance\n{'='*60}")
    for var in VARIABLES:
        cov  = total_cov[var]
        bins = BINS[var]
        diag_unc = np.sqrt(np.diag(cov))
        frac_unc = diag_unc / nom_hist[var].clip(1e-6)
        centers  = 0.5 * (bins[:-1] + bins[1:])

        print(f"\n  {var} total fractional det-sys uncertainty per bin:")
        for c, f in zip(centers, frac_unc):
            print(f"    {c:8.1f}: {f:.4f} ({f*100:.2f}%)")

        # Correlation matrix
        diag = np.sqrt(np.diag(cov))
        corr = cov / np.outer(diag.clip(1e-10), diag.clip(1e-10))

        out = f"{cfg.save_dir}/detsys_cov_total_{var}.npz"
        np.savez(out,
                 cov=cov, corr=corr, frac_unc=frac_unc,
                 h_nom=nom_hist[var], bins=bins,
                 variations=np.array(vars_to_run))
        print(f"  Saved: {out}")

    print(f"\nDone. Load in notebook:")
    print(f"  np.load('{cfg.save_dir}/detsys_cov_total_reco_p.npz')")


if __name__ == "__main__":
    main()