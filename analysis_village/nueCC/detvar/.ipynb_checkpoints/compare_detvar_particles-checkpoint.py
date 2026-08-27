"""
compare_detvar_particles.py
---------------------------
Compare particle-level distributions between nominal MC and det-var
BEFORE any selection cuts. This reveals whether the difference is in
reconstruction (SPINE) or CAF production (standalone vs merge).

Usage:
  python3 compare_detvar_particles.py
"""
import sys, warnings
import numpy as np
import pandas as pd

NUECC_DIR = "/home/castalyf/cafpyana/analysis_village/nueCC"
for p in [NUECC_DIR, "/home/castalyf/cafpyana", "/home/castalyf/cafpyana/pyanalib"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import nue_helpers as nh

# ── Files ─────────────────────────────────────────────────────────────────────
NOMINAL_DF = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_nueCC_v2-2.df"
DETVAR_DFS = {
    "wiremodxtheta": "/exp/sbnd/data/users/castalyf/nue_sel/production_files/detvar/detvar_dfs/wiremodxtheta/detvar_wiremodxtheta.df",
    "wiremodyz":     "/exp/sbnd/data/users/castalyf/nue_sel/production_files/detvar/detvar_dfs/wiremodyz/detvar_wiremodyz.df",
}


def particle_stats(evtdf, label, max_inter=50000):
    """Print particle-level statistics before any selection."""
    il = list(range(evtdf.index.nlevels - 1))
    all_inter = evtdf.groupby(level=il).size().index

    # Subsample if too large
    if len(all_inter) > max_inter:
        all_inter = all_inter[:max_inter]
        evtdf = evtdf[evtdf.index.droplevel(-1).isin(all_inter)]

    rp = nh.reco_particles(evtdf)
    rp_df = rp._df

    pid_col = ('pid', '') if ('pid', '') in rp_df.columns else 'pid'
    ke_col  = ('ke', '') if ('ke', '') in rp_df.columns else 'ke'
    pri_col = ('is_primary', '') if ('is_primary', '') in rp_df.columns else 'is_primary'
    cal_col = ('calo_ke', '') if ('calo_ke', '') in rp_df.columns else None

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Interactions: {len(all_inter):,}")
    print(f"  Particles:    {len(rp_df):,}")
    print(f"  Particles/interaction: {len(rp_df)/len(all_inter):.1f}")

    # PID distribution
    pids = rp_df[pid_col]
    print(f"\n  PID distribution:")
    for pid_val in sorted(pids.unique()):
        n = (pids == pid_val).sum()
        print(f"    pid={int(pid_val)}: {n:>8,} ({100*n/len(pids):.1f}%)")

    # is_primary
    pri = rp_df[pri_col]
    n_primary = (pri == 1).sum()
    print(f"\n  is_primary==1: {n_primary:,} ({100*n_primary/len(rp_df):.1f}%)")

    # Primary electrons
    ele_primary = (pids == 1) & (pri == 1)
    n_ep = ele_primary.sum()
    n_inter_with_ep = ele_primary.groupby(level=il).any().sum()
    print(f"  Primary electrons (pid==1 & is_primary==1): {n_ep:,}")
    print(f"  Interactions with >=1 primary electron: {n_inter_with_ep:,} "
          f"({100*n_inter_with_ep/len(all_inter):.1f}%)")

    # KE distribution for primary electrons
    ke_vals = rp_df.loc[ele_primary, ke_col].dropna()
    if len(ke_vals) > 0:
        print(f"\n  KE of primary electrons (ke branch):")
        print(f"    n:    {len(ke_vals):,}")
        print(f"    mean: {ke_vals.mean():.1f} MeV")
        print(f"    std:  {ke_vals.std():.1f} MeV")
        print(f"    min:  {ke_vals.min():.1f}")
        print(f"    max:  {ke_vals.max():.1f}")
        pcts = np.percentile(ke_vals, [10, 25, 50, 75, 90])
        print(f"    P10/25/50/75/90: {pcts[0]:.0f}/{pcts[1]:.0f}/{pcts[2]:.0f}/{pcts[3]:.0f}/{pcts[4]:.0f}")
        n_above = (ke_vals > 75).sum()
        print(f"    > 75 MeV: {n_above:,} ({100*n_above/len(ke_vals):.1f}%)")

    # calo_ke if available
    if cal_col is not None and cal_col in rp_df.columns:
        cke_vals = rp_df.loc[ele_primary, cal_col].dropna()
        if len(cke_vals) > 0:
            print(f"\n  calo_ke of primary electrons:")
            print(f"    n:    {len(cke_vals):,}")
            print(f"    mean: {cke_vals.mean():.1f} MeV")
            print(f"    std:  {cke_vals.std():.1f} MeV")
            pcts = np.percentile(cke_vals, [10, 25, 50, 75, 90])
            print(f"    P10/25/50/75/90: {pcts[0]:.0f}/{pcts[1]:.0f}/{pcts[2]:.0f}/{pcts[3]:.0f}/{pcts[4]:.0f}")
    else:
        print(f"\n  calo_ke: NOT FOUND in columns")

    # All electrons (pid==1, any is_primary) — KE
    all_ele = (pids == 1)
    ke_all_ele = rp_df.loc[all_ele, ke_col].dropna()
    if len(ke_all_ele) > 0:
        print(f"\n  KE of ALL electrons (pid==1, any is_primary):")
        print(f"    n:    {len(ke_all_ele):,}")
        print(f"    mean: {ke_all_ele.mean():.1f} MeV")
        print(f"    > 75 MeV: {(ke_all_ele > 75).sum():,}")

    # Score distributions for primary electrons
    for score_name, score_cols in [
        ("primary_scores", [('primary_scores', 'I0'), ('primary_scores', 'I1')]),
        ("pid_scores",     [('pid_scores', 'I0'), ('pid_scores', 'I1')]),
    ]:
        print(f"\n  {score_name} for primary electrons:")
        for sc in score_cols:
            if sc in rp_df.columns:
                sv = rp_df.loc[ele_primary, sc].dropna()
                if len(sv) > 0:
                    pcts = np.percentile(sv, [10, 25, 50, 75, 90])
                    print(f"    {sc}: mean={sv.mean():.3f} "
                          f"P10/50/90={pcts[0]:.3f}/{pcts[2]:.3f}/{pcts[4]:.3f}")

    # Column listing
    print(f"\n  Total columns: {len(rp_df.columns)}")
    print(f"  Column names (first 15): {list(rp_df.columns[:15])}")


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading nominal MC (first 50k interactions) ...")
nom_evtdf = pd.read_hdf(NOMINAL_DF, key="evt_0")
particle_stats(nom_evtdf, "NOMINAL MC", max_inter=50000)
del nom_evtdf

for var_name, path in DETVAR_DFS.items():
    print(f"\nLoading {var_name} ...")
    try:
        dv_evtdf = pd.read_hdf(path, key="evt_0")
        particle_stats(dv_evtdf, f"DET-VAR: {var_name}")
        del dv_evtdf
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*60)
print("If primary electron KE distributions look very different,")
print("the issue is in the CAF production (standalone vs merge),")
print("not in the selection code.")
print("="*60)