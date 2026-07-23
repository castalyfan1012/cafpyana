"""
check_nominal_reco_p.py — run on EAF
"""
import pandas as pd, numpy as np, sys

path = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_nueCC_v2.df"
nom = pd.read_hdf(path, key="evt_0")

print("Column structure (first 20 cols):")
print(nom.columns[:20].tolist())
print(f"\nColumn nlevels: {nom.columns.nlevels}")

# Handle both flat and MultiIndex columns
def getcol(df, name):
    if df.columns.nlevels == 1:
        return df[name]
    # Try top level
    matches = [c for c in df.columns if c[0] == name]
    if matches:
        return df[matches[0]]
    # Try any level
    matches = [c for c in df.columns if name in c]
    if matches:
        print(f"  Found '{name}' as {matches[0]}")
        return df[matches[0]]
    raise KeyError(name)

# Find selection columns
print("\nLooking for selection-related columns:")
sel_cols = [c for c in nom.columns
            if any(x in str(c) for x in ["is_sig","sel_","precut","fiducial","flash"])]
for c in sel_cols[:15]:
    print(f"  {c}")

# Find reco columns
print("\nLooking for reco columns:")
reco_cols = [c for c in nom.columns if "reco" in str(c).lower()]
for c in reco_cols[:15]:
    print(f"  {c}")

# Try to get selected signal
try:
    is_sig  = getcol(nom, "is_sig")
    sel_cut = getcol(nom, "sel_vertex_distance")
    sel = nom[is_sig & sel_cut]
    print(f"\nSelected signal events: {len(sel):,}")
    for var in ["reco_p", "reco_ke", "reco_costheta"]:
        try:
            v = getcol(sel, var)
            print(f"\n{var}:\n{v.describe()}")
        except KeyError:
            print(f"  '{var}' not found")
except Exception as e:
    print(f"\nError: {e}")
    print("Trying to dump all top-level column names:")
    if nom.columns.nlevels > 1:
        top = sorted(set(c[0] for c in nom.columns))
        print(top[:40])