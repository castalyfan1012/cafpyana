"""
check_detvar_branches.py
------------------------
Quick diagnostic: what do the particle-level branches actually contain
in Jacob's wiremod flat CAFs?

Usage:
  python3 check_detvar_branches.py [--nentry 100]
"""
import sys, argparse
import numpy as np
import uproot
import awkward as ak

parser = argparse.ArgumentParser()
parser.add_argument("--nentry", type=int, default=200,
                    help="Number of entries to sample")
args = parser.parse_args()

CAFS = {
    "wiremodxtheta": (
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/"
        "users/jzettle/sbnd_wiremod_x_thetaxw/"
        "combined_wiremod_x_thetaxw.flat.caf.root"
    ),
}

BRANCHES = [
    "rec.dlp.particles.is_primary",
    "rec.dlp.particles.pid",
    "rec.dlp.particles.primary_scores.0",
    "rec.dlp.particles.primary_scores.1",
    "rec.dlp.particles.pid_scores.0",
    "rec.dlp.particles.pid_scores.1",
    "rec.dlp.particles.ke",
    "rec.dlp.particles.start_dir.2",
    "rec.dlp.particles.momentum.0",
    "rec.dlp.particles.momentum.1",
    "rec.dlp.particles.momentum.2",
]

# Also check interaction-level branches for context
INTER_BRANCHES = [
    "rec.dlp.is_flash_matched",
    "rec.dlp.flash_total_pe",
    "rec.dlp.vertex.0",
]

for name, path in CAFS.items():
    print(f"\n{'='*70}")
    print(f"  {name}: {path.split('/')[-1]}")
    print(f"{'='*70}")

    f = uproot.open(path)
    rec = f["recTree"]
    n_total = rec.num_entries
    print(f"  Total entries: {n_total:,}")

    # Sample first N entries
    n = min(args.nentry, n_total)
    print(f"  Sampling first {n} entries\n")

    # List all available branches
    all_branches = rec.keys()
    dlp_particle_branches = [b for b in all_branches if "dlp.particles" in b]
    print(f"  Available particle branches ({len(dlp_particle_branches)}):")
    for b in sorted(dlp_particle_branches)[:30]:
        print(f"    {b}")
    if len(dlp_particle_branches) > 30:
        print(f"    ... and {len(dlp_particle_branches)-30} more")

    print(f"\n  --- Particle-level branch diagnostics (first {n} entries) ---")
    for branch in BRANCHES:
        if branch not in all_branches:
            print(f"\n  {branch}: NOT FOUND")
            continue
        try:
            arr = rec[branch].array(entry_stop=n)
            flat = ak.to_numpy(ak.flatten(arr, axis=None))
            n_vals = len(flat)
            if n_vals == 0:
                print(f"\n  {branch}: EMPTY (no particles in first {n} entries)")
                continue

            n_nan = np.isnan(flat.astype(float)).sum()
            n_zero = (flat == 0).sum()
            n_finite = np.isfinite(flat.astype(float)).sum()

            print(f"\n  {branch}:")
            print(f"    n_values : {n_vals:,}")
            print(f"    n_NaN    : {n_nan:,} ({100*n_nan/n_vals:.1f}%)")
            print(f"    n_zero   : {n_zero:,} ({100*n_zero/n_vals:.1f}%)")
            print(f"    n_finite : {n_finite:,}")

            finite = flat[np.isfinite(flat.astype(float))]
            if len(finite) > 0:
                print(f"    min      : {finite.min():.4f}")
                print(f"    max      : {finite.max():.4f}")
                print(f"    mean     : {finite.mean():.4f}")
                print(f"    std      : {finite.std():.4f}")

                # For integer-like branches, show value counts
                if branch.endswith("pid") or branch.endswith("is_primary"):
                    unique, counts = np.unique(finite, return_counts=True)
                    print(f"    unique values: {dict(zip(unique.astype(int), counts))}")

                # For score branches, show distribution
                if "scores" in branch:
                    pcts = np.percentile(finite, [0, 10, 25, 50, 75, 90, 100])
                    print(f"    percentiles [0,10,25,50,75,90,100]: "
                          f"{[f'{p:.3f}' for p in pcts]}")

                    # How many pass our cuts?
                    if "primary_scores.0" in branch:
                        n_pass = (finite > 0.99).sum()
                        print(f"    > 0.99 (PRIMARY_SCORE_CUT): {n_pass:,} "
                              f"({100*n_pass/len(finite):.1f}%)")
                    if "pid_scores.1" in branch:
                        n_pass = (finite > 0.915).sum()
                        print(f"    > 0.915 (PID_SCORE_CUT): {n_pass:,} "
                              f"({100*n_pass/len(finite):.1f}%)")

                # For ke, show how many pass threshold
                if branch.endswith(".ke"):
                    n_pass = (finite > 75).sum()
                    print(f"    > 75 MeV (ELECTRON_THRESHOLD): {n_pass:,} "
                          f"({100*n_pass/len(finite):.1f}%)")

        except Exception as e:
            print(f"\n  {branch}: ERROR: {e}")

    # Check what the combined mask looks like
    print(f"\n  --- Combined electron mask check (first {n} entries) ---")
    try:
        is_primary = rec["rec.dlp.particles.is_primary"].array(entry_stop=n)
        pid = rec["rec.dlp.particles.pid"].array(entry_stop=n)
        pri_score = rec["rec.dlp.particles.primary_scores.0"].array(entry_stop=n)
        pid_score = rec["rec.dlp.particles.pid_scores.1"].array(entry_stop=n)
        ke = rec["rec.dlp.particles.ke"].array(entry_stop=n)

        # Step by step
        m1 = (is_primary == 1)
        m2 = m1 & (pid == 1)
        m3 = m2 & (pri_score > 0.99)
        m4 = m3 & (pid_score > 0.915)
        m5 = m4 & (ke > 75)

        for label, mask in [
            ("is_primary==1", m1),
            ("+ pid==1 (electron)", m2),
            ("+ primary_score>0.99", m3),
            ("+ pid_score>0.915", m4),
            ("+ ke>75 MeV", m5),
        ]:
            n_pass = int(ak.sum(ak.flatten(mask, axis=None)))
            n_entries_with = int(ak.sum(ak.any(mask, axis=1)))
            print(f"    {label:<30s}: {n_pass:>6,} particles, "
                  f"{n_entries_with:>5,} entries")

        # Try relaxed cuts to see if anything passes
        print(f"\n  --- Relaxed cuts (first {n} entries) ---")
        m_relax = (pid == 1) & (ke > 75)
        n_relax = int(ak.sum(ak.flatten(m_relax, axis=None)))
        print(f"    pid==1 & ke>75 (no score cuts): {n_relax:,}")

        m_relax2 = (pid == 1)
        n_relax2 = int(ak.sum(ak.flatten(m_relax2, axis=None)))
        print(f"    pid==1 only: {n_relax2:,}")

        m_relax3 = (is_primary == 1)
        n_relax3 = int(ak.sum(ak.flatten(m_relax3, axis=None)))
        print(f"    is_primary==1 only: {n_relax3:,}")

    except Exception as e:
        print(f"    ERROR in combined check: {e}")

    print()