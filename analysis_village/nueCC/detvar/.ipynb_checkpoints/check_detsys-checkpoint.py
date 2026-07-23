"""
Minimal diagnostic to check exactly why make_spine_int_df gives 0 rows
on the combined flat CAF, and whether direct uproot works.
Run on GPVM:
  python3 check_detsys.py /path/to/combined_wiremod_x_thetaxw.flat.caf.root
"""
import sys, uproot, awkward as ak, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/pnfs/sbnd/persistent/users/jzettle/sbnd_wiremod_x_thetaxw/combined_wiremod_x_thetaxw.flat.caf.root"

print(f"Opening: {path}")
f = uproot.open(path)
rec = f["recTree"]

# ── 1. How many entries does recTree have? ────────────────────────────────────
n_entries = rec.num_entries
print(f"\n[1] recTree entries: {n_entries}")

# ── 2. Check the jagged index branch that make_spine_int_df uses ──────────────
# This is what breaks with hadd'd files
idx_branch = "rec.dlp..index"
try:
    arr = rec[idx_branch].array()
    print(f"\n[2] '{idx_branch}' loaded OK")
    print(f"    type:  {arr.type}")
    print(f"    len:   {len(arr)} entries")
    flat = ak.flatten(arr, axis=None)
    print(f"    flat:  {len(flat)} values")
    print(f"    first 10: {ak.to_list(arr[:3])}")
except Exception as e:
    print(f"\n[2] '{idx_branch}' FAILED: {e}")
    # Try alternative names
    for alt in ["rec.dlp.index", "rec.dlp..idx"]:
        try:
            arr = rec[alt].array()
            print(f"    Found under '{alt}' instead")
            break
        except:
            pass

# ── 3. Check interaction-level arrays (flat per entry) ───────────────────────
print(f"\n[3] Interaction-level branches (should be 1 value per interaction):")
for br in ["rec.dlp.is_flash_matched", "rec.dlp.flash_total_pe",
           "rec.dlp.vertex.0", "rec.dlp.vertex.1", "rec.dlp.vertex.2"]:
    try:
        arr = rec[br].array()
        flat = ak.to_numpy(ak.flatten(arr, axis=None))
        print(f"    {br}: {arr.type}  -> flat shape {flat.shape}, "
              f"first 3: {flat[:3]}")
    except Exception as e:
        print(f"    {br}: FAILED — {e}")

# ── 4. Check particle-level arrays ───────────────────────────────────────────
print(f"\n[4] Particle-level branches (ragged per interaction):")
for br in ["rec.dlp.particles.momentum.0",
           "rec.dlp.particles.start_dir.2",
           "rec.dlp.particles.vertex_distance"]:
    try:
        arr = rec[br].array()
        print(f"    {br}: {arr.type}")
        print(f"      entry 0 has {len(arr[0])} particles: {ak.to_list(arr[0])[:3]}")
    except Exception as e:
        print(f"    {br}: FAILED — {e}")

# ── 5. Apply FM+FV directly and count selected interactions ──────────────────
print(f"\n[5] Direct FM+FV selection (bypassing make_spine_int_df):")
try:
    fm = ak.to_numpy(ak.flatten(rec["rec.dlp.is_flash_matched"].array(), axis=None))
    pe = ak.to_numpy(ak.flatten(rec["rec.dlp.flash_total_pe"].array(),    axis=None))
    vx = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.0"].array(),          axis=None))
    vy = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.1"].array(),          axis=None))
    vz = ak.to_numpy(ak.flatten(rec["rec.dlp.vertex.2"].array(),          axis=None))

    passed_fm = (fm == 1) & (pe > 0)
    abs_x = np.abs(vx)
    passed_fv = (
        (abs_x > 5) & (abs_x < 190) & (
            ((vz > 10)  & (vz < 250) & (vy > -190) & (vy < 190)) |
            ((vz > 250) & (vz < 450) & (vy > -190) & (vy < 100) & (vx < 0)) |
            ((vz > 250) & (vz < 450) & (vy > -190) & (vy < 190) & (vx > 0))
        )
    )
    sel = passed_fm & passed_fv

    print(f"    Total interactions: {len(fm)}")
    print(f"    Pass FM:            {passed_fm.sum()}")
    print(f"    Pass FV:            {passed_fv.sum()}")
    print(f"    Pass FM+FV:         {sel.sum()}")

    # ── 6. Check particle branches for selected interactions ──────────────────
    print(f"\n[6] Particle kinematics for FM+FV selected events:")
    px_arr = rec["rec.dlp.particles.momentum.0"].array()
    py_arr = rec["rec.dlp.particles.momentum.1"].array()
    pz_arr = rec["rec.dlp.particles.momentum.2"].array()
    sd_arr = rec["rec.dlp.particles.start_dir.2"].array()  # costheta proxy

    # How many particles per selected interaction?
    # Need to map sel (flat over interactions) back to per-entry structure
    print(f"    px_arr type: {px_arr.type}")
    print(f"    First entry particle px values: {ak.to_list(px_arr[0])[:5]}")

except Exception as e:
    print(f"    FAILED: {e}")
    import traceback; traceback.print_exc()

print("\nDone. Share this output to diagnose the issue.")