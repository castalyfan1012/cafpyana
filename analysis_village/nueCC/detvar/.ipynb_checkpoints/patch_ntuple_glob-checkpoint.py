#!/usr/bin/env python3
"""
patch_ntuple_glob.py
---------------------
Patches pyanalib/ntuple_glob.py to handle combined flat CAF files where
TotalEvents = 0 (a known hadd artifact).

The fix: when TotalEvents == 0, fall back to checking if recTree actually
has entries rather than blindly skipping the file.

Usage (run from your cafpyana root):
    python analysis_village/nueCC/scripts/patch_ntuple_glob.py

Or apply manually — see the ONE_LINE_FIX below.
"""
import re
import shutil
from pathlib import Path

# ── Find ntuple_glob.py ──
CAFPYANA_ROOT = Path(__file__).resolve().parents[3]  # cafpyana/
TARGET = CAFPYANA_ROOT / "pyanalib" / "ntuple_glob.py"

if not TARGET.exists():
    # Try common fallback locations
    for candidate in [
        Path.home() / "cafpyana" / "pyanalib" / "ntuple_glob.py",
        Path("/home/castalyf/cafpyana/pyanalib/ntuple_glob.py"),
    ]:
        if candidate.exists():
            TARGET = candidate
            break

if not TARGET.exists():
    print(f"ERROR: Cannot find ntuple_glob.py. Tried {TARGET}")
    print("Pass the path explicitly: python patch_ntuple_glob.py /path/to/ntuple_glob.py")
    import sys
    if len(sys.argv) > 1:
        TARGET = Path(sys.argv[1])
    else:
        raise SystemExit(1)

print(f"Patching: {TARGET}")

# ── Read original ──
original = TARGET.read_text()

# ── Check if already patched ──
if "totevt < 1e-6 and _recTree_is_empty" in original or "DETVAR_PATCH" in original:
    print("Already patched — nothing to do.")
    raise SystemExit(0)

# ── The original problematic lines ──
OLD = """    elif totevt < 1e-6:
        print(f"File ({fname}) has 0 in TotalEvents. Skipping main DFS.", flush=True)"""

# ── The replacement: fall back to checking recTree entries directly ──
NEW = """    elif totevt < 1e-6:
        # DETVAR_PATCH: combined flat CAFs (hadd'd) often have TotalEvents=0
        # even when recTree has valid entries. Fall back to checking recTree directly.
        _recTree_is_empty = True
        try:
            _rt = f["recTree"]
            if hasattr(_rt, "num_entries"):
                _recTree_is_empty = (_rt.num_entries < 1)
            elif hasattr(_rt, "__len__"):
                _recTree_is_empty = (len(_rt) < 1)
        except Exception:
            pass
        if _recTree_is_empty:
            print(f"File ({fname}) has 0 in TotalEvents and empty recTree. Skipping main DFS.", flush=True)
        else:
            print(f"File ({fname}) has 0 in TotalEvents but recTree has entries — processing anyway (combined CAF).", flush=True)
            for i, applyf in enumerate(applyfs):
                df = applyf(f)
                if df is not None:
                    df = df.copy(deep=True)
                    df["__ntuple"] = index
                    df.set_index("__ntuple", append=True, inplace=True)
                    new_order = [df.index.nlevels - 1] + list(range(df.index.nlevels - 1))
                    df = df.reorder_levels(new_order)
                    results.append(df)"""

if OLD not in original:
    print("ERROR: Could not find the target code block to patch.")
    print("The ntuple_glob.py may have changed. Apply the fix manually:")
    print()
    print("  Find this block in pyanalib/ntuple_glob.py:")
    print("    elif totevt < 1e-6:")
    print('        print(f"File ({fname}) has 0 in TotalEvents. Skipping main DFS.", flush=True)')
    print()
    print("  Replace with the DETVAR_PATCH block from this script.")
    raise SystemExit(1)

# ── Back up original ──
backup = TARGET.with_suffix(".py.orig")
shutil.copy2(TARGET, backup)
print(f"Backup: {backup}")

# ── Write patched version ──
patched = original.replace(OLD, NEW)
TARGET.write_text(patched)
print("Patch applied successfully.")
print()
print("To revert: cp pyanalib/ntuple_glob.py.orig pyanalib/ntuple_glob.py")