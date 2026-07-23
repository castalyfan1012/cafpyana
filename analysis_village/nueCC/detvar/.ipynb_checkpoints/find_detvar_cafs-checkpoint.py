#!/usr/bin/env python3
"""
find_detvar_cafs.py
-------------------
Scan SBND production directories to find det-var CAF files and check
whether they contain SPINE (rec.dlp) branches.

This is the FIRST script you should run — before anything else.

Usage:
    # Scan the standard SBND production area for det-var samples
    python find_detvar_cafs.py --scan

    # Check a specific file for SPINE branches
    python find_detvar_cafs.py --check /path/to/detvar.flat.caf.root

    # Scan a specific directory
    python find_detvar_cafs.py --scan --base-dir /pnfs/sbnd/persistent/sbndpro/...
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Standard SBND production locations to search
STANDARD_SEARCH_PATHS = [
    "/pnfs/sbnd/persistent/sbndpro",
    "/pnfs/sbnd/persistent/physics/caf",
    "/exp/sbnd/data/users",
]

DET_VAR_KEYWORDS = [
    "pmtgain", "pmtqe", "pmtspe",      # PDS
    "nosce", "twicesce",                 # SCE
    "wiremodxtheta", "wiremodyz",        # Wire
    "wiremodangle",                      # sometimes used
    "detvar", "det_var",                 # generic
]


def check_spine_branches(filepath: str, verbose: bool = True) -> dict:
    """Check a CAF ROOT file for SPINE (rec.dlp) and Pandora (rec.slc/rec.pfp) branches."""
    try:
        import uproot
    except ImportError:
        print("ERROR: uproot not installed. Run: pip install uproot --break-system-packages")
        sys.exit(1)

    result = {
        "file": filepath,
        "has_spine": False,
        "has_pandora": False,
        "has_mcnu": False,
        "has_hdr": False,
        "spine_branches": [],
        "pandora_branches": [],
    }

    try:
        f = uproot.open(filepath)

        # Check recTree
        tree_name = None
        for name in ["recTree", "recTree;1", "caf"]:
            if name in f:
                tree_name = name
                break

        if tree_name is None:
            if verbose:
                print(f"  WARNING: No recTree found in {filepath}")
                print(f"  Available keys: {list(f.keys())[:20]}")
            return result

        tree = f[tree_name]
        branches = [b.name for b in tree.branches]

        # Check for SPINE branches
        spine_branches = [b for b in branches if b.startswith("rec.dlp")]
        result["has_spine"] = len(spine_branches) > 0
        result["spine_branches"] = spine_branches[:10]  # first 10

        # Check for Pandora branches
        pandora_branches = [b for b in branches if b.startswith(("rec.slc", "rec.pfp"))]
        result["has_pandora"] = len(pandora_branches) > 0
        result["pandora_branches"] = pandora_branches[:10]

        # Check for truth info
        result["has_mcnu"] = any(b.startswith("rec.mc.nu") for b in branches)
        result["has_hdr"] = any(b.startswith("rec.hdr") for b in branches)

        if verbose:
            print(f"\n  File: {filepath}")
            print(f"  SPINE (rec.dlp.*):   {'YES' if result['has_spine'] else 'NO'} ({len(spine_branches)} branches)")
            print(f"  Pandora (rec.slc.*): {'YES' if result['has_pandora'] else 'NO'} ({len(pandora_branches)} branches)")
            print(f"  MC truth (rec.mc.*): {'YES' if result['has_mcnu'] else 'NO'}")
            print(f"  Header (rec.hdr.*):  {'YES' if result['has_hdr'] else 'NO'}")
            if spine_branches:
                print(f"  Sample SPINE branches: {spine_branches[:5]}")

    except Exception as e:
        if verbose:
            print(f"  ERROR reading {filepath}: {e}")
        result["error"] = str(e)

    return result


def scan_for_detvar_dirs(base_dirs: list[str], verbose: bool = True) -> dict[str, list[str]]:
    """Scan directories for det-var CAF files, grouped by variation name."""
    found: dict[str, list[str]] = {}

    for base in base_dirs:
        if not Path(base).exists():
            if verbose:
                print(f"  Skipping {base} (not found)")
            continue

        if verbose:
            print(f"\nScanning {base} ...")

        # Search for directories or files matching det-var keywords
        for keyword in DET_VAR_KEYWORDS:
            # Try glob patterns
            patterns = [
                f"{base}/**/*{keyword}*/*.flat.caf.root",
                f"{base}/**/*{keyword}*/*.root",
                f"{base}/**/det_var/{keyword}/**/*.root",
                f"{base}/**/detvar/{keyword}/**/*.root",
            ]
            for pat in patterns:
                files = sorted(glob.glob(pat, recursive=True))
                if files:
                    key = keyword
                    # Try to extract the specific variation name
                    for var in ["pmtgain", "pmtqe", "pmtspe", "nosce", "twicesce",
                                "wiremodxtheta", "wiremodyz"]:
                        if var in keyword or any(var in f for f in files):
                            key = var
                    found.setdefault(key, []).extend(files)

    # Deduplicate
    for key in found:
        found[key] = sorted(set(found[key]))

    return found


def main():
    parser = argparse.ArgumentParser(
        description="Find and check det-var CAF files for SPINE branches."
    )
    parser.add_argument("--scan", action="store_true",
                        help="Scan standard SBND directories for det-var CAFs")
    parser.add_argument("--check", type=str, default=None,
                        help="Check a specific CAF file for SPINE branches")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Additional base directory to scan")
    parser.add_argument("--check-all", action="store_true",
                        help="When scanning, also check first file of each variation for SPINE")
    args = parser.parse_args()

    if args.check:
        print(f"Checking {args.check} ...")
        result = check_spine_branches(args.check)
        if result["has_spine"]:
            print("\n  ✓ This file HAS SPINE reco — usable for nueCC det-sys!")
        else:
            print("\n  ✗ This file does NOT have SPINE reco.")
            if result["has_pandora"]:
                print("    It has Pandora reco — usable for numuincl but NOT nueCC SPINE analysis.")
            print("    You need SPINE-processed det-var samples. Talk to Afro / production group.")
        return

    if args.scan:
        search_paths = list(STANDARD_SEARCH_PATHS)
        if args.base_dir:
            search_paths.insert(0, args.base_dir)

        print("=" * 70)
        print("Scanning for detector variation CAF files")
        print("=" * 70)

        found = scan_for_detvar_dirs(search_paths)

        if not found:
            print("\nNo det-var CAF files found in standard locations.")
            print("Try specifying --base-dir /path/to/your/detvar/production")
            return

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for var, files in sorted(found.items()):
            print(f"\n  {var}: {len(files)} files")
            if files:
                print(f"    Example: {files[0]}")
                # Check common parent directory
                parents = set(str(Path(f).parent) for f in files)
                if len(parents) <= 3:
                    for p in sorted(parents):
                        n = sum(1 for f in files if str(Path(f).parent) == p)
                        print(f"    Dir: {p} ({n} files)")

        if args.check_all:
            print("\n" + "=" * 70)
            print("SPINE BRANCH CHECK (first file per variation)")
            print("=" * 70)
            has_spine_any = False
            for var, files in sorted(found.items()):
                if files:
                    result = check_spine_branches(files[0], verbose=True)
                    if result["has_spine"]:
                        has_spine_any = True

            if has_spine_any:
                print("\n✓ Some det-var files have SPINE reco — you can proceed!")
            else:
                print("\n✗ No det-var files have SPINE reco.")
                print("  Options:")
                print("  1. Request SPINE reco on det-var samples from production group")
                print("  2. Process the CAFs through SPINE yourself (if you have the workflow)")
                print("  3. Use Bear's Pandora-based det-sys results as a proxy (ask Afro)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()