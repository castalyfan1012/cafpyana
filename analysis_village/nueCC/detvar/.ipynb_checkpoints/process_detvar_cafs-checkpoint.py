#!/usr/bin/env python3
"""
process_detvar_cafs.py  (Step 1)
---------------------------------
Run the nueCC SPINE selection on each det-var CAF sample, producing
per-variation .df files with the same structure as your nominal production.

This is the SPINE-equivalent of what Bear's build_detsys_universes.py does
with Pandora's CAFSlice — but here we reuse YOUR existing make_nueCC_df.py
pipeline to ensure selection cuts are identical to nominal.

Prerequisites:
    - Det-var CAFs with SPINE reco (verify with find_detvar_cafs.py --check)
    - Your make_nueCC_df.py and nue_selection.py on the Python path

Usage:
    # Process all variations (run from your nueCC directory)
    python scripts/process_detvar_cafs.py --var all

    # Process a single variation
    python scripts/process_detvar_cafs.py --var pmtgain

    # Dry run — just print what would be processed
    python scripts/process_detvar_cafs.py --var all --dry-run

    # Process with grid jobs (recommended for full production)
    python scripts/process_detvar_cafs.py --var pmtgain --grid --njobs 50

Environment:
    source /exp/sbnd/app/users/castalyf/setup_cafpyana.sh  # or your setup
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add nueCC dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
NUECC_DIR = SCRIPT_DIR.parent
CAFPYANA_DIR = NUECC_DIR.parents[1]
for p in [str(NUECC_DIR), str(CAFPYANA_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nuecc_detsys_config import (
    NueCCDetsysConfig,
    DET_VARS_ALL,
    build_config,
    find_detvar_cafs,
    det_var_subdir,
)


def process_single_caf(
    caf_path: str,
    output_path: str,
    *,
    verbose: bool = False,
) -> dict:
    """
    Process a single det-var CAF file through the nueCC SPINE selection.

    This imports and runs your existing make_nueCC_df pipeline.
    Returns a dict with stats (n_events, n_selected, etc.).
    """
    from makedf.make_nueCC_df import (
        make_nuecc_evtdf,
        # If you have a lighter version:
        # make_nuecc_evtdf_slim,
    )

    stats = {"caf": caf_path, "output": output_path}

    try:
        t0 = time.time()

        # Load and process the CAF file through your pipeline
        # This should produce the same DataFrame structure as nominal
        evtdf = make_nuecc_evtdf([caf_path], verbose=verbose)

        if evtdf is None or len(evtdf) == 0:
            stats["n_events"] = 0
            stats["status"] = "empty"
            return stats

        stats["n_events"] = len(evtdf)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        evtdf.to_hdf(output_path, key="evtdf_0", mode="w")

        stats["status"] = "ok"
        stats["time_s"] = time.time() - t0

        if verbose:
            print(f"  {Path(caf_path).name}: {len(evtdf)} events -> {output_path} "
                  f"({stats['time_s']:.1f}s)")

    except Exception as e:
        stats["status"] = f"error: {e}"
        if verbose:
            print(f"  ERROR {Path(caf_path).name}: {e}")

    return stats


def process_variation(
    cfg: NueCCDetsysConfig,
    var: str,
    caf_files: list[str],
    *,
    dry_run: bool = False,
    verbose: bool = False,
    max_files: int | None = None,
) -> list[dict]:
    """Process all CAF files for one detector variation."""
    out_dir = Path(cfg.detvar_df_dir) / var
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_files:
        caf_files = caf_files[:max_files]

    print(f"\n{'='*60}")
    print(f"Variation: {var} ({len(caf_files)} files)")
    print(f"Output:    {out_dir}")
    print(f"{'='*60}")

    if dry_run:
        for f in caf_files:
            out_name = Path(f).stem.replace(".flat.caf", "") + f"_{var}.df"
            print(f"  {Path(f).name} -> {out_name}")
        return []

    all_stats = []
    for i, caf_path in enumerate(caf_files):
        out_name = Path(caf_path).stem.replace(".flat.caf", "") + f"_{var}.df"
        out_path = str(out_dir / out_name)

        if Path(out_path).exists():
            print(f"  [{i+1}/{len(caf_files)}] SKIP (exists): {out_name}")
            continue

        print(f"  [{i+1}/{len(caf_files)}] Processing: {Path(caf_path).name}")
        stats = process_single_caf(caf_path, out_path, verbose=verbose)
        all_stats.append(stats)

    return all_stats


def generate_grid_script(
    cfg: NueCCDetsysConfig,
    var: str,
    caf_files: list[str],
    njobs: int = 50,
) -> str:
    """Generate a grid submission script for processing det-var CAFs."""
    out_dir = Path(cfg.detvar_df_dir) / var
    script_dir = Path(cfg.output_dir) / cfg.version / "grid_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    # Create a file list
    filelist_path = script_dir / f"{var}_filelist.txt"
    with open(filelist_path, "w") as f:
        for caf in caf_files:
            f.write(f"{caf}\n")

    # Create the worker script
    worker_path = script_dir / f"run_{var}.sh"
    worker_content = f"""#!/bin/bash
# Grid worker for {var} det-var processing
# Auto-generated by process_detvar_cafs.py

source /exp/sbnd/app/users/castalyf/setup_cafpyana.sh

FILELIST="{filelist_path}"
OUTDIR="{out_dir}"
NJOBS={njobs}
JOBID=${{PROCESS}}

# Calculate which files this job handles
NFILES=$(wc -l < $FILELIST)
FILES_PER_JOB=$(( (NFILES + NJOBS - 1) / NJOBS ))
START=$(( JOBID * FILES_PER_JOB + 1 ))
END=$(( START + FILES_PER_JOB - 1 ))

mkdir -p $OUTDIR

for CAFFILE in $(sed -n "${{START}},${{END}}p" $FILELIST); do
    BASENAME=$(basename $CAFFILE .flat.caf.root)
    OUTFILE="${{OUTDIR}}/${{BASENAME}}_{var}.df"

    if [ -f "$OUTFILE" ]; then
        echo "SKIP (exists): $OUTFILE"
        continue
    fi

    echo "Processing: $CAFFILE"
    python -c "
import sys
sys.path.insert(0, '{NUECC_DIR}')
sys.path.insert(0, '{CAFPYANA_DIR}')
from makedf.make_nueCC_df import make_nuecc_evtdf
evtdf = make_nuecc_evtdf(['$CAFFILE'], verbose=False)
if evtdf is not None and len(evtdf) > 0:
    evtdf.to_hdf('$OUTFILE', key='evtdf_0', mode='w')
    print(f'  Saved {{len(evtdf)}} events to $OUTFILE')
else:
    print('  No events')
"
done
"""
    with open(worker_path, "w") as f:
        f.write(worker_content)
    os.chmod(worker_path, 0o755)

    # Create the submission script
    submit_path = script_dir / f"submit_{var}.sh"
    submit_content = f"""#!/bin/bash
# Submit grid jobs for {var} det-var processing
# Usage: source {submit_path}

jobsub_submit \\
    -G sbnd \\
    -N {njobs} \\
    --memory=8GB \\
    --disk=10GB \\
    --expected-lifetime=4h \\
    --resource-provides=usage_model=OPPORTUNISTIC \\
    file://{worker_path}

echo "Submitted {njobs} jobs for {var}"
echo "Output will be in: {out_dir}"
"""
    with open(submit_path, "w") as f:
        f.write(submit_content)
    os.chmod(submit_path, 0o755)

    print(f"  Grid scripts written:")
    print(f"    Worker:  {worker_path}")
    print(f"    Submit:  {submit_path}")
    print(f"    Files:   {filelist_path} ({len(caf_files)} files)")
    print(f"    Jobs:    {njobs}")

    return str(submit_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process det-var CAFs through nueCC SPINE selection."
    )
    parser.add_argument("--var", required=True,
                        help="Variation to process ('all' for all, or specific name)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would be done")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files per variation (for testing)")
    parser.add_argument("--grid", action="store_true",
                        help="Generate grid submission scripts instead of running locally")
    parser.add_argument("--njobs", type=int, default=50,
                        help="Number of grid jobs (with --grid)")

    # Config overrides
    parser.add_argument("--detvar-caf-base", type=str, default=None)
    parser.add_argument("--nominal-caf-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Build config
    cfg_kwargs = {}
    if args.detvar_caf_base:
        cfg_kwargs["detvar_caf_base"] = args.detvar_caf_base
    if args.nominal_caf_dir:
        cfg_kwargs["nominal_caf_dir"] = args.nominal_caf_dir
    if args.output_dir:
        cfg_kwargs["output_dir"] = args.output_dir
    cfg = build_config(**cfg_kwargs)

    # Which variations to process
    if args.var == "all":
        vars_to_process = list(DET_VARS_ALL)
    else:
        vars_to_process = [args.var]

    for var in vars_to_process:
        caf_files = find_detvar_cafs(cfg, var)
        if not caf_files:
            print(f"WARNING: No CAF files found for {var}. "
                  f"Check detvar_caf_base={cfg.detvar_caf_base}")
            continue

        if args.grid:
            generate_grid_script(cfg, var, caf_files, njobs=args.njobs)
        else:
            process_variation(
                cfg, var, caf_files,
                dry_run=args.dry_run,
                verbose=args.verbose,
                max_files=args.max_files,
            )


if __name__ == "__main__":
    main()