#!/usr/bin/env python3
"""
run_skim_pool.py
----------------
Parallel wrapper for skim_nue.process_file using multiprocessing.Pool.

Usage
-----
    python run_skim_pool.py [--workers N] [--nfiles N] [--mem-per-worker GB] [--dry-run]

Worker count is chosen as the minimum of:
  - CPU count
  - available RAM / mem_per_worker  (default 3 GB — each file peaks at ~2-3 GB)
  - number of files
Pass --workers N to override.

maxtasksperchild=1 ensures each worker process exits after one file so
memory is fully returned to the OS before the next file is loaded.
"""

import argparse
import glob
import multiprocessing as mp
import os
import time
from pathlib import Path

import skim_nue as sk


# ── Memory estimate ────────────────────────────────────────────────────────────
def _available_gb():
    """Return available system RAM in GB (reads /proc/meminfo, falls back to 8)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        pass
    return 8.0


def _safe_workers(n_files, mem_per_worker_gb):
    avail    = _available_gb()
    by_mem   = max(1, int(avail / mem_per_worker_gb))
    by_cpu   = mp.cpu_count()
    chosen   = min(by_cpu, by_mem, n_files)
    print(f"  Available RAM : {avail:.1f} GB")
    print(f"  Mem/worker    : {mem_per_worker_gb} GB  →  RAM allows {by_mem} workers")
    print(f"  CPU count     : {by_cpu}")
    return chosen


# ── Worker (must be top-level for pickling) ────────────────────────────────────
def _worker(args):
    src, dst, pot_key, dry_run = args
    return sk.process_file(
        src_path         = src,
        dst_path         = dst,
        pot_key          = pot_key,
        dry_run          = dry_run,
        print_col_report = False,
    )


def main():
    parser = argparse.ArgumentParser(description="Memory-safe pool runner for skim_nue")
    parser.add_argument("--workers",          type=int,   default=None,
                        help="Override worker count (default: auto from RAM)")
    parser.add_argument("--mem-per-worker",   type=float, default=3.0,
                        help="GB of RAM to reserve per worker (default: 3)")
    parser.add_argument("--nfiles",           type=int,   default=None,
                        help="Cap number of files (default: all)")
    parser.add_argument("--dry-run",          action="store_true")
    args = parser.parse_args()

    cfg   = sk.CONFIG
    files = sorted(glob.glob(cfg["input_glob"]))
    if not files:
        print(f"ERROR: no files match {cfg['input_glob']!r}")
        return
    if args.nfiles:
        files = files[: args.nfiles]

    os.makedirs(cfg["output_dir"], exist_ok=True)

    jobs = [
        (
            f,
            os.path.join(cfg["output_dir"], Path(f).stem + "_skimmed.df"),
            cfg["pot_key"],
            args.dry_run,
        )
        for f in files
    ]

    if args.workers:
        n_workers = args.workers
        print(f"  Available RAM : (manual override)")
    else:
        n_workers = _safe_workers(len(files), args.mem_per_worker)

    print(f"Files   : {len(files)}")
    print(f"Workers : {n_workers}  (override with --workers N)")
    print(f"Output  : {cfg['output_dir']}")
    print(f"Dry run : {args.dry_run}\n")

    t0 = time.time()
    # maxtasksperchild=1: each worker exits after one file → memory fully freed
    with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        all_stats = pool.map(_worker, jobs, chunksize=1)

    ok  = [s for s in all_stats if "error" not in s]
    bad = [s for s in all_stats if "error" in s]

    print(f"\n{'='*60}")
    print(f"  Files OK / total : {len(ok)} / {len(all_stats)}")
    if bad:
        for s in bad:
            print(f"  FAILED : {s['src']}  ({s['error']})")
    if ok:
        total_pot = sum(s["pot"]               for s in ok)
        n_sig     = sum(s["n_true_signal"]      for s in ok)
        n_sig_pre = sum(s["n_sig_after_presel"] for s in ok)
        print(f"  True signal (pre-cut)   : {n_sig:,}  <- denominator")
        print(f"  True signal after FM+FV : {n_sig_pre:,}  "
              f"({n_sig_pre/max(n_sig,1)*100:.1f}%)")
        print(f"  Total POT               : {total_pot:.4e}")
        print(f"  Scale to {cfg['target_pot']:.2e}  : "
              f"{cfg['target_pot']/total_pot:.4f}")
    print(f"  Wall time               : {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()