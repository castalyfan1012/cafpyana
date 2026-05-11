#!/usr/bin/env python3
"""
skim_nue_df.py
--------------
Step 2 of the two-step nueCC weights pipeline.

Workflow
--------
Step 1 — make MC dfs (fast, already done):
    python run_df_maker.py -c nueCC_mc.py -l filelist.txt \\
        -o mc1e20_nueCC_test -nfile 10 -ncpu 10

Step 2 — compute weights (sequential, memory-safe):
    python skim_nue_df.py \\
        -l  /path/to/caf_filelist.txt \\
        -df /path/to/mc1e20_nueCC_test.df \\
        -o  /path/to/mc1e20_nueCC_sys_test \\
        [-nfile 10] [--nuniv 100] [--dry-run]

    Output: /path/to/mc1e20_nueCC_sys_test.df  (key: mcnu_0)

How file matching works
-----------------------
The input .df contains data from N ROOT files stacked with a '__ntuple' index
level (0..N-1).  CAF file i in -l is matched to __ntuple==i in truth_info.
This means the filelist order must be the same as when nueCC_mc.py was run.

Memory per file: ~9 GB peak  (vs ~13 GB via run_df_maker pool)
"""

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from tqdm.auto import tqdm

import tables
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=tables.exceptions.NaturalNameWarning)
pd.set_option("future.no_silent_downcasting", True)

_CAFPYANA_WD = os.environ.get("CAFPYANA_WD", "/home/castalyf/cafpyana")
if _CAFPYANA_WD not in sys.path:
    sys.path.insert(0, _CAFPYANA_WD)

from makedf.makedf import make_mcnudf
from makedf import bnbsyst, geniesyst
from pyanalib.pandas_helpers import multicol_concat


# ── Read truth_info from .df (handles multiple splits) ───────────────────────

def _read_truth_info(df_path):
    """
    Read and concatenate all truth_info_* splits from a .df file.
    Raises if mct_index column is missing.
    """
    with pd.HDFStore(df_path, mode="r") as store:
        keys = sorted(k for k in store.keys() if "truth_info" in k)
        if not keys:
            raise KeyError(
                f"No truth_info key in {df_path}.\n"
                "Run nueCC_mc.py with the updated make_nueCC_df.py first."
            )
        truth_info = pd.concat([store[k] for k in keys])

    if "mct_index" not in truth_info.columns:
        raise KeyError(
            "truth_info is missing 'mct_index'.\n"
            "Add mct_index to _build_truth_info() in make_nueCC_df.py "
            "and rerun nueCC_mc.py."
        )
    return truth_info


def _presel_pairs_for_ntuple(truth_info, ntuple_idx):
    """
    Extract preselected (entry, mct_index) pairs for one ROOT file.
    ntuple_idx is the __ntuple level value (= position in the input filelist).
    """
    NTUPLE_LEVEL = "__ntuple"

    if NTUPLE_LEVEL in truth_info.index.names:
        try:
            ti = truth_info.xs(ntuple_idx, level=NTUPLE_LEVEL)
        except KeyError:
            return set()   # this ntuple produced no interactions
    else:
        # single-file df — no __ntuple level
        ti = truth_info

    presel = ti[ti["passed_presel"]]
    if presel.empty:
        return set()

    mct   = presel["mct_index"]
    valid = mct[mct >= 0]
    if valid.empty:
        return set()

    entry_level = presel.index.names.index("entry")
    entries     = presel.index.get_level_values(entry_level)[
        presel.index.isin(valid.index)
    ]
    return set(zip(entries.tolist(), valid.astype(int).tolist()))


# ── Per-file weights computation ──────────────────────────────────────────────

def _compute_weights(caf_path, presel_pairs, multisim_nuniv):
    """
    Open one ROOT file, run BNB+GENIE, filter to presel_pairs.
    Returns filtered DataFrame (empty on failure).
    """
    t0 = time.time()
    print(f"    presel pairs : {len(presel_pairs)}")

    if not presel_pairs:
        print("    no preselected interactions — skipping")
        return pd.DataFrame(), 0.0

    f           = uproot.open(caf_path)
    mcdf        = make_mcnudf(f, include_weights=False)
    mcdf["ind"] = mcdf.index.get_level_values(-1)
    full_ind    = mcdf["ind"]
    print(f"    total MC nu  : {len(mcdf)}")

    wgtdf = mcdf.copy()

    try:
        bnb = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=multisim_nuniv, slim=True)
        if not bnb.empty:
            wgtdf = multicol_concat(wgtdf, bnb)
            print(f"    BNB          : {bnb.shape[1]} columns")
        del bnb;  gc.collect()
    except Exception as e:
        warnings.warn(f"BNB failed: {e}")

    try:
        genie = geniesyst.geniesyst(f, full_ind, multisim_nuniv=multisim_nuniv, slim=True)
        if not genie.empty:
            wgtdf = multicol_concat(wgtdf, genie)
            print(f"    GENIE        : {genie.shape[1]} columns")
        del genie; gc.collect()
    except Exception as e:
        warnings.warn(f"GENIE failed: {e}")

    del f, mcdf

    e_vals = wgtdf.index.get_level_values(0)
    m_vals = wgtdf.index.get_level_values(-1)
    mask   = np.array([(e, m) in presel_pairs for e, m in zip(e_vals, m_vals)])
    out    = wgtdf[mask].copy()
    print(f"    filtered     : {mask.sum()} / {len(wgtdf)} rows  ({time.time()-t0:.1f}s)")

    del wgtdf; gc.collect()
    return out, time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nueCC weights skimmer — sequential, memory-safe.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example
-------
  python skim_nue_df.py \\
      -l  /exp/sbnd/data/.../mc1e20_filelist_1.txt \\
      -df ../cafpyana_out/mc1e20_nueCC_test_small.df \\
      -o  ../cafpyana_out/mc1e20_nueCC_sys_test_small \\
      -nfile 10 --nuniv 100
""",
    )
    parser.add_argument("-l",        required=True,
                        help="Text file: one ROOT CAF path per line (same order as nueCC_mc.py run)")
    parser.add_argument("-df",       required=True,
                        help="Input .df file produced by nueCC_mc.py (e.g. mc1e20_nueCC_test.df)")
    parser.add_argument("-o",        required=True,
                        help="Output prefix without .df (e.g. mc1e20_nueCC_sys_test)")
    parser.add_argument("-nfile",  type=int, default=0,
                        help="Number of CAF files to process (default 0 = all)")
    parser.add_argument("--nuniv",   type=int, default=100,
                        help="Systematic universes (default 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without processing")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(args.df):
        print(f"ERROR: -df file not found: {args.df}")
        sys.exit(1)

    out_path = args.o if args.o.endswith(".df") else args.o + ".df"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    caf_files = [
        l.strip() for l in open(args.l)
        if l.strip() and not l.startswith("#")
    ]
    if args.nfiles > 0:
        caf_files = caf_files[:args.nfiles]

    print(f"Input df  : {args.df}")
    print(f"CAF files : {len(caf_files)}")
    print(f"Output    : {out_path}")
    print(f"Universes : {args.nuniv}")

    if args.dry_run:
        print("\nDRY RUN — reading truth_info to check ntuple coverage …")
        truth_info = _read_truth_info(args.df)
        if "__ntuple" in truth_info.index.names:
            ntuples = truth_info.index.get_level_values("__ntuple").unique().sort_values()
            print(f"  __ntuple values in df : {list(ntuples)}")
            print(f"  CAF files requested   : 0 .. {len(caf_files)-1}")
        else:
            print("  Single-file df (no __ntuple level)")
        for i, c in enumerate(caf_files):
            print(f"  [{i}] {Path(c).name}")
        return

    # ── Load truth_info once ──────────────────────────────────────────────────
    print("\nLoading truth_info …")
    truth_info = _read_truth_info(args.df)
    print(f"  truth_info rows : {len(truth_info):,}  "
          f"(presel: {truth_info['passed_presel'].sum():,})")

    # ── Process files sequentially ────────────────────────────────────────────
    all_wgt, failed = [], []

    for i, caf_path in enumerate(tqdm(caf_files, unit="file")):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(caf_files)}]  {Path(caf_path).name}  (__ntuple={i})")

        try:
            pairs       = _presel_pairs_for_ntuple(truth_info, i)
            wgt, elapsed = _compute_weights(caf_path, pairs, args.nuniv)
            if not wgt.empty:
                all_wgt.append(wgt)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append((i, caf_path))

    # ── Write combined output ─────────────────────────────────────────────────
    if all_wgt:
        print(f"\nConcatenating {len(all_wgt)} weight tables …")
        combined = pd.concat(all_wgt, ignore_index=False)
        print(f"Writing {len(combined):,} rows → {out_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined.to_hdf(out_path, key="mcnu_0", mode="w",
                            complevel=1, complib="blosc")
        # copy histpotdf from input df for downstream POT accounting
        try:
            with pd.HDFStore(args.df,  mode="r") as src, \
                 pd.HDFStore(out_path, mode="a") as dst:
                for k in src.keys():
                    if "histpotdf" in k:
                        dst[k] = src[k]
        except Exception:
            pass
        del combined
    else:
        print("WARNING: no weight rows produced — output file not written.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OK      : {len(caf_files) - len(failed)}/{len(caf_files)}")
    print(f"Output  : {out_path}")
    if failed:
        print("Failed:")
        for idx, f in failed:
            print(f"  [{idx}] {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()