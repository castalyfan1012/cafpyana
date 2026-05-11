#!/usr/bin/env python3
"""
skim_nue_df.py
--------------
Step 2 of the two-step nueCC weights pipeline.

Workflow
--------
Step 1 (already done — fast, low memory):
    python run_df_maker.py -c nueCC_mc.py -ngrid <N_files> -l filelist.txt -o mc1e20_nueCC
    → one .df per ROOT file in the output directory

Step 2 (this script — sequential, predictable memory):
    python skim_nue_df.py \\
        -l  /path/to/caf_filelist.txt \\
        -df /path/to/dfs/ \\
        -o  /path/to/weights/ \\
        [-nfile 10] [--nuniv 100] [--dry-run]

File matching
-------------
CAF files from -l and .df files from -df are paired in sorted order (1:1).
Both lists must have the same length. Use -ngrid <N_files> in run_df_maker
to ensure one .df per ROOT file.

Memory profile per file
-----------------------
    mcdf    ~200 MB
    BNB     ~3 GB
    GENIE   ~8 GB  (unavoidable — full MC index needed for universe alignment)
    Total   ~9 GB  (vs ~13 GB with nueCC_weights.py via run_df_maker pool)

Requirements
------------
truth_info_0 must contain a 'mct_index' column.
This is produced by the updated make_nueCC_df.py.
Run with the cafpyana venv:
    source /home/castalyf/cafpyana/envs/venv_py310_cafpyana/bin/activate
"""

import argparse
import gc
import glob
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

OUTPUT_KEY = "mcnu_0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_presel_pairs(df_path):
    """
    Read preselected (entry, mct_index) pairs from truth_info_0.
    Requires mct_index column (produced by updated make_nueCC_df.py).
    """
    with pd.HDFStore(df_path, mode="r") as store:
        keys    = store.keys()
        ti_keys = [k for k in keys if "truth_info" in k]
        if not ti_keys:
            raise KeyError(
                f"No truth_info key in {df_path}. Keys: {keys}\n"
                "Did you run nueCC_mc.py with the updated make_nueCC_df.py?"
            )
        truth_info = store[ti_keys[0]]

    if "mct_index" not in truth_info.columns:
        raise KeyError(
            "truth_info_0 is missing 'mct_index'.\n"
            "Add mct_index to _build_truth_info() in make_nueCC_df.py "
            "and rerun nueCC_mc.py."
        )

    presel = truth_info[truth_info["passed_presel"]]
    if presel.empty:
        return set()

    entry_level = presel.index.names.index("entry")
    mct         = presel["mct_index"]
    valid       = mct[mct >= 0]
    if valid.empty:
        return set()

    valid_entries = presel.index.get_level_values(entry_level)[
        presel.index.isin(valid.index)
    ]
    return set(zip(valid_entries.tolist(), valid.astype(int).tolist()))


# ── Per-file processor ────────────────────────────────────────────────────────

def process_one(caf_path, df_path, out_path, multisim_nuniv=100):
    t0 = time.time()

    presel_pairs = _read_presel_pairs(df_path)
    print(f"    presel pairs : {len(presel_pairs)}")
    if not presel_pairs:
        print("    WARNING: no preselected interactions — skipping")
        return {"n_presel": 0, "n_out": 0, "elapsed_s": round(time.time()-t0, 1)}

    f           = uproot.open(caf_path)
    mcdf        = make_mcnudf(f, include_weights=False)
    mcdf["ind"] = mcdf.index.get_level_values(-1)
    full_ind    = mcdf["ind"]
    print(f"    total MC nu  : {len(mcdf)}")

    wgtdf = mcdf.copy()

    try:
        bnb_wgt = bnbsyst.bnbsyst(f, full_ind, multisim_nuniv=multisim_nuniv, slim=True)
        if not bnb_wgt.empty:
            wgtdf = multicol_concat(wgtdf, bnb_wgt)
            print(f"    BNB          : {bnb_wgt.shape[1]} columns")
        del bnb_wgt
        gc.collect()
    except Exception as e:
        warnings.warn(f"    BNB failed: {e}")

    try:
        genie_wgt = geniesyst.geniesyst(f, full_ind, multisim_nuniv=multisim_nuniv, slim=True)
        if not genie_wgt.empty:
            wgtdf = multicol_concat(wgtdf, genie_wgt)
            print(f"    GENIE        : {genie_wgt.shape[1]} columns")
        del genie_wgt
        gc.collect()
    except Exception as e:
        warnings.warn(f"    GENIE failed: {e}")

    del f

    e_vals = wgtdf.index.get_level_values(0)
    m_vals = wgtdf.index.get_level_values(-1)
    mask   = np.array([(e, m) in presel_pairs for e, m in zip(e_vals, m_vals)])
    out    = wgtdf[mask].copy()
    print(f"    filtered     : {mask.sum()} / {len(wgtdf)} rows")

    del wgtdf, mcdf
    gc.collect()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.to_hdf(out_path, key=OUTPUT_KEY, mode="w", complevel=1, complib="blosc")
        try:
            with pd.HDFStore(df_path,  mode="r") as src, \
                 pd.HDFStore(out_path, mode="a") as dst:
                for k in src.keys():
                    if "histpotdf" in k:
                        dst[k] = src[k]
        except Exception:
            pass

    stats = dict(n_presel=len(presel_pairs), n_out=int(mask.sum()),
                 elapsed_s=round(time.time()-t0, 1))
    print(f"    written      : {out_path}  ({stats['elapsed_s']:.1f}s)")
    del out
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nueCC weights skimmer — sequential, memory-safe.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples
--------
  # test run (10 files):
  python skim_nue_df.py -l caf_filelist.txt -df /path/to/dfs/ -o /path/to/weights/ -nfile 10

  # full production:
  python skim_nue_df.py -l caf_filelist.txt -df /path/to/dfs/ -o /path/to/weights/

CAF files from -l and .df files from -df are paired in sorted order.
Both must have the same length (use -ngrid <N_files> in run_df_maker).
""",
    )
    parser.add_argument("-l",        required=True,
                        help="Text file: one ROOT CAF path per line")
    parser.add_argument("-df",       required=True, dest="df_dir",
                        help="Directory containing evt .df files (one per ROOT file)")
    parser.add_argument("-o",        required=True, dest="output_dir",
                        help="Output directory for weights .df files")
    parser.add_argument("-nfile",  type=int, default=0,
                        help="Max files to process (0 = all)")
    parser.add_argument("--nuniv",   type=int, default=100,
                        help="Systematic universes (default 100)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Build file lists ──────────────────────────────────────────────────────
    caf_files = [
        l.strip() for l in open(args.l)
        if l.strip() and not l.startswith("#")
    ]
    df_files = sorted(glob.glob(os.path.join(args.df_dir, "*.df")))

    if not df_files:
        print(f"ERROR: no .df files found in {args.df_dir}")
        sys.exit(1)

    if len(caf_files) != len(df_files):
        print(f"ERROR: {len(caf_files)} CAF files but {len(df_files)} .df files.")
        print("  Use -ngrid <N_files> in run_df_maker to get one .df per ROOT file.")
        print(f"  CAF[0] : {caf_files[0]}")
        print(f"  DF [0] : {df_files[0]}")
        sys.exit(1)

    if args.nfiles > 0:
        caf_files = caf_files[:args.nfiles]
        df_files  = df_files[:args.nfiles]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Files     : {len(caf_files)}")
    print(f"DF dir    : {args.df_dir}")
    print(f"Output    : {args.output_dir}")
    print(f"Universes : {args.nuniv}")
    if args.dry_run:
        print("DRY RUN")
        for i, (c, d) in enumerate(zip(caf_files, df_files)):
            print(f"  [{i}] {Path(c).name}  <->  {Path(d).name}")
        return

    # ── Process sequentially ──────────────────────────────────────────────────
    all_stats, failed = [], []

    for i, (caf_path, df_path) in enumerate(
        tqdm(zip(caf_files, df_files), total=len(caf_files), unit="file")
    ):
        stem     = Path(df_path).stem
        out_path = os.path.join(args.output_dir, f"{stem}_weights.df")

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(caf_files)}]")
        print(f"  CAF : {Path(caf_path).name}")
        print(f"  DF  : {Path(df_path).name}")
        print(f"  OUT : {Path(out_path).name}")

        try:
            s = process_one(caf_path, df_path, out_path, args.nuniv)
            all_stats.append(s)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(caf_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OK : {len(all_stats)}/{len(caf_files)}")
    if all_stats:
        print(f"Total output rows : {sum(s['n_out'] for s in all_stats):,}")
        print(f"Total wall time   : {sum(s['elapsed_s'] for s in all_stats):.0f}s")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()