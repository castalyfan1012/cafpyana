#!/usr/bin/env python3
"""
merge_v2.py — Memory-efficient HDF5 merger for EAF (XRootD).

Each file fetched exactly TWICE (pass 1: scan index, pass 2: read+write).
Each key type flushes to disk independently every WRITE_BATCH chunks.
Peak memory ≈ WRITE_BATCH × largest-single-chunk-size per key type.

Usage: 
1. Produce shared filelist: bash make_shared_filelist.sh (on SBND machine)
2. Change the config in this script
3. Run the script: python3 merge_v2.py [--nfiles] (on EAF)
"""

import argparse, ctypes, glob, gc, os, re, shutil, subprocess, sys
import tempfile, time, warnings
import numpy as np, pandas as pd
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

# ── lowE evtdf ────────────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_12_180822__mc1e20_lowE_v2/mc*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_lowE_v2.df"
# JOB_LIST      = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/lowE_joblist.txt"

# ── lowE sys ──────────────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_12_180953__mc1e20_lowE_sys_v2/mc*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_lowE_sys_v2.df"
# JOB_LIST      = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/lowE_joblist.txt"

# ── main MC evtdf ─────────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_10_165955__mc1e20_nueCC_v2-2/mc*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_nueCC_v2-2.df"
# JOB_LIST      = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/nueCC_joblist.txt"

# ── main MC sys ───────────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_10_170310__mc1e20_nueCC_sys_v2-2/mc*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_nueCC_sys_v2-2.df"
# JOB_LIST      = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/nueCC_joblist.txt"

# ── data (on-beam) ────────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_10_171943__data_dev_v2-2/data*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/data_dev_v2-2.df"
# JOB_LIST      = None

# ── data (off-beam) ───────────────────────────────────────────────────────
# INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_10_172144__data_offbeamlight_v2-2/data*.df"
# OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/data_offbeamlight_v2-2.df"
# JOB_LIST      = None

# ── ACTIVE ────────────────────────────────────────────────────────────────
INPUT_PATTERN = "/pnfs/sbnd/scratch/users/castalyf/cafpyana_out/dfs/2026_08_10_170310__mc1e20_nueCC_sys_v2-2/mc*.df"
OUTPUT_FILE   = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/mc1e20_nueCC_sys_v2-2.df"
JOB_LIST      = "/exp/sbnd/data/users/castalyf/nue_sel/production_files/gen1/nueCC_joblist.txt"

XROOTD_REDIRECTOR = "root://fndca1.fnal.gov:1094/"
WRITE_BATCH = 50

# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default=INPUT_PATTERN)
    p.add_argument("--output",   default=OUTPUT_FILE)
    p.add_argument("--job-list", default=JOB_LIST)
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--nfiles",   type=int, default=None)
    p.add_argument("--xrootd",   default=XROOTD_REDIRECTOR)
    p.add_argument("--local",    action="store_true")
    p.add_argument("--write-batch", type=int, default=WRITE_BATCH)
    return p.parse_args()

def _extract_job_index(fn):
    m = re.search(r'_(\d+)\.df$', os.path.basename(fn))
    return int(m.group(1)) if m else None

def pnfs_to_xrootd(p, r):
    if p.startswith("/pnfs/sbnd/"): return f"{r}/pnfs/fnal.gov/usr/sbnd/{p[11:]}"
    if p.startswith("/pnfs/"): return f"{r}/pnfs/fnal.gov/usr/{p[6:]}"
    return p

def fetch_file(p, tmpdir, r, local=False):
    if local: return p
    lf = os.path.join(tmpdir, os.path.basename(p))
    if os.path.exists(lf): return lf
    url = pnfs_to_xrootd(p, r)
    res = subprocess.run(["xrdcp","-f","--nopbar",url,lf],
                         capture_output=True, text=True, timeout=600)
    if res.returncode: raise RuntimeError(f"xrdcp: {res.stderr.strip()}")
    return lf

def _rm(p):
    try: os.remove(p)
    except OSError: pass

def _free():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass

def _kt(raw):
    parts = raw.rsplit('_',1)
    return parts[0] if len(parts)==2 and parts[1].isdigit() else None

def _remap(df, fi, mp):
    if not isinstance(df.index, pd.MultiIndex):
        o = df.copy()
        o.index = pd.Index(np.full(len(df), mp[(fi,0)], dtype=np.int64), name=df.index.name)
        return o
    l0 = df.index.get_level_values(0).to_numpy()
    new = np.array([mp[(fi,int(v))] for v in l0], dtype=np.int64)
    o = df.copy()
    o.index = pd.MultiIndex.from_arrays(
        [new]+[df.index.get_level_values(n) for n in range(1,df.index.nlevels)],
        names=df.index.names)
    return o

def _flush(acc, output, kt, si):
    if not acc: return si, 0
    b = pd.concat(acc)
    b.to_hdf(output, key=f"{kt}_{si}", mode='a', complevel=5, complib='blosc', format='fixed')
    nr = len(b); del b; _free()
    return si+1, nr


def main():
    args = parse_args()
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Job list: {args.job_list or '(none)'}")
    print(f"Mode   : {'local' if args.local else 'xrdcp'}  batch={args.write_batch}\n")

    mc_files_raw = sorted(glob.glob(args.input))
    input_dir = os.path.dirname(args.input)

    if not mc_files_raw and args.job_list:
        prefix = os.path.basename(args.input).split('*')[0]
        xrd_dir = pnfs_to_xrootd(input_dir, args.xrootd)
        print(f"Listing via xrdfs ...")
        r = subprocess.run(
            ["xrdfs", xrd_dir.split("//")[0]+"//"+xrd_dir.split("//")[1],
             "ls", "/"+"/".join(xrd_dir.split("//")[2:])],
            capture_output=True, text=True, timeout=120)
        if r.returncode==0 and r.stdout.strip():
            mc_files_raw = sorted([
                os.path.join(input_dir, os.path.basename(p))
                for p in r.stdout.strip().split('\n')
                if os.path.basename(p).startswith(prefix) and p.endswith('.df')])
            print(f"Found {len(mc_files_raw)} files")

    if not mc_files_raw and not args.job_list:
        sys.exit("ERROR: no files found.")

    if args.job_list:
        with open(args.job_list) as fh:
            cset = {int(l.strip()) for l in fh if l.strip()}
        mc_file_map = {_extract_job_index(f): f for f in mc_files_raw
                       if _extract_job_index(f) in cset} if mc_files_raw else \
                      {ji: os.path.join(input_dir, f"{os.path.basename(args.input).split('*')[0]}{ji}.df")
                       for ji in sorted(cset)}
        print(f"Job list: {len(cset)}, matched: {len(mc_file_map)}")
    else:
        mc_file_map = {(_extract_job_index(f) or i): f for i,f in enumerate(mc_files_raw)}
        print(f"Files: {len(mc_file_map)}")

    if args.nfiles:
        mc_file_map = dict(sorted(mc_file_map.items())[:args.nfiles])
        print(f"Limited to {len(mc_file_map)}")
    if args.dry_run:
        for ji in sorted(mc_file_map): print(f"  [{ji:>5d}] {mc_file_map[ji]}")
        return

    tmpdir = tempfile.mkdtemp(prefix="merge_v2_")
    print(f"Temp: {tmpdir}")
    try: _run(args, mc_file_map, tmpdir)
    finally: shutil.rmtree(tmpdir, ignore_errors=True); print("Cleaned up.")


def _run(args, mc_file_map, tmpdir):
    t0 = time.time()
    items = sorted(mc_file_map.items())
    WB = args.write_batch

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1 — Scan: key types + shared mapping (one fetch per file)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nPASS 1: Scanning ({len(items)} files)\n{'='*60}")
    key_types = None; all_pairs = set(); failed = set()

    for ji, fp in tqdm(items, desc="Scan"):
        try: local = fetch_file(fp, tmpdir, args.xrootd, args.local)
        except Exception as e:
            tqdm.write(f"  SKIP: {os.path.basename(fp)} — {e}"); failed.add(fp); continue
        try:
            with pd.HDFStore(local, mode='r') as store:
                rk = [k.lstrip('/') for k in store.keys()]
                if key_types is None:
                    key_types = sorted({_kt(k) for k in rk} - {None})
                    print(f"\n  Key types: {key_types}")
                for kt in key_types:
                    for k in rk:
                        if _kt(k) != kt: continue
                        df = store[k]
                        if not isinstance(df.index, pd.MultiIndex):
                            all_pairs.add((ji,0))
                        else:
                            for v in df.index.get_level_values(0).unique():
                                all_pairs.add((ji,int(v)))
                        del df
        except Exception as e:
            tqdm.write(f"  SKIP: {os.path.basename(fp)} — {e}"); failed.add(fp)
        finally:
            if not args.local: _rm(os.path.join(tmpdir, os.path.basename(fp)))
            gc.collect()

    if not key_types: sys.exit("ERROR: no readable files.")
    smap = {p: g for g,p in enumerate(sorted(all_pairs))}
    print(f"\nOK: {len(items)-len(failed)}/{len(items)} files, "
          f"{len(smap)} unique (job,ntuple) pairs")

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2 — Read ALL key types per file, flush each independently
    #          → each file fetched only ONCE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nPASS 2: Read + write ({len(items)} files, batch={WB})\n{'='*60}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if os.path.exists(args.output): os.remove(args.output)

    accs   = {kt: [] for kt in key_types}   # per-key-type accumulator
    counts = {kt: 0  for kt in key_types}   # chunks read
    sub_i  = {kt: 0  for kt in key_types}   # sub-key write index
    totals = {kt: 0  for kt in key_types}   # total rows written

    for ji, fp in tqdm(items, desc="Read+Write"):
        if fp in failed: continue
        try: local = fetch_file(fp, tmpdir, args.xrootd, args.local)
        except: continue
        try:
            with pd.HDFStore(local, mode='r') as store:
                rk = [k.lstrip('/') for k in store.keys()]
                for kt in key_types:
                    for k in sorted(rk):
                        if _kt(k) != kt: continue
                        df = store[k]
                        accs[kt].append(_remap(df, ji, smap))
                        counts[kt] += 1
                        del df
        except Exception as e:
            tqdm.write(f"  SKIP: {os.path.basename(fp)} — {e}")
        finally:
            if not args.local: _rm(os.path.join(tmpdir, os.path.basename(fp)))

        # Flush any key type that hit the batch size
        for kt in key_types:
            if len(accs[kt]) >= WB:
                si, nr = _flush(accs[kt], args.output, kt, sub_i[kt])
                sub_i[kt] = si; totals[kt] += nr; accs[kt] = []
                tqdm.write(f"    flushed {kt}_{si-1}: {nr:,} rows")
                _free()

    # Final flush for each key type
    for kt in key_types:
        if accs[kt]:
            si, nr = _flush(accs[kt], args.output, kt, sub_i[kt])
            sub_i[kt] = si; totals[kt] += nr; accs[kt] = []
            _free()

    # Summary
    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for kt in key_types:
        print(f"  {kt:<16}: {counts[kt]:>4} chunks → {sub_i[kt]:>3} sub-keys, "
              f"{totals[kt]:>12,} rows")
    print(f"\nDone in {elapsed:.1f}s — output: {size_mb:.1f} MB")
    if failed:
        print(f"\nSkipped {len(failed)} file(s):")
        for f in sorted(failed): print(f"  {f}")


if __name__ == "__main__":
    main()