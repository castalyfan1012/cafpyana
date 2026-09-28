"""
mem_optimizer.py  (a.k.a. nue_memory_utils.py)
----------------------------------------------
Memory-efficient loading utilities for the nueCC selection notebook.

Supports HDF5 files with either:
  - Single key per type  (evt_0, truth_info_0)           — old merge_df.py
  - Multiple sub-keys    (evt_0..evt_N, truth_info_0..N)  — merge_v2.py

What changed vs. the previous version
-------------------------------------
1. Per-sub-key dtype downcast (float64 -> float32, ints -> smallest int)
   BEFORE concatenation.  evtdf is ~all float64, so this alone halves it,
   and because it happens per part the concat peak is also halved.
2. Optional `drop_fields`: removes wide per-PID arrays that the selection
   never reads (chi2_per_pid, csda_ke_per_pid, mcs_ke_per_pid,
   particle_counts, primary_particle_counts) -> 30 fewer columns.
3. `load_frac`: load only a fraction of the sub-keys (e.g. 60 or 0.6 = 60%).
   Sub-keys are chosen BY SUFFIX NUMBER, evenly spread over the range, so
   evt_i / truth_info_i / histpotdf_i stay aligned when every loader gets
   the same load_frac.  ALWAYS pass the same load_frac to the evtdf,
   truth_info AND histpotdf loads, otherwise POT normalisation is wrong.
4. `columns` for load_all_subkeys: keep only the truth_info columns you use.
"""

import gc
import ctypes
import math
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Iterable

__all__ = [
    "load_evtdf_slim", "load_all_subkeys", "free_evtdf",
    "report_memory", "col_group_summary", "select_subkeys",
    "DEFAULT_DROP_FIELDS",
]

# ── Minimal dlp_true particle fields to keep in "reco_min_true" mode ──────────
_TRUE_PARTICLE_KEEP = frozenset([
    "pid", "is_primary", "ke", "p", "momentum",
    "interaction_id", "nu_id", "mct_index", "pdg_code",
    "shape", "is_valid", "is_contained", "is_matched", "group_primary",
])

# ── Reco fields not used anywhere in the nueCC selection notebook ─────────────
# Each of these is a 6-wide per-PID array (I0..I5).  Dropping them removes
# 30 of the 124 columns.  Pass drop_fields=() to keep everything.
DEFAULT_DROP_FIELDS = frozenset([
    "chi2_per_pid",             # reco particle
    "csda_ke_per_pid",          # reco particle
    "mcs_ke_per_pid",           # reco particle
    "particle_counts",          # reco interaction
    "primary_particle_counts",  # reco interaction
])


def _col_parts(col) -> list:
    if isinstance(col, tuple):
        return [str(p) for p in col if str(p) != ""]
    return [str(col)] if col else []


def _col_group(col) -> str:
    parts = _col_parts(col)
    if not parts:
        return "other"
    p0 = parts[0]
    if p0 == "mcnu":
        return "mcnu"
    if "." in p0:
        return "index_artifact"
    if p0 == "rec" and len(parts) >= 2:
        branch = parts[1]
        if branch == "dlp":
            if len(parts) >= 3 and parts[2] == "particles":
                return "dlp_reco_particle"
            return "dlp_reco_interaction"
        if branch == "dlp_true":
            if len(parts) >= 3 and parts[2] == "particles":
                return "dlp_true_particle"
            return "dlp_true_interaction"
    return "other"


def _col_field(col) -> str:
    """Leaf field name, e.g. ('rec','dlp','particles','chi2_per_pid','I0') -> 'chi2_per_pid'."""
    parts = _col_parts(col)
    group = _col_group(col)
    if group in ("dlp_reco_particle", "dlp_true_particle"):
        return parts[3] if len(parts) > 3 else ""
    if group in ("dlp_reco_interaction", "dlp_true_interaction"):
        return parts[2] if len(parts) > 2 else ""
    return ""


def _should_keep(col, mode: str, drop_fields: Iterable[str] = ()) -> bool:
    group = _col_group(col)
    if drop_fields and group in ("dlp_reco_particle", "dlp_reco_interaction"):
        if _col_field(col) in drop_fields:
            return False
    if group in ("dlp_reco_particle", "dlp_reco_interaction",
                 "index_artifact", "other"):
        return True
    if group == "mcnu":
        return False
    if mode == "reco_only":
        return False
    if mode == "reco_min_true":
        if group == "dlp_true_particle":
            return _col_field(col) in _TRUE_PARTICLE_KEEP
        return False
    return True


# ── dtype downcast ────────────────────────────────────────────────────────────

_INT_CANDIDATES  = (np.int8, np.int16, np.int32)
_UINT_CANDIDATES = (np.uint8, np.uint16, np.uint32)


def _target_dtype(s: pd.Series, float_dtype):
    dt = s.dtype
    if dt == np.float64:
        return np.dtype(float_dtype)
    if len(s) == 0:
        return None
    if pd.api.types.is_signed_integer_dtype(dt) and dt.itemsize > 1:
        cands = _INT_CANDIDATES
    elif pd.api.types.is_unsigned_integer_dtype(dt) and dt.itemsize > 1:
        cands = _UINT_CANDIDATES
    else:
        return None
    lo, hi = s.min(), s.max()
    for c in cands:
        ii = np.iinfo(c)
        if lo >= ii.min and hi <= ii.max:
            return np.dtype(c) if np.dtype(c).itemsize < dt.itemsize else None
    return None


def _downcast_df(df: pd.DataFrame, float_dtype=np.float32) -> pd.DataFrame:
    """Column-by-column downcast (low transient memory). Values are unchanged
    except float64 -> float32 rounding (~7 significant digits)."""
    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        t = _target_dtype(s, float_dtype)
        if t is None:
            continue
        if hasattr(df, "isetitem"):          # pandas >= 1.5
            df.isetitem(i, s.astype(t))
        else:                                # very old pandas fallback
            col = df.columns[i]
            df[col] = s.astype(t)
    return df


# ── Multi-key helpers ─────────────────────────────────────────────────────────

def _discover_subkeys(path, prefix):
    """Sorted (numerically) HDF5 keys matching prefix, e.g. 'evt' → ['/evt_0', '/evt_1', ...]."""
    with pd.HDFStore(path, mode='r') as store:
        keys = [k for k in store.keys()
                if k.lstrip('/').startswith(prefix + '_')
                and k.lstrip('/')[len(prefix) + 1:].isdigit()]
    return sorted(keys, key=lambda k: int(k.rsplit('_', 1)[-1]))


def _normalise_frac(load_frac):
    """None/100/1.0 -> None (load all); 60 -> 0.6; 0.6 -> 0.6."""
    if load_frac is None:
        return None
    f = float(load_frac)
    if f > 1.0:
        f = f / 100.0
    if not (0.0 < f <= 1.0):
        raise ValueError(f"load_frac must be in (0, 1] or (0, 100]; got {load_frac!r}")
    return None if f >= 1.0 else f


def select_subkeys(keys, load_frac=None):
    """
    Choose a subset of sub-keys, spread evenly over the suffix range.

    Deterministic given (number of keys, load_frac), and selection is done on
    the numeric suffix, so evt_i / truth_info_i / histpotdf_i are picked
    consistently as long as each prefix has the same set of suffixes.
    """
    f = _normalise_frac(load_frac)
    if f is None or len(keys) <= 1:
        return list(keys)
    n = len(keys)
    k = max(1, math.ceil(f * n))
    pos = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    return [keys[p] for p in pos]


def _keys_for(path, prefix, load_frac):
    all_keys = _discover_subkeys(path, prefix)
    if not all_keys:
        raise KeyError(f"No keys matching '{prefix}_*' in {path}")
    return all_keys, select_subkeys(all_keys, load_frac)


def load_all_subkeys(path, prefix, verbose=True, load_frac=None,
                     columns=None, downcast=False):
    """
    Load and concatenate sub-keys for a given prefix.

    Parameters
    ----------
    path      : HDF5 file path
    prefix    : key prefix without trailing _N (e.g. 'truth_info', 'histpotdf')
    verbose   : print progress
    load_frac : fraction of sub-keys to load (60 or 0.6 → 60%). None = all.
                Use the SAME value as for load_evtdf_slim.
    columns   : optional list of columns to keep (dropped per part, before concat)
    downcast  : also downcast dtypes per part (float64 → float32 etc.)
    """
    all_keys, keys = _keys_for(path, prefix, load_frac)

    if verbose and len(all_keys) > 1:
        sel = [int(k.rsplit('_', 1)[-1]) for k in keys]
        tag = "" if len(keys) == len(all_keys) else f"  (load_frac → suffixes {sel})"
        print(f"  Loading {len(keys)}/{len(all_keys)} sub-keys for '{prefix}'{tag} ...")

    parts = []
    with pd.HDFStore(path, mode='r') as store:
        for k in keys:
            df = store[k]
            if columns is not None and isinstance(df, pd.DataFrame):
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise KeyError(f"{k}: columns {missing} not found")
                df = df[list(columns)]
            if downcast and isinstance(df, pd.DataFrame):
                df = _downcast_df(df)
            parts.append(df)

    result = pd.concat(parts) if len(parts) > 1 else parts[0]
    del parts
    gc.collect()

    if verbose:
        mem = result.memory_usage(deep=False, index=True)
        mem = mem.sum() if hasattr(mem, "sum") else mem
        print(f"  {prefix}: shape={result.shape}  (~{mem/1e9:.2f} GB)")
    return result


# ── Main loader ───────────────────────────────────────────────────────────────

def load_evtdf_slim(
    path: str,
    key: Optional[str] = "evt_0",
    mode: str = "reco_min_true",
    verbose: bool = True,
    load_frac=None,
    downcast: bool = True,
    drop_fields: Iterable[str] = DEFAULT_DROP_FIELDS,
) -> pd.DataFrame:
    """
    Load evtdf from HDF5, dropping unneeded columns and downcasting dtypes
    per sub-key before concatenation.

    Parameters
    ----------
    path        : path to the HDF5 merged file
    key         : "evt_0" → only that sub-key; None → all evt_* sub-keys
    mode        : "reco_min_true" (default) or "reco_only"
    verbose     : print column summary and memory
    load_frac   : fraction of evt_* sub-keys to load (60 or 0.6 → 60%).
                  Only used when key=None.  Pass the SAME value to the
                  truth_info and histpotdf loads.
    downcast    : float64 → float32, ints → smallest int (default True)
    drop_fields : reco field names to drop (default DEFAULT_DROP_FIELDS);
                  pass () to keep every column.
    """
    drop_fields = frozenset(drop_fields or ())

    if key is None:
        all_keys, keys = _keys_for(path, 'evt', load_frac)
    else:
        all_keys = keys = [f'/{key}' if not key.startswith('/') else key]

    if verbose:
        sel = "" if len(keys) == len(all_keys) else \
            f"  load_frac → suffixes {[int(k.rsplit('_', 1)[-1]) for k in keys]}"
        print(f"\nload_evtdf_slim  mode={mode!r}  keys={len(keys)}/{len(all_keys)} "
              f"downcast={downcast}{sel}")

    parts = []
    keep_cols = None
    n_rows = 0
    for i, k in enumerate(keys):
        with pd.HDFStore(path, mode='r') as store:
            df = store[k]
        if keep_cols is None:
            all_cols = list(df.columns)
            keep_cols = [c for c in all_cols if _should_keep(c, mode, drop_fields)]
            if verbose:
                _print_col_summary_from_cols(all_cols, mode, k, drop_fields)
        drop = [c for c in df.columns if c not in set(keep_cols)]
        if drop:
            df = df.drop(columns=drop)
        if downcast:
            df = _downcast_df(df)
        n_rows += len(df)
        parts.append(df)
        del df
        gc.collect()
        if verbose:
            print(f"  [{i+1:>2}/{len(keys)}] {k:<8} rows={len(parts[-1]):>9,}  "
                  f"cum={n_rows:>11,}", end="")
            report_memory(inline=True)

    if len(parts) == 1:
        result = parts[0]
    else:
        if verbose:
            print(f"  Concatenating {len(parts)} parts ({n_rows:,} rows) ...")
        result = pd.concat(parts)
    del parts
    _trim()

    if verbose:
        mem = result.memory_usage(deep=False, index=True).sum()
        print(f"  Final shape: {result.shape}  (~{mem/1e9:.2f} GB)")
        report_memory("after evtdf load")
    return result


def _print_col_summary_from_cols(all_cols, mode, key, drop_fields=()):
    """Print column group breakdown."""
    from collections import Counter
    needed = [c for c in all_cols if _should_keep(c, mode, drop_fields)]
    dropped = len(all_cols) - len(needed)
    group_counts = Counter(_col_group(c) for c in all_cols)
    keep_counts  = Counter(_col_group(c) for c in needed)

    print(f"  columns from {key!r}:")
    print(f"  Total cols : {len(all_cols):>4}  →  kept: {len(needed):>4}  "
          f"dropped: {dropped:>4}  ({100*dropped/max(len(all_cols),1):.0f}% reduced)")
    print(f"  {'Group':<30} {'all':>5}  {'kept':>5}")
    print(f"  {'-'*42}")
    for g in ("dlp_reco_interaction", "dlp_reco_particle",
              "dlp_true_interaction", "dlp_true_particle",
              "mcnu", "index_artifact", "other"):
        tot = group_counts.get(g, 0)
        kpt = keep_counts.get(g, 0)
        flag = "" if tot == kpt else f"  ← dropped {tot - kpt}"
        print(f"  {g:<30} {tot:>5}  {kpt:>5}{flag}")
    print()


# ── Memory management ────────────────────────────────────────────────────────

def _trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def free_evtdf(evtdf_local: Optional[dict] = None, extra_vars: list = None):
    if evtdf_local is not None:
        candidates = ["evtdf", "rp", "tp", "ri", "ti_view"] + (extra_vars or [])
        for name in candidates:
            if name in evtdf_local:
                del evtdf_local[name]
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print("free_evtdf: gc + malloc_trim(0) done.")
    except Exception:
        print("free_evtdf: gc done (malloc_trim not available).")


def report_memory(label: str = "", inline: bool = False):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    kb = int(line.split()[1])
                    if inline:
                        print(f"  RSS={kb/1e6:.2f} GB")
                    else:
                        tag = f"[{label}] " if label else ""
                        print(f"  {tag}RSS = {kb/1e6:.2f} GB")
                    return
    except Exception:
        pass
    print("" if inline else "  report_memory: /proc/self/status not available")


def col_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    counts = Counter(_col_group(c) for c in df.columns)
    return pd.DataFrame(
        [(g, n) for g, n in sorted(counts.items(), key=lambda x: -x[1])],
        columns=["group", "n_cols"],
    )