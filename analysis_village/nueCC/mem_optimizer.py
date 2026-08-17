"""
nue_memory_utils.py  (a.k.a. mem_optimizer.py)
-------------------
Memory-efficient loading utilities for the nueCC selection notebook.

Supports HDF5 files with either:
  - Single key per type  (evt_0, truth_info_0)           — old merge_df.py
  - Multiple sub-keys    (evt_0..evt_9, truth_info_0..9)  — merge_v2.py
"""

import gc
import ctypes
import warnings
import numpy as np
import pandas as pd
from typing import Optional

__all__ = [
    "load_evtdf_slim", "load_all_subkeys", "free_evtdf",
    "report_memory", "col_group_summary",
]

# ── Minimal dlp_true particle fields to keep in "reco_min_true" mode ──────────
_TRUE_PARTICLE_KEEP = frozenset([
    "pid", "is_primary", "ke", "p", "momentum",
    "interaction_id", "nu_id", "mct_index", "pdg_code",
    "shape", "is_valid", "is_contained", "is_matched", "group_primary",
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


def _should_keep(col, mode: str) -> bool:
    group = _col_group(col)
    if group in ("dlp_reco_particle", "dlp_reco_interaction",
                 "index_artifact", "other"):
        return True
    if group == "mcnu":
        return False
    if mode == "reco_only":
        return False
    if mode == "reco_min_true":
        if group == "dlp_true_particle":
            parts = _col_parts(col)
            field = parts[3] if len(parts) > 3 else ""
            return field in _TRUE_PARTICLE_KEEP
        if group == "dlp_true_interaction":
            return False
        return False
    return True


# ── Multi-key helpers ─────────────────────────────────────────────────────────

def _discover_subkeys(path, prefix):
    """Return sorted list of HDF5 keys matching prefix (e.g. 'evt' → ['/evt_0', '/evt_1', ...])."""
    with pd.HDFStore(path, mode='r') as store:
        return sorted(k for k in store.keys()
                      if k.lstrip('/').startswith(prefix + '_')
                      and k.lstrip('/')[len(prefix)+1:].isdigit())


def load_all_subkeys(path, prefix, verbose=True):
    """
    Load and concatenate all sub-keys for a given prefix.

    Handles both single-key (evt_0) and multi-key (evt_0..evt_9) files.
    For small key types (histpotdf, hdr, pot) this is fine to call directly.

    Parameters
    ----------
    path    : HDF5 file path
    prefix  : key prefix without trailing _N (e.g. 'truth_info', 'histpotdf')
    verbose : print progress
    """
    keys = _discover_subkeys(path, prefix)
    if not keys:
        raise KeyError(f"No keys matching '{prefix}_*' in {path}")

    if verbose and len(keys) > 1:
        print(f"  Loading {len(keys)} sub-keys for '{prefix}' ...")

    parts = []
    with pd.HDFStore(path, mode='r') as store:
        for k in keys:
            parts.append(store[k])

    result = pd.concat(parts) if len(parts) > 1 else parts[0]
    del parts; gc.collect()

    if verbose:
        print(f"  {prefix}: shape={result.shape}")
    return result


# ── Main loader ───────────────────────────────────────────────────────────────

def load_evtdf_slim(
    path: str,
    key: str = "evt_0",
    mode: str = "reco_min_true",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load evtdf from HDF5, dropping columns not needed for the nueCC selection.

    Supports multi-key files from merge_v2.py:
      - key="evt_0"  → load only that one sub-key (backward compatible)
      - key=None      → auto-detect and load ALL evt_* sub-keys

    Parameters
    ----------
    path    : path to the HDF5 merged file
    key     : HDF5 key (default "evt_0"), or None to auto-detect all evt_* keys
    mode    : "reco_min_true" (default) or "reco_only"
    verbose : print column counts and group breakdown
    """
    # ── Auto-detect multi-key ─────────────────────────────────────────────
    if key is None:
        keys = _discover_subkeys(path, 'evt')
        if not keys:
            raise KeyError(f"No 'evt_*' keys found in {path}")
    else:
        keys = [f'/{key}' if not key.startswith('/') else key]

    if verbose and len(keys) > 1:
        print(f"\nload_evtdf_slim  mode={mode!r}  keys={len(keys)} sub-keys")

    # ── Load first key to discover columns and print summary ──────────────
    first_df = _load_one_slim(path, keys[0], mode, verbose=(verbose and len(keys) == 1))

    if verbose and len(keys) > 1:
        # Print column summary once (same for all sub-keys)
        _print_col_summary(first_df, path, keys[0], mode)

    if len(keys) == 1:
        return first_df

    # ── Load remaining sub-keys ───────────────────────────────────────────
    parts = [first_df]
    cols_to_keep = list(first_df.columns)  # reuse same column set

    for k in keys[1:]:
        part = _load_one_slim(path, k, mode, verbose=False, force_cols=cols_to_keep)
        parts.append(part)

    if verbose:
        total_rows = sum(len(p) for p in parts)
        print(f"  Concatenating {len(parts)} parts ({total_rows:,} total rows) ...")

    result = pd.concat(parts)
    del parts; gc.collect()

    if verbose:
        print(f"  Final shape: {result.shape}")

    return result


def _load_one_slim(path, key, mode, verbose=True, force_cols=None):
    """Load a single HDF5 key with column slimming."""
    with pd.HDFStore(path, mode='r') as store:
        storer = store.get_storer(key)
        if storer is None:
            raise KeyError(f"Key {key!r} not found in {path!r}")

        is_table = getattr(storer, "is_table", False)

        if force_cols is not None:
            # Fast path: we already know which columns to keep
            if verbose:
                print(f"  Loading {key} (fixed cols) ...")
            df = store[key]
            drop = [c for c in df.columns if c not in set(force_cols)]
            if drop:
                df.drop(columns=drop, inplace=True)
            return df

        # Discover columns
        try:
            all_cols = list(storer.non_index_axes[0][1])
        except (AttributeError, IndexError, TypeError):
            all_cols = None

        if all_cols is None:
            _sample = store.select(key, start=0, stop=1) if is_table else store[key]
            all_cols = list(_sample.columns)
            if not is_table:
                # We already loaded the full df
                cols_to_drop = [c for c in all_cols if not _should_keep(c, mode)]
                _sample.drop(columns=cols_to_drop, inplace=True)
                if verbose:
                    _print_col_summary_from_cols(all_cols, mode, key)
                    print(f"  Post-drop complete. Final shape: {_sample.shape}")
                return _sample
            del _sample

        needed = [c for c in all_cols if _should_keep(c, mode)]

        if verbose:
            _print_col_summary_from_cols(all_cols, mode, key)

        # Try table-format select with column projection
        if is_table:
            try:
                return store.select(key, columns=needed)
            except Exception:
                pass

        # Fixed format: load then drop
        if verbose:
            print("  Loading full df then dropping columns (fixed-format HDF5) …")
        df = store[key]
        cols_to_drop = [c for c in df.columns if not _should_keep(c, mode)]
        df.drop(columns=cols_to_drop, inplace=True)
        if verbose:
            print(f"  Post-drop complete. Final shape: {df.shape}")
        return df


def _print_col_summary(df, path, key, mode):
    """Print column group summary from an already-loaded DataFrame."""
    # Get original column count from the file
    with pd.HDFStore(path, mode='r') as store:
        storer = store.get_storer(key)
        try:
            all_cols = list(storer.non_index_axes[0][1])
        except (AttributeError, IndexError, TypeError):
            all_cols = list(df.columns)  # can't determine original count
    _print_col_summary_from_cols(all_cols, mode, key)
    print(f"  Post-drop shape per sub-key: ~{df.shape}")


def _print_col_summary_from_cols(all_cols, mode, key):
    """Print column group breakdown."""
    from collections import Counter
    needed = [c for c in all_cols if _should_keep(c, mode)]
    dropped = len(all_cols) - len(needed)
    group_counts = Counter(_col_group(c) for c in all_cols)
    keep_counts  = Counter(_col_group(c) for c in needed)

    print(f"\nload_evtdf_slim  mode={mode!r}  key={key!r}")
    print(f"  Total cols : {len(all_cols):>4}  →  kept: {len(needed):>4}  "
          f"dropped: {dropped:>4}  ({100*dropped/len(all_cols):.0f}% reduced)")
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


def report_memory(label: str = ""):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    kb = int(line.split()[1])
                    tag = f"[{label}] " if label else ""
                    print(f"  {tag}RSS = {kb/1e6:.2f} GB")
                    return
    except Exception:
        pass
    print("  report_memory: /proc/self/status not available")


def col_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    counts = Counter(_col_group(c) for c in df.columns)
    return pd.DataFrame(
        [(g, n) for g, n in sorted(counts.items(), key=lambda x: -x[1])],
        columns=["group", "n_cols"],
    )