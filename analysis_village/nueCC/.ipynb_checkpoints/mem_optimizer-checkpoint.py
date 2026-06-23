"""
nue_memory_utils.py
-------------------
Memory-efficient loading utilities for the nueCC selection notebook.

Problem
-------
evtdf for the production mc1e20_nueCC.df file has ~2.1M particle rows × 393
MultiIndex-tuple columns.  Raw float data is ~6.6 GB, but pandas loading
overhead (intermediate copies, MultiIndex object metadata, glibc page
fragmentation) multiplies this to 100+ GB RSS.

The 393 columns break down roughly as:

    Group                        Count   Needed for selection?
    ─────────────────────────────────────────────────────────
    rec -> dlp -> (interaction)  ~30     YES  (FM, FV, vertex)
    rec -> dlp -> particles      ~65     YES  (all PID/score/KE cuts)
    rec -> dlp_true -> (inter)   ~65     NO*  (truth_info already has all this)
    rec -> dlp_true -> particles ~85     PARTIAL (only pid/ke/momentum/is_primary)
    mcnu -> *                    ~90     NO   (neutrino truth, not used in cuts)
    index-merge artifacts         ~3     YES  (keep)
    ─────────────────────────────────────────────────────────
    Total                        ~393

* truth_info_0 already contains is_nu, is_cc, is_fv_true, passed_fm,
  passed_fv_reco, true_leading_e_ke/costheta/p — all recomputed from
  dlp_true interaction columns inside make_nueCC_df.py.  Loading them
  again from evtdf is redundant.

Modes
-----
mode="reco_min_true"  (DEFAULT, RECOMMENDED)
    Keep: all reco (dlp) cols + minimal dlp_true particle cols (pid, ke,
           momentum x/y/z, p, is_primary, mct_index, nu_id, interaction_id,
           pdg_code, shape, is_valid)
    Drop: all mcnu cols + most dlp_true particle tracking cols
           + all dlp_true interaction cols (use truth_info instead)
    Columns: 393 → ~145  |  Expected memory: ~20–35 GB  (was 100+ GB)
    Compatible with: ns.true_leading_electron_*(), nh.true_particles()

mode="reco_only"
    Keep: all reco (dlp) cols only
    Drop: ALL dlp_true + ALL mcnu
    Columns: 393 → ~95  |  Expected memory: ~10–18 GB
    BREAKS: ns.true_leading_electron_*(), nh.true_particles()
    Use truth_info directly for true kinematics (see notebook patch below).

Usage in Notebook 1
-------------------
Replace:
    evtdf = pd.read_hdf(MERGED_FILE, key="evt_0")

With:
    from nue_memory_utils import load_evtdf_slim, free_evtdf
    evtdf = load_evtdf_slim(MERGED_FILE, key="evt_0", mode="reco_min_true")

And replace the true-kinematics arguments in build_and_save_selection:
    # OLD (forces all dlp_true particle cols to be loaded):
    true_ke  = ns.true_leading_electron_ke(evtdf),
    true_p   = ns.true_leading_electron_p(evtdf),
    true_cos = ns.true_leading_electron_costheta(evtdf),

    # NEW (uses precomputed truth_info values — identical result, zero extra cols):
    true_ke  = truth_info["true_leading_e_ke"].reindex(presel_inter),
    true_p   = truth_info["true_leading_e_p"].reindex(presel_inter),
    true_cos = truth_info["true_leading_e_costheta"].reindex(presel_inter),

After sel_topo is saved, call:
    free_evtdf(evtdf_local=locals())   # frees evtdf + returns ~20-35 GB
"""

import gc
import ctypes
import warnings
import numpy as np
import pandas as pd
from typing import Optional

__all__ = ["load_evtdf_slim", "free_evtdf", "report_memory", "col_group_summary"]

# ── Minimal dlp_true particle fields to keep in "reco_min_true" mode ──────────
# These are the only dlp_true particle fields used by:
#   ns.true_leading_electron_ke / p / costheta
#   nh.true_particles(evtdf).pid  etc.
_TRUE_PARTICLE_KEEP = frozenset([
    "pid",
    "is_primary",
    "ke",
    "p",
    "momentum",      # matches momentum -> x / y / z
    "interaction_id",
    "nu_id",
    "mct_index",
    "pdg_code",
    "shape",
    "is_valid",
    "is_contained",
    "is_matched",
    "group_primary",
])


def _col_parts(col) -> list:
    """Return non-empty string parts of a column label."""
    if isinstance(col, tuple):
        return [str(p) for p in col if str(p) != ""]
    return [str(col)] if col else []


def _col_group(col) -> str:
    """
    Classify a column into one of:
        dlp_reco_interaction   rec -> dlp -> (non-particles)
        dlp_reco_particle      rec -> dlp -> particles -> *
        dlp_true_interaction   rec -> dlp_true -> (non-particles)
        dlp_true_particle      rec -> dlp_true -> particles -> *
        mcnu                   mcnu -> *
        index_artifact         rec.dlp_true..index_*, rec.mc.nu..index, etc.
        other
    """
    parts = _col_parts(col)
    if not parts:
        return "other"

    p0 = parts[0]

    if p0 == "mcnu":
        return "mcnu"

    # index-merge artifact columns written by cafpyana's multicol_merge
    # e.g. "rec.dlp_true..index_x", "rec.mc.nu..index"
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

    # Always keep reco columns and index artifacts
    if group in ("dlp_reco_particle", "dlp_reco_interaction",
                 "index_artifact", "other"):
        return True

    # Always drop mcnu (neutrino-level truth, never used in selection)
    if group == "mcnu":
        return False

    if mode == "reco_only":
        # Drop everything that isn't pure reco
        return False

    if mode == "reco_min_true":
        if group == "dlp_true_particle":
            parts = _col_parts(col)
            # parts: ['rec', 'dlp_true', 'particles', field, ...]
            field = parts[3] if len(parts) > 3 else ""
            return field in _TRUE_PARTICLE_KEEP
        if group == "dlp_true_interaction":
            # truth_info_0 already contains everything from here;
            # drop the full interaction-level truth block in evtdf.
            return False
        return False

    # Default: keep everything
    return True


def load_evtdf_slim(
    path: str,
    key: str = "evt_0",
    mode: str = "reco_min_true",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load evtdf from HDF5, dropping columns not needed for the nueCC selection.

    Parameters
    ----------
    path    : path to the HDF5 merged file
    key     : HDF5 key for the particle-level df (default "evt_0")
    mode    : "reco_min_true" (default) or "reco_only" — see module docstring
    verbose : print column counts and group breakdown

    Returns
    -------
    pd.DataFrame with the same MultiIndex structure as the full evtdf,
    but with a reduced column set.

    Notes
    -----
    This function tries pd.HDFStore.select() with column projection first
    (works for table-format HDF5, avoids loading dropped columns entirely).
    If the store is in fixed format, it falls back to loading the full df
    then dropping columns in-place — still saves memory vs keeping them.
    """
    with pd.HDFStore(path, mode="r") as store:
        storer = store.get_storer(key)
        if storer is None:
            raise KeyError(f"Key {key!r} not found in {path!r}")

        # ── Discover all column labels ────────────────────────────────────────
        is_table = getattr(storer, "is_table", False)
        try:
            all_cols = list(storer.non_index_axes[0][1])
        except (AttributeError, IndexError, TypeError):
            all_cols = None

        if all_cols is None:
            # Fallback: read one row to get columns
            _sample = store.select(key, start=0, stop=1) if is_table else store[key]
            all_cols = list(_sample.columns)
            del _sample

        needed = [c for c in all_cols if _should_keep(c, mode)]
        dropped = len(all_cols) - len(needed)

        # ── Group-level breakdown for verbose output ──────────────────────────
        if verbose:
            from collections import Counter
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

        # ── Load with column projection (table format) or post-drop (fixed) ──
        if is_table and all_cols is not None:
            try:
                df = store.select(key, columns=needed)
                if verbose:
                    print(f"  Loaded via select() column projection  ✓")
                return df
            except Exception as e:
                warnings.warn(
                    f"load_evtdf_slim: select() with columns failed ({e}), "
                    "falling back to full load + post-drop."
                )

        # Fallback: load full df, then drop
        if verbose:
            print("  Loading full df then dropping columns (fixed-format HDF5) …")
        df = store[key]
        cols_to_drop = [c for c in df.columns if not _should_keep(c, mode)]
        df.drop(columns=cols_to_drop, inplace=True)
        if verbose:
            print(f"  Post-drop complete. Final shape: {df.shape}")
        return df


def free_evtdf(evtdf_local: Optional[dict] = None, extra_vars: list = None):
    """
    Aggressively free evtdf and optional extras from memory.

    Parameters
    ----------
    evtdf_local  : pass locals() from the calling scope so this function can
                   del 'evtdf', 'rp', 'tp', 'ri', 'ti_view' from that scope.
                   If None, only gc.collect() + malloc_trim are called.
    extra_vars   : additional variable names to del from evtdf_local

    Example
    -------
    # After sel_topo is saved:
    free_evtdf(evtdf_local=locals())
    # Then manually: del evtdf, rp, tp, ri, ti_view  (Python scope limitation)
    """
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
        print("free_evtdf: gc done (malloc_trim not available on this platform).")


def report_memory(label: str = ""):
    """Print current process RSS via /proc/self/status (Linux only)."""
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
    """Return a summary DataFrame of column counts by group (diagnostic helper)."""
    from collections import Counter
    counts = Counter(_col_group(c) for c in df.columns)
    return pd.DataFrame(
        [(g, n) for g, n in sorted(counts.items(), key=lambda x: -x[1])],
        columns=["group", "n_cols"],
    )