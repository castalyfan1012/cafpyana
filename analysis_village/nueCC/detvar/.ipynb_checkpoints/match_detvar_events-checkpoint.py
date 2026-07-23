#!/usr/bin/env python3
"""
match_detvar_events.py  (Step 2)
---------------------------------
Match events between nominal and each det-var sample on truth-level
physics keys: (run, subrun, evt, neutrino_energy).

Works with your merged .df files (from merge_detvar_dfs.py) which have
the same HDF5 structure as your nominal production:
    evt_0, truth_info_0, hdr_0, pot_0

The matched event subset ensures both samples see the same physics,
so the only difference is the detector response.

Usage:
    python scripts/match_detvar_events.py
    python scripts/match_detvar_events.py --var wiremodxtheta
    python scripts/match_detvar_events.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NUECC_DIR = SCRIPT_DIR.parent
for p in [str(NUECC_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nuecc_detsys_config import (
    NueCCDetsysConfig,
    DET_VARS_ALL,
    build_config,
)

# Match on these truth-level columns (identical across nom/var)
PHYSICS_MATCH_COLS = ["run", "subrun", "evt", "nu_energy"]


def _load_truth_keys(df_path: str, label: str = "") -> pd.DataFrame:
    """
    Extract (run, subrun, evt, nu_energy) from a merged .df file.

    Tries truth_info_0 first (has per-interaction truth metadata),
    then falls back to evt_0 (the particle-level preselected df).
    """
    store = pd.HDFStore(df_path, mode="r")
    keys = [k.strip("/") for k in store.keys()]

    df = None
    source_key = None

    # Prefer truth_info — it has ALL interactions (not just preselected)
    for k in keys:
        if "truth_info" in k:
            df = store[f"/{k}"]
            source_key = k
            break

    # Fall back to evt
    if df is None:
        for k in keys:
            if "evt" in k and "truth" not in k:
                df = store[f"/{k}"]
                source_key = k
                break

    store.close()

    if df is None:
        raise FileNotFoundError(
            f"No truth_info or evt table in {df_path}. Keys: {keys}"
        )

    print(f"  {label}: loaded {len(df)} rows from {source_key}")

    # ── Extract run/subrun/evt ──
    result = {}

    # Handle MultiIndex columns (common in cafpyana .df)
    def _find_col(name):
        """Find a column, handling MultiIndex and flat columns."""
        if name in df.columns:
            return df[name]
        if isinstance(df.columns, pd.MultiIndex):
            for c in df.columns:
                if isinstance(c, tuple) and any(name == part for part in c):
                    return df[c]
                if str(c).endswith(f".{name}") or str(c).endswith(f"_{name}"):
                    return df[c]
        # Check for common prefixes
        for prefix in ["hdr.", "truth.", "mcnu.", ""]:
            cand = f"{prefix}{name}" if prefix else name
            if cand in df.columns:
                return df[cand]
        return None

    for col_name in ["run", "subrun", "evt"]:
        series = _find_col(col_name)
        if series is None:
            raise ValueError(
                f"Cannot find '{col_name}' in {df_path}. "
                f"Columns: {list(df.columns)[:20]}"
            )
        result[col_name] = series.to_numpy(dtype=np.int64)

    # Neutrino energy
    energy_series = None
    for cand in ["nu_energy", "E", "neutrino_energy", "mc_nu_energy"]:
        energy_series = _find_col(cand)
        if energy_series is not None:
            break

    if energy_series is not None:
        result["nu_energy"] = np.round(energy_series.to_numpy(dtype=float), 6)
    else:
        print(f"  WARNING: No neutrino energy found — matching on (run,subrun,evt) only")
        result["nu_energy"] = np.zeros(len(df), dtype=float)

    # Index keys
    if isinstance(df.index, pd.MultiIndex):
        if "__ntuple" in df.index.names:
            result["__ntuple"] = df.index.get_level_values("__ntuple").to_numpy()
        if "entry" in df.index.names:
            result["entry"] = df.index.get_level_values("entry").to_numpy()
    if "__ntuple" not in result:
        result["__ntuple"] = np.zeros(len(df), dtype=np.int64)
    if "entry" not in result:
        result["entry"] = np.arange(len(df), dtype=np.int64)

    out = pd.DataFrame(result)

    # Deduplicate on physics keys
    n_before = len(out)
    out = out.drop_duplicates(subset=PHYSICS_MATCH_COLS, keep="first").reset_index(drop=True)
    if len(out) < n_before:
        print(f"    Deduped: {n_before} → {len(out)} unique physics events")

    return out


def match_events(
    nom_events: pd.DataFrame,
    var_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Inner-merge nominal and variation events on physics keys.

    Returns (nom_matched, var_matched, stats_dict).
    """
    n_nom = len(nom_events)
    n_var = len(var_events)

    common = nom_events.merge(
        var_events,
        on=PHYSICS_MATCH_COLS,
        how="inner",
        suffixes=("_nom", "_var"),
    )
    n_common = len(common)

    if n_common == 0:
        raise ValueError(
            f"No common events! nom={n_nom}, var={n_var}. "
            f"Are these from the same generated event set?"
        )

    nom_matched = common[["__ntuple_nom", "entry_nom"]].rename(
        columns={"__ntuple_nom": "__ntuple", "entry_nom": "entry"}
    ).drop_duplicates().reset_index(drop=True)

    var_matched = common[["__ntuple_var", "entry_var"]].rename(
        columns={"__ntuple_var": "__ntuple", "entry_var": "entry"}
    ).drop_duplicates().reset_index(drop=True)

    stats = {
        "n_nominal": n_nom,
        "n_variation": n_var,
        "n_common": n_common,
        "ratio_nom": n_common / n_nom if n_nom else 0.0,
        "ratio_var": n_common / n_var if n_var else 0.0,
    }

    print(f"    Matched: {n_common} common / {n_nom} nom / {n_var} var "
          f"(ratio_nom={stats['ratio_nom']:.3f}, ratio_var={stats['ratio_var']:.3f})")

    return nom_matched, var_matched, stats


def main():
    parser = argparse.ArgumentParser(
        description="Match events between nominal and det-var samples."
    )
    parser.add_argument("--var", default=None,
                        help="Single variation (default: all available)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config()
    out_dir = Path(cfg.event_lists_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which variations to process
    if args.var:
        vars_to_process = [args.var]
    else:
        vars_to_process = cfg.available_variations()
        if not vars_to_process:
            print(f"No merged det-var .df files found in {cfg.detvar_df_base}")
            print("Run merge_detvar_dfs.py first.")
            return

    print("═" * 60)
    print("  Matching nominal ↔ det-var events")
    print(f"  Nominal:    {cfg.nominal_df_file}")
    print(f"  Variations: {vars_to_process}")
    print("═" * 60)

    if args.dry_run:
        for var in vars_to_process:
            p = cfg.merged_detvar_df(var)
            exists = Path(p).exists()
            print(f"  {var}: {p} {'✓' if exists else '✗ NOT FOUND'}")
        return

    # Load nominal once
    nom_events = _load_truth_keys(cfg.nominal_df_file, label="nominal")

    all_stats = {}
    for var in vars_to_process:
        print(f"\n── {var} ──")
        detvar_df = cfg.merged_detvar_df(var)

        if not Path(detvar_df).exists():
            print(f"  SKIP: {detvar_df} not found")
            all_stats[var] = {"error": f"file not found: {detvar_df}"}
            continue

        try:
            var_events = _load_truth_keys(detvar_df, label=var)
            nom_matched, var_matched, stats = match_events(nom_events, var_events)

            # Save
            nom_matched.to_csv(out_dir / f"{var}_matched_nom.csv", index=False)
            var_matched.to_csv(out_dir / f"{var}_matched_var.csv", index=False)
            all_stats[var] = stats

        except Exception as e:
            print(f"  ERROR: {e}")
            all_stats[var] = {"error": str(e)}

    # Save summary
    with open(out_dir / "pot_scaling.json", "w") as f:
        json.dump({"variations": all_stats}, f, indent=2)
    print(f"\nSaved: {out_dir / 'pot_scaling.json'}")


if __name__ == "__main__":
    main()