"""
nuecc_detsys_config.py
----------------------
Configuration for nueCC SPINE-based detector systematics.

Two modes supported:
  1. Flat-CAF direct reader (current — works with Jacob's data-mode wiremod files)
  2. Store-based approach (future — requires MC-mode reprocessed det-var files)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ── Detector variation groups ─────────────────────────────────────────────────
PDS_VARS     = ["pmtgain", "pmtqe", "pmtspe"]
SCE_VARS     = ["nosce", "twicesce"]
WIREMOD_VARS = ["wiremodxtheta", "wiremodyz"]

DET_VARS_ALL = PDS_VARS + SCE_VARS + WIREMOD_VARS

# ── nueCC selection cut stages (from nue_selection.py) ────────────────────────
NUECC_CUTS = [
    "precut",
    "valid_flashmatch",
    "fiducial",
    "single_electron",
    "electron_primary_score",
    "electron_pid_score",
    "vertex_distance",
]

# ── Flat-CAF xrootd paths (for direct reader approach) ────────────────────────
# These are Jacob Zettle's combined det-var CAFs.
# NOTE: processed as data through SPINE → no truth branches.
DET_VAR_CAFS = {
    "wiremodxtheta": (
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/"
        "users/jzettle/sbnd_wiremod_x_thetaxw/"
        "combined_wiremod_x_thetaxw.flat.caf.root"
    ),
    "wiremodyz": (
        "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/persistent/"
        "users/jzettle/sbnd_wiremod_y_z/"
        "combined_wiremod_y_z.flat.caf.root"
    ),
    # Uncomment when files become available:
    # "pmtgain": "root://...",
    # "pmtqe":   "root://...",
    # "nosce":   "root://...",
}


def det_var_subdir(var: str) -> str:
    if var in PDS_VARS:
        return "pds"
    if var in SCE_VARS:
        return "sce"
    if var in WIREMOD_VARS:
        return "wiremod"
    raise ValueError(f"Unknown detector variation: {var}")


@dataclass
class NueCCDetsysConfig:
    """Configuration for nueCC detector systematics."""

    # ── Your nueCC analysis directory ──
    nuecc_dir: str = "/home/castalyf/cafpyana/analysis_village/nueCC"

    # ── Your existing nominal production .df (the merged one) ──
    nominal_df_file: str = (
        "/exp/sbnd/data/users/castalyf/nue_sel/production_files/"
        "gen1/mc1e20_nueCC_v2.df"
    )

    # ── Det-var output base directory ──
    detvar_df_base: str = (
        "/exp/sbnd/data/users/castalyf/nue_sel/detsys/gen1/detvar_dfs"
    )

    # ── Output directories ──
    output_dir: str = "/exp/sbnd/data/users/castalyf/nue_sel/detsys"
    version: str = "gen1"
    day: str = "v1"

    @property
    def save_dir(self) -> str:
        return f"{self.output_dir}/{self.version}/{self.day}"

    @property
    def plot_dir(self) -> str:
        return f"{self.output_dir}/{self.version}/plots/{self.day}"

    @property
    def event_lists_dir(self) -> str:
        return f"{self.output_dir}/{self.version}/event_lists"

    def merged_detvar_df(self, var: str) -> str:
        """Path to the merged .df for a given det variation."""
        return f"{self.detvar_df_base}/merged_{var}.df"

    def available_variations(self) -> list[str]:
        """Return list of variations that have merged .df files."""
        return [v for v in DET_VARS_ALL
                if Path(self.merged_detvar_df(v)).exists()]


def build_config(**kwargs) -> NueCCDetsysConfig:
    return NueCCDetsysConfig(**kwargs)