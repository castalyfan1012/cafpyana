"""
make_nueCCdf.py
--------------
cafpyana / makedf functions for the SPINE DLP nueCC inclusive analysis.

Produces a SKIMMED particle-level DataFrame restricted to interactions that pass:
  (1) Flash match   rec.dlp.is_flash_matched == 1
  (2) Fiducial cut  rec.dlp.is_fiducial       == 1

Reduction: ~95% fewer rows vs. the full 'evt' table.
Column structure is IDENTICAL to the full table, so nue_helpers.py and
nue_selection.py work without any modification.

Truth information is joined via the best-match DLP interaction stored in
rec.dlp..match_ids (index 0 = highest-overlap truth interaction).

Usage in a cafpyana config  (nueCC_mc.py):
    from make_nueCCdf import make_nuecc_evtdf, make_nuecc_statsdf
    from makedf.makedf import make_hdrdf, make_potdf_bnb
    DFS   = [make_nuecc_evtdf, make_nuecc_statsdf, make_hdrdf, make_potdf_bnb]
    NAMES = ["evt",            "stats",             "hdr",      "pot"         ]

NOTE ON BRANCH NAMES
--------------------
Branch names follow the flat-tree convention used by SPINE CAFs at SBND.
The double-dot (..) marks the vector dimension that becomes a MultiIndex level.
Verify / adjust them against your CAF with:

    import uproot
    with uproot.open("your_file.caf.root") as f:
        print(f["recTree"].keys())

If loadbranches raises a KeyError for any branch, comment it out from the
relevant list below – the pre-selection and nue_selection logic only require
the branches that are listed without a "# optional" comment.
"""

from makedf.makedf import *          # loadbranches, make_hdrdf, make_potdf_bnb, …
from pyanalib.pandas_helpers import *  # multicol_merge, multicol_add, …

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Branch lists  –  edit only if your CAF uses a different naming scheme
# ─────────────────────────────────────────────────────────────────────────────

# ── Reco DLP: interaction-level ───────────────────────────────────────────────
_DLP_INTER_BRANCHES = [
    # Vertex
    "rec.dlp..vertex.x",
    "rec.dlp..vertex.y",
    "rec.dlp..vertex.z",
    # Selection flags  (required for pre-selection)
    "rec.dlp..is_flash_matched",
    "rec.dlp..is_fiducial",
    "rec.dlp..is_contained",
    "rec.dlp..is_time_contained",
    "rec.dlp..is_matched",
    "rec.dlp..is_cathode_crosser",
    "rec.dlp..is_truth",
    # Flash
    "rec.dlp..flash_hypo_pe",
    "rec.dlp..flash_total_pe",
    # Multiplicity / size
    "rec.dlp..id",
    "rec.dlp..size",
    "rec.dlp..num_particles",
    "rec.dlp..num_primary_particles",
    "rec.dlp..depositions_sum",
    "rec.dlp..cathode_offset",
    # Truth-reco matching  (required for truth join)
    "rec.dlp..match_ids",
    "rec.dlp..match_overlaps",
    # Particle counts per PID  (0=photon 1=electron 2=muon 3=pion 4=proton 5=other)
    "rec.dlp..particle_counts.0",
    "rec.dlp..particle_counts.1",
    "rec.dlp..particle_counts.2",
    "rec.dlp..particle_counts.3",
    "rec.dlp..particle_counts.4",
    "rec.dlp..particle_counts.5",
    "rec.dlp..primary_particle_counts.0",
    "rec.dlp..primary_particle_counts.1",
    "rec.dlp..primary_particle_counts.2",
    "rec.dlp..primary_particle_counts.3",
    "rec.dlp..primary_particle_counts.4",
    "rec.dlp..primary_particle_counts.5",
]

# ── Reco DLP: particle-level ─────────────────────────────────────────────────
_DLP_PART_BRANCHES = [
    # PID + scores  (required for cut flow)
    "rec.dlp.particles..pid",
    "rec.dlp.particles..pid_scores.0",
    "rec.dlp.particles..pid_scores.1",    # electron PID softmax
    "rec.dlp.particles..pid_scores.2",
    "rec.dlp.particles..pid_scores.3",
    "rec.dlp.particles..pid_scores.4",
    "rec.dlp.particles..pid_scores.5",
    "rec.dlp.particles..chi2_pid",
    "rec.dlp.particles..chi2_per_pid.0",
    "rec.dlp.particles..chi2_per_pid.1",
    "rec.dlp.particles..chi2_per_pid.2",
    "rec.dlp.particles..chi2_per_pid.3",
    "rec.dlp.particles..chi2_per_pid.4",
    "rec.dlp.particles..chi2_per_pid.5",
    "rec.dlp.particles..primary_scores.0",
    "rec.dlp.particles..primary_scores.1",  # electron primary softmax
    # Kinematics  (required for leading-electron observables)
    "rec.dlp.particles..ke",
    "rec.dlp.particles..p",
    "rec.dlp.particles..mass",
    "rec.dlp.particles..calo_ke",
    "rec.dlp.particles..csda_ke",
    "rec.dlp.particles..csda_ke_per_pid.0",
    "rec.dlp.particles..csda_ke_per_pid.1",
    "rec.dlp.particles..csda_ke_per_pid.2",
    "rec.dlp.particles..csda_ke_per_pid.3",
    "rec.dlp.particles..csda_ke_per_pid.4",
    "rec.dlp.particles..csda_ke_per_pid.5",
    "rec.dlp.particles..mcs_ke",
    "rec.dlp.particles..mcs_ke_per_pid.0",
    "rec.dlp.particles..mcs_ke_per_pid.1",
    "rec.dlp.particles..mcs_ke_per_pid.2",
    "rec.dlp.particles..mcs_ke_per_pid.3",
    "rec.dlp.particles..mcs_ke_per_pid.4",
    "rec.dlp.particles..mcs_ke_per_pid.5",
    # Momentum vector  (required for cos-theta)
    "rec.dlp.particles..momentum.x",
    "rec.dlp.particles..momentum.y",
    "rec.dlp.particles..momentum.z",
    # Geometry
    "rec.dlp.particles..start_point.x",
    "rec.dlp.particles..start_point.y",
    "rec.dlp.particles..start_point.z",
    "rec.dlp.particles..end_point.x",
    "rec.dlp.particles..end_point.y",
    "rec.dlp.particles..end_point.z",
    "rec.dlp.particles..start_dir.x",
    "rec.dlp.particles..start_dir.y",
    "rec.dlp.particles..start_dir.z",
    "rec.dlp.particles..end_dir.x",
    "rec.dlp.particles..end_dir.y",
    "rec.dlp.particles..end_dir.z",
    "rec.dlp.particles..length",
    # Shower quality  (required for shower-quality cut flow)
    "rec.dlp.particles..start_dedx",
    "rec.dlp.particles..vertex_distance",
    "rec.dlp.particles..directional_spread",
    "rec.dlp.particles..axial_spread",
    "rec.dlp.particles..start_straightness",
    "rec.dlp.particles..axial_spread",
    # Topology flags
    "rec.dlp.particles..is_primary",
    "rec.dlp.particles..is_valid",
    "rec.dlp.particles..is_contained",
    "rec.dlp.particles..is_matched",
    "rec.dlp.particles..is_cathode_crosser",
    "rec.dlp.particles..is_time_contained",
    "rec.dlp.particles..is_truth",
    "rec.dlp.particles..shape",
    "rec.dlp.particles..size",
    "rec.dlp.particles..num_fragments",
    "rec.dlp.particles..depositions_sum",
    "rec.dlp.particles..cathode_offset",
    "rec.dlp.particles..interaction_id",
    # Reco↔truth matching (particle level)
    "rec.dlp.particles..match_ids",
    "rec.dlp.particles..match_overlaps",
]

# ── Truth DLP: interaction-level ──────────────────────────────────────────────
_DLP_TRUE_INTER_BRANCHES = [
    # Neutrino identity  (required for classify_truth)
    "rec.dlp_true..nu_id",
    "rec.dlp_true..pdg_code",
    "rec.dlp_true..current_type",        # 0=CC 1=NC
    "rec.dlp_true..interaction_mode",
    "rec.dlp_true..interaction_type",
    "rec.dlp_true..lepton_pdg_code",
    "rec.dlp_true..lepton_p",
    "rec.dlp_true..lepton_track_id",
    # Fiducial / containment  (required for classify_truth + true_signal_index)
    "rec.dlp_true..is_fiducial",
    "rec.dlp_true..is_contained",
    "rec.dlp_true..is_matched",
    "rec.dlp_true..is_flash_matched",
    "rec.dlp_true..is_time_contained",
    "rec.dlp_true..is_cathode_crosser",
    "rec.dlp_true..is_truth",
    # Position / kinematics
    "rec.dlp_true..vertex.x",
    "rec.dlp_true..vertex.y",
    "rec.dlp_true..vertex.z",
    "rec.dlp_true..reco_vertex.x",
    "rec.dlp_true..reco_vertex.y",
    "rec.dlp_true..reco_vertex.z",
    "rec.dlp_true..energy_init",
    "rec.dlp_true..energy_transfer",
    "rec.dlp_true..momentum.x",
    "rec.dlp_true..momentum.y",
    "rec.dlp_true..momentum.z",
    "rec.dlp_true..momentum_transfer",
    "rec.dlp_true..momentum_transfer_mag",
    "rec.dlp_true..inelasticity",
    "rec.dlp_true..bjorken_x",
    "rec.dlp_true..hadronic_invariant_mass",
    "rec.dlp_true..theta",
    "rec.dlp_true..distance_travel",
    # Nuclear target
    "rec.dlp_true..target",
    "rec.dlp_true..nucleon",
    "rec.dlp_true..quark",
    # Size / id
    "rec.dlp_true..id",
    "rec.dlp_true..orig_id",
    "rec.dlp_true..mct_index",
    "rec.dlp_true..track_id",
    "rec.dlp_true..num_particles",
    "rec.dlp_true..num_primary_particles",
    "rec.dlp_true..size",
    "rec.dlp_true..size_adapt",
    "rec.dlp_true..size_g4",
    "rec.dlp_true..depositions_sum",
    "rec.dlp_true..depositions_q_sum",
    "rec.dlp_true..depositions_adapt_sum",
    "rec.dlp_true..depositions_adapt_q_sum",
    "rec.dlp_true..depositions_g4_sum",
    "rec.dlp_true..cathode_offset",
    "rec.dlp_true..flash_hypo_pe",
    "rec.dlp_true..flash_total_pe",
    # Particle counts per PID
    "rec.dlp_true..particle_counts.0",
    "rec.dlp_true..particle_counts.1",
    "rec.dlp_true..particle_counts.2",
    "rec.dlp_true..particle_counts.3",
    "rec.dlp_true..particle_counts.4",
    "rec.dlp_true..particle_counts.5",
    "rec.dlp_true..primary_particle_counts.0",
    "rec.dlp_true..primary_particle_counts.1",
    "rec.dlp_true..primary_particle_counts.2",
    "rec.dlp_true..primary_particle_counts.3",
    "rec.dlp_true..primary_particle_counts.4",
    "rec.dlp_true..primary_particle_counts.5",
]

# ── Truth DLP: particle-level ─────────────────────────────────────────────────
_DLP_TRUE_PART_BRANCHES = [
    # Particle identity  (required for classify_truth: pid, parent_pdg_code)
    "rec.dlp_true.particles..pid",
    "rec.dlp_true.particles..pdg_code",
    "rec.dlp_true.particles..parent_pdg_code",
    "rec.dlp_true.particles..ancestor_pdg_code",
    "rec.dlp_true.particles..parent_track_id",
    # Kinematics  (required for true_leading_electron_ke)
    "rec.dlp_true.particles..ke",
    "rec.dlp_true.particles..energy_init",
    "rec.dlp_true.particles..energy_deposit",
    "rec.dlp_true.particles..p",
    "rec.dlp_true.particles..mass",
    "rec.dlp_true.particles..momentum.x",
    "rec.dlp_true.particles..momentum.y",
    "rec.dlp_true.particles..momentum.z",
    "rec.dlp_true.particles..csda_ke",
    "rec.dlp_true.particles..csda_ke_per_pid.0",
    "rec.dlp_true.particles..csda_ke_per_pid.1",
    "rec.dlp_true.particles..csda_ke_per_pid.2",
    "rec.dlp_true.particles..csda_ke_per_pid.3",
    "rec.dlp_true.particles..csda_ke_per_pid.4",
    "rec.dlp_true.particles..csda_ke_per_pid.5",
    "rec.dlp_true.particles..mcs_ke",
    "rec.dlp_true.particles..mcs_ke_per_pid.0",
    "rec.dlp_true.particles..mcs_ke_per_pid.1",
    "rec.dlp_true.particles..mcs_ke_per_pid.2",
    "rec.dlp_true.particles..mcs_ke_per_pid.3",
    "rec.dlp_true.particles..mcs_ke_per_pid.4",
    "rec.dlp_true.particles..mcs_ke_per_pid.5",
    "rec.dlp_true.particles..calo_ke",
    "rec.dlp_true.particles..reco_ke",
    "rec.dlp_true.particles..reco_length",
    # Geometry
    "rec.dlp_true.particles..start_point.x",
    "rec.dlp_true.particles..start_point.y",
    "rec.dlp_true.particles..start_point.z",
    "rec.dlp_true.particles..end_point.x",
    "rec.dlp_true.particles..end_point.y",
    "rec.dlp_true.particles..end_point.z",
    "rec.dlp_true.particles..start_dir.x",
    "rec.dlp_true.particles..start_dir.y",
    "rec.dlp_true.particles..start_dir.z",
    "rec.dlp_true.particles..end_dir.x",
    "rec.dlp_true.particles..end_dir.y",
    "rec.dlp_true.particles..end_dir.z",
    "rec.dlp_true.particles..length",
    "rec.dlp_true.particles..distance_travel",
    # Topology flags  (required for is_primary cuts)
    "rec.dlp_true.particles..is_primary",
    "rec.dlp_true.particles..interaction_primary",
    "rec.dlp_true.particles..group_primary",
    "rec.dlp_true.particles..is_valid",
    "rec.dlp_true.particles..is_contained",
    "rec.dlp_true.particles..is_matched",
    "rec.dlp_true.particles..is_cathode_crosser",
    "rec.dlp_true.particles..is_time_contained",
    "rec.dlp_true.particles..is_truth",
    # Linking
    "rec.dlp_true.particles..nu_id",
    "rec.dlp_true.particles..interaction_id",
    "rec.dlp_true.particles..group_id",
    "rec.dlp_true.particles..orig_id",
    "rec.dlp_true.particles..shape",
    "rec.dlp_true.particles..size",
    "rec.dlp_true.particles..size_adapt",
    "rec.dlp_true.particles..size_g4",
    "rec.dlp_true.particles..num_fragments",
    "rec.dlp_true.particles..num_voxels",
    "rec.dlp_true.particles..depositions_sum",
    "rec.dlp_true.particles..depositions_q_sum",
    "rec.dlp_true.particles..depositions_g4_sum",
    "rec.dlp_true.particles..cathode_offset",
    "rec.dlp_true.particles..mct_index",
    "rec.dlp_true.particles..mcst_index",
    "rec.dlp_true.particles..track_id",
    "rec.dlp_true.particles..t",
    "rec.dlp_true.particles..end_t",
    # Reco-matched info on truth particles
    "rec.dlp_true.particles..reco_ke",
    "rec.dlp_true.particles..reco_length",
    "rec.dlp_true.particles..reco_start_dir.0",
    "rec.dlp_true.particles..reco_start_dir.1",
    "rec.dlp_true.particles..reco_start_dir.2",
    "rec.dlp_true.particles..reco_end_dir.0",
    "rec.dlp_true.particles..reco_end_dir.1",
    "rec.dlp_true.particles..reco_end_dir.2",
    "rec.dlp_true.particles..reco_momentum.0",
    "rec.dlp_true.particles..reco_momentum.1",
    "rec.dlp_true.particles..reco_momentum.2",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_load(tree, branches):
    """
    Load branches one by one, silently skipping any that are not in the tree.
    Returns the DataFrame of successfully loaded branches.
    This guard lets the same branch list work across CAF versions without
    crashing on newly-added or renamed branches.
    """
    available = []
    for br in branches:
        try:
            _ = loadbranches(tree, [br])
            available.append(br)
        except Exception:
            pass
    if not available:
        raise RuntimeError("No branches could be loaded from provided list.")
    return loadbranches(tree, available)


def _interaction_index_names(df):
    """Return the index level names that identify a unique interaction (all but last)."""
    return list(df.index.names[:-1])


def _find_col(df, *name_parts):
    """
    Find a MultiIndex column whose levels contain all name_parts.
    Returns the first match or raises KeyError.
    """
    for col in df.columns:
        col_str = str(col)
        if all(str(p) in col_str for p in name_parts):
            return col
    raise KeyError(f"Could not find column matching {name_parts} in {list(df.columns[:20])}")


# ─────────────────────────────────────────────────────────────────────────────
#  Public makedf functions
# ─────────────────────────────────────────────────────────────────────────────

def make_nuecc_evtdf(f):
    """
    Load SPINE/DLP branches from a CAF ROOT file and return a pre-skimmed
    particle-level DataFrame (flashmatch + fiducial interactions only).

    Output MultiIndex
    -----------------
    Within a single file (before cafpyana adds '__ntuple'):
        ['entry', 'rec.dlp..index', 'rec.dlp.particles..index']

    Column structure
    ----------------
    Identical to the full 'evt' table (MultiIndex: rec / dlp|dlp_true / …),
    so nue_helpers.py / nue_selection.py work without any modification.

    Truth join
    ----------
    For each reco DLP interaction, the best-matching truth interaction
    (rec.dlp..match_ids, index 0) is looked up and its properties are
    attached under the 'rec.dlp_true' column namespace.
    Truth particles from the matched truth interaction are joined similarly
    via rec.dlp.particles..match_ids.
    """
    tree = f["recTree"]

    # ── 1. Reco DLP particles ──────────────────────────────────────────────
    print("  loading reco DLP particles …")
    reco_part = _safe_load(tree, _DLP_PART_BRANCHES)
    # index: (entry, rec.dlp..index, rec.dlp.particles..index)

    # ── 2. Reco DLP interaction-level ─────────────────────────────────────
    print("  loading reco DLP interactions …")
    reco_inter = _safe_load(tree, _DLP_INTER_BRANCHES)
    # index: (entry, rec.dlp..index)

    # ── 3. Truth DLP interactions ──────────────────────────────────────────
    print("  loading truth DLP interactions …")
    true_inter = _safe_load(tree, _DLP_TRUE_INTER_BRANCHES)
    # index: (entry, rec.dlp_true..index)

    # ── 4. Truth DLP particles ─────────────────────────────────────────────
    print("  loading truth DLP particles …")
    true_part = _safe_load(tree, _DLP_TRUE_PART_BRANCHES)
    # index: (entry, rec.dlp_true..index, rec.dlp_true.particles..index)

    # ── 5. Build the merged DataFrame ─────────────────────────────────────
    #
    # Strategy:
    #   a) Repeat interaction-level reco columns onto particle-level rows
    #   b) Identify the best-matched truth interaction per reco interaction
    #   c) Attach truth interaction columns (repeated for every reco particle)
    #   d) Attach truth particle columns aligned by particle match_ids
    #
    # This reproduces the structure of the existing full 'evt' table.

    # ── 5a. Broadcast reco interaction cols → particle level ───────────────
    il_reco = _interaction_index_names(reco_part)  # [entry, rec.dlp..index]
    evtdf = multicol_merge(
        reco_part.reset_index(),
        reco_inter.reset_index(),
        left_on  = [tuple([n] + [''] * 5) for n in il_reco],
        right_on = [tuple([n] + [''] * 5) for n in il_reco],
        how      = "left",
    )
    evtdf = evtdf.set_index(reco_part.index.names, verify_integrity=False)

    # ── 5b. Best-matched truth interaction per reco interaction ────────────
    # rec.dlp..match_ids column: the first entry is the best match
    # After loadbranches this typically appears as ('rec','dlp','match_ids','','','')
    # or similar – _find_col handles either naming.
    try:
        match_id_col = _find_col(reco_inter, "dlp", "match_ids")
        # match_id_col values are the rec.dlp_true..index of the best match
        best_match_ser = reco_inter[match_id_col]   # indexed by (entry, rec.dlp..index)
    except KeyError:
        # No match_ids available: attach empty truth placeholders
        print("  WARNING: rec.dlp..match_ids not found; truth info will be NaN.")
        best_match_ser = pd.Series(
            np.full(len(reco_inter), np.nan),
            index=reco_inter.index,
            name="_best_match",
        )

    # ── 5c. Join truth interaction info onto reco interactions ─────────────
    # Build a DataFrame: (entry, rec.dlp..index) → best-match truth inter index
    best_match_df = best_match_ser.rename("_best_truth_inter_idx").to_frame()
    best_match_df = best_match_df.reset_index()   # entry, rec.dlp..index, _best_truth_inter_idx

    # Reset true_inter to merge on truth inter index
    true_inter_reset = true_inter.reset_index()
    # The truth interaction index column name (e.g. 'rec.dlp_true..index')
    true_inter_idx_col = true_inter.index.names[-1]   # last level

    # Merge reco interactions → best-matched truth interaction
    reco_with_truth_inter = best_match_df.merge(
        true_inter_reset,
        left_on  = ["entry", "_best_truth_inter_idx"],
        right_on = ["entry", true_inter_idx_col],
        how      = "left",
        suffixes = ("_x", "_y"),
    )
    # Drop the duplicate index column
    reco_with_truth_inter = reco_with_truth_inter.drop(
        columns=["_best_truth_inter_idx"], errors="ignore"
    )

    # ── 5d. Broadcast truth interaction cols → particle level ──────────────
    reco_inter_idx_col = reco_inter.index.names[-1]   # rec.dlp..index
    evtdf_reset = evtdf.reset_index()

    evtdf_reset = evtdf_reset.merge(
        reco_with_truth_inter,
        on    = ["entry", reco_inter_idx_col],
        how   = "left",
        suffixes = ("", "_truth_inter"),
    )

    # ── 5e. Join truth particles via particle-level match_ids ──────────────
    try:
        part_match_col = _find_col(reco_part, "dlp", "particles", "match_ids")
        # part_match_col values: truth particle index matched to each reco particle
        part_match_vals = reco_part[part_match_col]
        evtdf_reset["_best_truth_part_idx"] = part_match_vals.values

        true_part_reset = true_part.reset_index()
        true_part_idx_col = true_part.index.names[-1]   # rec.dlp_true.particles..index

        evtdf_reset = evtdf_reset.merge(
            true_part_reset,
            left_on  = ["entry", "_best_truth_part_idx"],
            right_on = ["entry", true_part_idx_col],
            how      = "left",
            suffixes = ("", "_truth_part"),
        )
        evtdf_reset = evtdf_reset.drop(
            columns=["_best_truth_part_idx"], errors="ignore"
        )
    except KeyError:
        print("  WARNING: particle-level match_ids not found; truth particles will be NaN.")

    # ── 6. Restore MultiIndex ──────────────────────────────────────────────
    evtdf = evtdf_reset.set_index(reco_part.index.names, verify_integrity=False)

    # ── 7. Pre-selection: flashmatch + fiducial ────────────────────────────
    print("  applying pre-selection (flashmatch + fiducial) …")
    try:
        fm_col = _find_col(evtdf, "dlp", "is_flash_matched")
        fv_col = _find_col(evtdf, "dlp", "is_fiducial")
        # Avoid accidentally matching dlp_true columns
        # (dlp_true also has is_fiducial – we want the reco one)
        fm_col = next(
            c for c in evtdf.columns
            if "dlp" in str(c) and "dlp_true" not in str(c)
               and "is_flash_matched" in str(c)
        )
        fv_col = next(
            c for c in evtdf.columns
            if "dlp" in str(c) and "dlp_true" not in str(c)
               and "is_fiducial" in str(c)
        )
        keep = (evtdf[fm_col] == 1) & (evtdf[fv_col] == 1)
        n_before = len(evtdf)
        evtdf = evtdf[keep]
        n_after = len(evtdf)
        print(f"  pre-selection: {n_before:,} → {n_after:,} particle rows")
    except StopIteration:
        print("  WARNING: could not locate is_flash_matched / is_fiducial; no pre-selection applied.")

    return evtdf


def make_nuecc_statsdf(f):
    """
    Return a small one-row DataFrame recording the number of DLP interactions
    at each pre-selection stage (before the df was skimmed).

    Columns
    -------
    n_total   : total reconstructed DLP interactions in the file
    n_fm      : interactions passing flash match  (is_flash_matched == 1)
    n_fm_fv   : interactions passing flash match + fiducial

    These numbers are used in the notebook to display the full cut-flow table
    (rows 'No cut' and 'Flash match') even though those events were removed
    from the skimmed evt table.
    """
    tree = f["recTree"]
    inter = _safe_load(tree, [
        "rec.dlp..is_flash_matched",
        "rec.dlp..is_fiducial",
    ])

    fm_col = _find_col(inter, "dlp", "is_flash_matched")
    fv_col = _find_col(inter, "dlp", "is_fiducial")

    n_total = len(inter)
    n_fm    = int((inter[fm_col] == 1).sum())
    n_fm_fv = int(((inter[fm_col] == 1) & (inter[fv_col] == 1)).sum())

    return pd.DataFrame({
        "n_total": [n_total],
        "n_fm":    [n_fm],
        "n_fm_fv": [n_fm_fv],
    })