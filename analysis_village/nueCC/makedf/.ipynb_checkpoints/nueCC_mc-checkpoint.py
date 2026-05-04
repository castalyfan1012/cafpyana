"""
nueCC_mc.py
-----------
cafpyana configuration for the SPINE DLP nueCC inclusive MC analysis.

Usage
-----
    python run_df_maker.py -c nueCC_mc.py -l <file_list.txt> -o <output_prefix>

Or with a file limit for testing:
    python run_df_maker.py -c nueCC_mc.py -l <file_list.txt> -o test -nfile 2

Output tables (per split, keyed as <name>_<k>)
-----------------------------------------------
    evt    particle-level df, FM+FV preselected, merged with MC truth
             → loaded by the notebook as the main evtdf
             → consumed by nue_selection.py / nue_helpers.py unchanged
    stats  one row per input file: pre-cut interaction + signal counts
             → efficiency denominators; equivalent to truth_info_0 in skim_nue_v2
    hdr    run/subrun/event header info
    pot    BNB POT per file (swap to make_potdf_numi for NuMI running)

Notes
-----
* This config replaces the two-step workflow (run_df_maker full SPINE config
  → skim_nue_v2.py) with a single pass that produces a preselected,
  MC-truth-merged df directly from the CAF files.

* The 'stats' table accumulates one row per file.  In the notebook, load it
  and sum the columns to get the global efficiency denominators:
      statsdf = pd.read_hdf(merged_file, key="stats_0")
      n_true_signal = statsdf["n_true_signal"].sum()   # ← THE denominator
      n_after_presel = statsdf["n_after_presel"].sum()

* ELECTRON_THRESHOLD_MEV and the truth-signal definition in make_nueCC_df.py
  MUST stay in sync with nue_selection.py and skim_nue_v2.py.
"""
# nueCC_mc_noweights.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
from make_nueCC_df import make_nuecc_evtdf, make_nuecc_statsdf
from makedf.makedf import make_hdrdf, make_potdf_bnb

DFS   = [make_nuecc_evtdf, make_nuecc_statsdf, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "stats", "hdr", "pot"]