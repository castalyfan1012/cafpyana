"""
nueCC_mc.py
-----------
cafpyana configuration for the SPINE DLP nueCC inclusive MC analysis.

Run with:
    python run_df_maker.py -c nueCC_mc.py -l <file_list.txt> -o <output_dir>

This config produces four tables per job that are concatenated across files:

  evt     particle-level, pre-skimmed (flashmatch + fiducial applied)
            → the main analysis DataFrame used by nue_selection.py
  stats   one row per file: DLP interaction counts before pre-selection
            → lets the notebook reconstruct the full cut-flow table
  hdr     header info (run, subrun, event)
  pot     beam POT per file (used for normalisation)

The 'evt' table is typically ~5% the size of the unskimmed equivalent,
making it fast to load and process interactively.
"""

import sys, os
# run_df_maker.py exec()s this file from the cafpyana working directory,
# so the directory that contains make_nueCCdf.py is not on sys.path.
# args.config is available in the exec scope – use it to locate the directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))

from make_nueCCdf import make_nuecc_evtdf, make_nuecc_statsdf
from makedf.makedf import make_hdrdf, make_potdf_bnb   # standard cafpyana helpers

DFS = [
    make_nuecc_evtdf,     # → "evt"
    make_nuecc_statsdf,   # → "stats"
    make_hdrdf,           # → "hdr"
    make_potdf_bnb,       # → "pot"  (change to make_potdf_numi for NuMI beam)
]

NAMES = [
    "evt",
    "stats",
    "hdr",
    "pot",
]