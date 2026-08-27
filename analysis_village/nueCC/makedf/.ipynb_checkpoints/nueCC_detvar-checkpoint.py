# nueCC_detvar.py — config for processing det-var CAFs through run_df_maker.py
#
# Uses make_nuecc_evtdf_data (reco-only loader) because Jacob's wiremod CAFs
# were processed through SPINE in data mode and lack dlp_true branches.
#
# Usage:
#   python3 run_df_maker.py \
#       -c analysis_village/nueCC/makedf/nueCC_detvar.py \
#       -l <filelist> -o <output_prefix> -nfile 1

import sys
sys.path.insert(0, "analysis_village/nueCC/makedf")

from make_nueCC_df import make_nuecc_evtdf_data
from makedf.makedf import make_hdrdf, make_potdf_bnb

DFS   = [make_nuecc_evtdf_data, make_hdrdf, make_potdf_bnb]
NAMES = ["evt",                  "hdr",      "pot"]

GRID_PARAMS = {
    "memory":   "16GB",
    "cpu":      1,
    "disk":     "30GB",
    "lifetime": "4h",
}