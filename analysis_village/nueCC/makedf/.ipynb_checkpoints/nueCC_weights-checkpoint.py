# # nueCC_mc_weights.py — single config, weights included
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
 
from make_nueCC_df import make_nuecc_wgtdf
from makedf.makedf import make_hdrdf, make_potdf_bnb
 
DFS   = [make_nuecc_wgtdf, make_hdrdf, make_potdf_bnb]
NAMES = ["mcnu",            "hdr",      "pot"]
 
# Honored by the patched run_df_maker.py. Pre-patch run_df_maker.py will
# silently ignore this dict (no harm done).
GRID_PARAMS = {
    "memory":   "25GB",   # was 29GB; light wgtdf should never need it
    "cpu":      1,
    "disk":     "50GB",
    "lifetime": "6h",
}