# # nueCC_mc_weights.py — single config, weights included
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))

from make_nueCC_df import make_nuecc_wgtdf, make_nuecc_wgtdf_full
from makedf.makedf import make_hdrdf, make_potdf_bnb

DFS   = [make_nuecc_wgtdf, make_nuecc_wgtdf_full, make_hdrdf, make_potdf_bnb]
NAMES = ["mcnu",            "mcnu_full",           "hdr",      "pot"]

GRID_PARAMS = {"memory": "29GB", "cpu": 1, "disk": "50GB", "lifetime": "6h"}