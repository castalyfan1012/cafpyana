# nueCC_mc_weights.py — single config, weights included
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
from make_nueCC_df import make_nuecc_evtdf, make_nuecc_statsdf, make_nuecc_wgtdf
from makedf.makedf import make_hdrdf, make_potdf_bnb

DFS   = [make_nuecc_evtdf, make_nuecc_statsdf, make_hdrdf, make_potdf_bnb, make_nuecc_wgtdf]
NAMES = ["evt", "stats", "hdr", "pot", "mcnu"]