# nueCC_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
from makedf.makedf import make_hdrdf, make_potdf_bnb
from make_nueCC_df import make_nuecc_evtdf_data

DFS   = [make_nuecc_evtdf_data, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "hdr", "pot"]