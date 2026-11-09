# nueCC_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
from makedf.makedf import make_hdrdf, make_potdf_bnb
from make_nueCC_df import make_nuecc_evtdf_data, make_nuecc_flashdf_data

# flash: one row per reco interaction before FM/FV (flash-match study); after evt
# so it reuses the cached spine df and FM/FV flags
DFS   = [make_nuecc_evtdf_data, make_nuecc_flashdf_data, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "flash", "hdr", "pot"]