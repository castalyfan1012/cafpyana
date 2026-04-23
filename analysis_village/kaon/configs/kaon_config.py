from makedf.makedf import make_hdrdf
from analysis_village.kaon.makedf import make_kaon_mcdf_truthcols, make_kaon_recodf

DFS = [make_kaon_mcdf_truthcols, make_kaon_recodf, make_hdrdf]
NAMES = ["kmc", "kreco", "hdr"]
