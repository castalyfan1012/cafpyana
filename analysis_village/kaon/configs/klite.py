# lite version (signal events only)

from makedf.makedf import make_hdrdf
from analysis_village.kaon.makedf import make_kaon_mcdf_lite

DFS = [make_kaon_mcdf_lite, make_hdrdf]
NAMES = ["kmc_lite", "hdr"]
