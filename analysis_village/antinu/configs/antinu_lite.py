# lite version (signal events only)

from makedf.makedf import make_hdrdf
from analysis_village.antinu.makedf import make_antinu_mcdf_lite

DFS = [make_antinu_mcdf_lite, make_hdrdf]
NAMES = ["antinu_lite", "hdr"]
