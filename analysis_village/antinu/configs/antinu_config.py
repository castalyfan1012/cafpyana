from makedf.makedf import make_hdrdf
from analysis_village.antinu.makedf import make_antinu_mcdf_truthcols, make_antinu_recodf

DFS = [make_antinu_mcdf_truthcols, make_antinu_recodf, make_hdrdf]
NAMES = ["antimc", "antireco", "hdr"]
