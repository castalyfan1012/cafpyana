# nueCC_mc_weightsonly.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
from makedf.makedf import make_mcnuwgtdf_slim

DFS   = [make_mcnuwgtdf_slim]
NAMES = ["mcnu"]