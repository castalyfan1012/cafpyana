#!/usr/bin/env python3

import sys
import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cafpyana.pyanalib.pandas_helpers as ph
import cafpyana.makedf.util as util
from cafpyana.pyanalib import sbnanaobj_enums as sbn_enum


KAON_DF = 'kaon_df.df'
BINS = np.arange(0.0, 3.5, 0.5)
DX = BINS[1] - BINS[0]

INDEX_COLS = [("__ntuple", "", "", "", "", ""), ("entry", "", "", "", "", "")] 


def run_subrun_event(hdrdf: pd.DataFrame, df: pd.DataFrame):
    """Print run, subrun, and event number from a dataframe, squashing duplicates (e.g., from a track df)."""
    return hdrdf.loc[df.index.droplevel([
        l for l in df.index.names if l not in ['__ntuple', 'entry']
    ]).unique(), ['run', 'subrun', 'evt']].sort_index()


def main(hdrdf: pd.DataFrame, mcdf: pd.DataFrame, recodf: pd.DataFrame):
    levels = ('__ntuple', 'entry', 'rec.slc..index')
    levels_mc = ('__ntuple', 'entry', 'rec.mc.nu..index', 'rec.mc.nu.prim..index')
    recodf = ph.multicol_add(recodf, recodf.groupby(level=levels).size().rename('npfp'))

    fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=300)

    # slices with truth match
    matchdf = ph.multicol_merge(mcdf.reset_index(), recodf.reset_index(),
                                left_on=INDEX_COLS + [("rec.mc.nu..index", "", "", "", "", ""), ("G4ID", "", "", "", "", "")],
                                right_on=INDEX_COLS + [("slc", "tmatch", "idx", "", "", ""), ("pfp", "trk", "truth", "p", "G4ID", "")],
                                how="left").set_index(list(levels_mc) + ['rec.mc.nu.prim.daughters..index']).sort_index()

    # all rows with pfp matched or not to kaon
    primary_k_reco_match = matchdf[(matchdf.is_primary & (matchdf.pdg == 321) & ~np.isnan(matchdf.pfp.trk.start.x))]

    # record if we got a match
    matchdf['has_reco_k'] = matchdf.index.droplevel([-1, -2]).isin(primary_k_reco_match.index.droplevel([-1, -2]))
    pass_e = matchdf[matchdf.has_reco_k & matchdf.is_signal_kp_cc & matchdf.is_primary & (matchdf.pdg == 321)].genE.to_numpy()
    fail_e = matchdf[~matchdf.has_reco_k & matchdf.is_signal_kp_cc & matchdf.is_primary & (matchdf.pdg == 321)].genE.to_numpy()
    pass_hist = np.histogram(pass_e, bins=BINS)[0].astype(float)
    fail_hist = np.histogram(fail_e, bins=BINS)[0].astype(float)
    total_hist = pass_hist + fail_hist
    errs = np.sqrt(pass_hist) / total_hist

    ax.errorbar(BINS[:-1] + 0.5 * DX, pass_hist / total_hist, yerr=errs, xerr=0.5 * DX)
    ax.set_xlabel('Kaon Energy (GeV)')
    ax.set_ylabel('Efficiency')
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_ylim(0, 1.2)

    plt.savefig('ktest.png')
    print(run_subrun_event(matchdf[matchdf.is_signal_kp_cc]))


if __name__ == '__main__':
    with pd.HDFStore(KAON_DF) as store:
        print(store.keys())

    # mcprimdf = pd.read_hdf(KAON_DF, key='mcprim_0')
    for i in range(1):
        hdrdf = pd.read_hdf(KAON_DF, key=f'hdr_{i}')
        mcdf = pd.read_hdf(KAON_DF, key=f'kmc_{i}')
        recodf = pd.read_hdf(KAON_DF, key=f'kreco_{i}')
        main(hdrdf, mcdf, recodf)
