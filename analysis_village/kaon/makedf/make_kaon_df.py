"""
Kaon analysis data frame maker
"""
import functools

import numpy as np
import pandas as pd

import makedf.makedf as makedf
import makedf.branches as branches
import pyanalib.pandas_helpers as ph
import makedf.util as util


KPDG = {
    'kplus': makedf.PDG["kaon_p"][0],
    'kzero': makedf.PDG["kaon_0"][0]
}
KMASS = {
    'kplus': makedf.PDG["kaon_p"][2],
    'kzero': makedf.PDG["kaon_0"][2]
}

# TODO pick a better number
TRUE_KE_CUT = 0.

# For daughter df merging: This ensures we use the equivalent mc.nu.prim branch
# names (SRTrueParticle) as the daughter branches
PRIM_BRANCHES = list(set(b.replace('.true_particles.', '.mc.nu.prim.') for b in branches.trueparticlebranches))


# InFV requires inzback but it does nothing for SBND case
# put NaN here so we'll hopefully get an error if this ever changes
InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')
InAV_SBND = functools.partial(util.InAV, det='SBND')


def k_has_daughter(df: pd.DataFrame, ktype: str, require_contained: bool=True) -> pd.DataFrame:
    """
    # example event: we want to mark all these rows as "true" since the event
    # contains a kaon + mu daughter (regardless of other primaries)
    entry  rec.mc.nu..index  rec.mc.nu.prim..index  rec.mc.nu.prim.daughters..index
    23     0                 1                      -1                                 3222
                                                     0                                  211
                                                     1                                 2112
                             2                      -1                                  321
                                                     0                                   14
                                                     1                                  -13
    """
    daughter_rows = ~df.is_primary & ((df.pdg == -13) | (df.pdg == 211)) 
    if require_contained:
        daughter_rows &= InAV_SBND(df.end)

    primary_k_rows = (df.is_primary & (df.pdg == KPDG[ktype]))

    # true for entry,mcnu index,prim index rows, false otherwise
    k_has_daughter = df[
        (df.index.droplevel(-1).isin(df[primary_k_rows].index.droplevel(-1))) \
        & (df.index.droplevel(-1).isin(df[daughter_rows].index.droplevel(-1)))
    ]

    # true for entry, mcnu index rows
    return (df.index.droplevel([-1, -2]).isin(k_has_daughter.index.droplevel([-1, -2])))


def signal(df: pd.DataFrame, ktype: str, cc: bool=True) -> pd.DataFrame:
    """
    Signal definition for mcdf.
    Final state kaon from (CC, NC) interaction that decays within the FV
    """
    # There must be a kaon
    has_k = (getattr(df, f'n{ktype}') > 0)

    # kplus: kaon must have a mu or pi daughter, daughter must be contained
    has_daughter = True
    if ktype == 'kplus':
        has_daughter = k_has_daughter(df, ktype, require_contained=True)

    cc_nc = (df.iscc == cc)

    return df.is_true_fv & has_k & cc_nc & has_daughter


def make_kaon_mcdf(f: pd.DataFrame, signal_cut_columns: bool=False) -> pd.DataFrame:
    mcdf = ph.loadbranches(f["recTree"], branches.mcbranches).rec.mc.nu
    # prevent name clash with mcprim pdg column
    mcdf.columns = [
        ('nu_pdg', '') if col == ('pdg', '') else col
        for col in mcdf.columns
    ]

    mcprimdf = ph.loadbranches(f["recTree"], PRIM_BRANCHES).rec.mc.nu.prim
    mcprimdf['is_primary'] = True

    # add number of primaries above threshold
    for kname in ('kplus', 'kzero'):
        # number of kaons above KE threshold
        ke = mcprimdf[mcprimdf.pdg==KPDG[kname]].genE - KMASS[kname] 
        mcdf = ph.multicol_add(mcdf, ((mcprimdf.pdg==KPDG[kname]) \
                                              & (ke > TRUE_KE_CUT)).groupby(level=[0, 1]).sum().rename(f'n{kname}'))

    # daughter info
    tpartdf = ph.loadbranches(f["recTree"], branches.trueparticlebranches).rec.true_particles
    tpartdf = tpartdf.reset_index().set_index(['entry', 'G4ID'])

    mcprimdaughtersdf = makedf.make_mcprimdaughtersdf(f).rec.mc.nu.prim
    daughterdf = mcprimdaughtersdf[mcprimdaughtersdf.index.droplevel(-1).isin(mcprimdf.index)]
    daughter_tpartdf = tpartdf[
        tpartdf.index.isin(pd.MultiIndex.from_frame(daughterdf.reset_index()[['entry', 'daughters']]))
    ]
    daughterdf = ph.multicol_merge(daughterdf, daughter_tpartdf, how="left", left_on=['entry', 'daughters'], right_index=True)
    daughterdf['is_primary'] = False
    daughterdf = daughterdf.drop(columns=[('rec.true_particles..index', '', '', '')])

    # .rename doesn't work for me, so do this instead
    daughterdf.columns = [
        ('G4ID', '', '', '') if col == ('daughters', '', '', '') else col
        for col in daughterdf.columns
    ]

    # add daughter index to primaries as "-1" to allow concat
    mcprimdf["rec.mc.nu.prim.daughters..index"] = -1
    mcprimdf = mcprimdf.set_index("rec.mc.nu.prim.daughters..index", append=True)
    mcprimdf = pd.concat([mcprimdf, daughterdf], sort=True).sort_index()

    # finally, merge primaries into mc
    mcdf = ph.multicol_merge(mcdf, mcprimdf, how="left", left_index=True, right_index=True, validate="one_to_one")

    mcdf['is_true_fv'] = InFV_SBND(mcdf.position)
    mcdf['is_signal_kp_cc'] = signal(mcdf, ktype='kplus', cc=True)
    mcdf['is_signal_kp_nc'] = signal(mcdf, ktype='kplus', cc=False)

    # extra columns for truth studies
    if signal_cut_columns:
        mcdf['has_daughter'] = k_has_daughter(mcdf, 'kplus', require_contained=False)
        mcdf['has_daughter_cont'] = k_has_daughter(mcdf, 'kplus', require_contained=True)

    return mcdf


# use this in configs
make_kaon_mcdf_truthcols = functools.partial(make_kaon_mcdf, signal_cut_columns=True)


def make_kaon_recodf(f: pd.DataFrame) -> pd.DataFrame:
    pandora_df = makedf.make_pandora_df(f)

    # precuts
    pandora_df = pandora_df[InFV_SBND(pandora_df.slc.vertex)]
    pandora_df = pandora_df[pandora_df.slc.is_clear_cosmic == 0]

    # daughter info
    # daughterdf = ph.loadbranches(f["recTree"], branches.pfp_daughter_branch)

    return pandora_df


def make_kaon_mcdf_lite(f: pd.DataFrame) -> pd.DataFrame:
    """Bare-bones check for any k events."""
    df = ph.loadbranches(f["recTree"], ['rec.mc.nu.prim.pdg', 'rec.mc.nu.prim.genE']).rec.mc.nu.prim
    for kname in ('kplus', 'kzero'):
        ke = df[df.pdg==KPDG[kname]].genE - KMASS[kname] 
        df = ph.multicol_add(df, ((df.pdg==KPDG[kname]) \
                                              & (ke > TRUE_KE_CUT)).groupby(level=[0, 1]).sum().rename(f'n{kname}'))

    return (df[(df.nkplus > 0) | (df.nkzero > 0)]
            .drop(['pdg', 'genE'], axis=1)
            .droplevel(-1)
            .groupby(level=[0,1]).first()
    )
