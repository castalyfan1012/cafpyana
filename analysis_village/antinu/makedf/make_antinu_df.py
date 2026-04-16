"""
Anti-neutrino analysis data frame maker
"""
import functools

import numpy as np
import pandas as pd

import makedf.makedf as makedf
import makedf.branches as branches
import pyanalib.pandas_helpers as ph
import makedf.util as util

import analysis_village.antinu.const as aconst


MUPLUS_PDG = -makedf.PDG['muon'][0]
MU_MASS = makedf.PDG['muon'][2]

# TODO pick a better number
TRUE_KE_CUT = 0.

# For daughter df merging: This ensures we use the equivalent mc.nu.prim branch
# names (SRTrueParticle) as the daughter branches
PRIM_BRANCHES = list(set(b.replace('.true_particles.', '.mc.nu.prim.') for b in branches.trueparticlebranches))


# InFV requires inzback but it does nothing for SBND case
# put NaN here so we'll hopefully get an error if this ever changes
InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')
InAV_SBND = functools.partial(util.InAV, det='SBND')


def has_daughter(df: pd.DataFrame, require_contained: bool=True) -> pd.DataFrame:
    """Flag rows as True where the interaction has a decay electron"""
    daughter_rows = ~df.is_primary & (df.pdg == -11)
    if require_contained:
        daughter_rows &= InAV_SBND(df.end)

    primary_rows = df.is_primary & (df.pdg == MUPLUS_PDG)

    # true for entry,mcnu index,prim index rows, false otherwise
    has_daughter = df[
        (df.index.droplevel(-1).isin(df[primary_rows].index.droplevel(-1))) \
        & (df.index.droplevel(-1).isin(df[daughter_rows].index.droplevel(-1)))
    ]

    # true for entry, mcnu index rows
    return (df.index.droplevel([-1, -2]).isin(has_daughter.index.droplevel([-1, -2])))


def n_primaries(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of primaries (assuming entry, nu, primary, daughter index levels."""
    primary_counts = (df.index
            .droplevel(-1)
            .to_frame(index=False)
            .groupby(['entry', 'rec.mc.nu..index'])['rec.mc.nu.prim..index'].nunique()
    )
    return (df.reset_index().set_index(['entry', 'rec.mc.nu..index']).index
            .map(primary_counts)
    )


def true_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal definition for mcdf.
    Final state mu plus from CC anti-nu interaction that decays within the FV
    """
    has_muplus = df.nmuplus > 0
    has_michel = has_daughter(df, require_contained=True)

    result = pd.DataFrame(pd.NA, index=df.index, columns=['col_name'])

    # signal + other anti nus
    result[df.is_true_fv & has_muplus & df.iscc & has_michel] = aconst.TrueType.SIGNAL.value
    result[df.is_true_fv & has_muplus & df.iscc & ~has_michel] = aconst.TrueType.ANTINU_CC_NODECAY.value

    # backgrounds
    result[df.is_true_fv & df.iscc & ~has_muplus] = aconst.TrueType.NU_CC.value
    result[df.is_true_fv & ~df.iscc] = aconst.TrueType.NC.value
    result[~df.is_true_fv & (df.nu_pdg > 0)] = aconst.TrueType.NU_OUT_OF_FV.value
    result[(df.nu_pdg == -1)] = aconst.TrueType.COSMIC.value

    return result


def make_antinu_mcdf(f: pd.DataFrame, signal_cut_columns: bool=False) -> pd.DataFrame:
    mcdf = ph.loadbranches(f["recTree"], branches.mcbranches).rec.mc.nu
    # prevent name clash with mcprim pdg column
    mcdf.columns = [
        ('nu_pdg', '') if col == ('pdg', '') else col
        for col in mcdf.columns
    ]

    mcprimdf = ph.loadbranches(f["recTree"], PRIM_BRANCHES).rec.mc.nu.prim
    mcprimdf['is_primary'] = True

    # number above KE threshold
    ke = mcprimdf[mcprimdf.pdg==MUPLUS_PDG].genE - MU_MASS 
    mcdf = ph.multicol_add(mcdf, ((mcprimdf.pdg==MUPLUS_PDG) \
                                          & (ke > TRUE_KE_CUT)).groupby(level=[0, 1]).sum().rename(f'nmuplus'))

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

    mcdf['n_primaries'] = n_primaries(mcdf)
    mcdf['is_true_fv'] = InFV_SBND(mcdf.position)
    mcdf['true_type'] = true_type(mcdf)

    # extra columns for truth studies
    if signal_cut_columns:
        mcdf['has_daughter'] = has_daughter(mcdf, require_contained=False)
        mcdf['has_daughter_cont'] = has_daughter(mcdf, require_contained=True)

    return mcdf


# use this in configs
make_antinu_mcdf_truthcols = functools.partial(make_antinu_mcdf, signal_cut_columns=True)


def make_antinu_recodf(f: pd.DataFrame) -> pd.DataFrame:
    pandora_df = makedf.make_pandora_df(f)

    # precuts
    pandora_df = pandora_df[InFV_SBND(pandora_df.slc.vertex)]
    pandora_df = pandora_df[pandora_df.slc.is_clear_cosmic == 0]

    # daughter info
    # daughterdf = ph.loadbranches(f["recTree"], branches.pfp_daughter_branch)

    return pandora_df


def make_antinu_mcdf_lite(f: pd.DataFrame) -> pd.DataFrame:
    """Bare-bones check for any anti-nu events."""
    df = ph.loadbranches(f["recTree"], ['rec.mc.nu.prim.pdg', 'rec.mc.nu.prim.genE', 'rec.mc.nu.prim.start.x', 'rec.mc.nu.prim.start.y', 'rec.mc.nu.prim.start.z']).rec.mc.nu.prim
    ke = df[df.pdg==MUPLUS_PDG].genE - MU_MASS
    df = ph.multicol_add(df, ((df.pdg==MUPLUS_PDG) \
                                          & (ke > TRUE_KE_CUT) & InFV_SBND(df.start)).groupby(level=[0, 1]).sum().rename('nmuplus_fv'))

    return (df[df.nmuplus_fv > 0]
            .drop(['pdg', 'genE'], axis=1)
            .droplevel(-1)
            .groupby(level=[0,1]).first()
    )
