"""
nue_helpers.py
--------------
Core utilities for the nueCC inclusive SPINE analysis.
Plotting functions have been moved to nue_plotter.py.
"""

import numpy as np
import pandas as pd

import os, gc, warnings
from collections import defaultdict

from tqdm.auto import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Generic HDF helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_hdf_key(hdf_file, key_prefix):
    """Load a (possibly split) HDF key, concatenating sub-keys."""
    with pd.HDFStore(hdf_file, mode="r") as store:
        keys = store.keys()
        if '/split' in keys:
            try:
                split_df = store['split']
                n = int(split_df['n_split'].iloc[0])
                return pd.concat([store[f'{key_prefix}_{k}'] for k in range(n)])
            except (AttributeError, KeyError, TypeError):
                pass
        matching = sorted(
            k for k in keys
            if k.lstrip('/').startswith(key_prefix.lstrip('/'))
        )
        if not matching:
            raise KeyError(f"No keys matching {key_prefix!r} in {hdf_file}")
        return pd.concat([store[k] for k in matching])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  sel_topo loader
# ══════════════════════════════════════════════════════════════════════════════

def load_sel_topo(sel_file, df_file=None, verbose=True):
    """
    Load the selection DataFrame and derived quantities.

    Returns
    -------
    sel_topo : pd.DataFrame
    pot_scale : float
    n_true_signal : int
    histpotdf : pd.DataFrame  (only if df_file given)
    statsdf   : pd.DataFrame  (only if df_file given)
    """
    sel_topo = pd.read_hdf(sel_file)
    if sel_topo.index.duplicated().any():
        n_dup = sel_topo.index.duplicated().sum()
        if verbose:
            print(f'WARNING: {n_dup} duplicate rows — deduplicating')
        sel_topo = sel_topo[~sel_topo.index.duplicated(keep='first')]

    pot_scale = float(sel_topo['pot_scale'].iloc[0])
    n_true_signal = int(sel_topo['is_sig'].sum())

    if verbose:
        print(f'sel_topo shape : {sel_topo.shape}')
        print(f'pot_scale      : {pot_scale:.4f}')
        print(f'n_true_signal  : {n_true_signal:,}')
        for col in [c for c in sel_topo.columns if c.startswith('sel_')]:
            n_sig = int((sel_topo[col] & sel_topo['is_sig']).sum())
            n_tot = int(sel_topo[col].sum())
            print(f'  {col:<35} sig={n_sig:>6,}  total={n_tot:>9,}')

    out = dict(sel_topo=sel_topo, pot_scale=pot_scale,
               n_true_signal=n_true_signal)

    if df_file is not None:
        out['histpotdf'] = load_hdf_key(df_file, 'histpotdf')
        out['statsdf'] = load_hdf_key(df_file, 'truth_info')

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Weight classification
# ══════════════════════════════════════════════════════════════════════════════

_FLUX_KWS = ['flux', 'expskin', 'horncurrent', 'kplus', 'kminus', 'kzero',
             'piplus', 'piminus', 'nucleon', 'pion', 'hadronic']
_GENIE_KWS = ['genie', 'reweight_sbn', 'zexp', 'rpa', 'coulomb', 'mec',
              'ncel', 'ccres', 'ncres', 'nonres', 'rdecbr', 'theta_delta',
              'normc', 'mfp', 'frabs', 'frcel', 'frinel', 'frpiprod',
              'ahtby', 'bhtby', 'cv1u', 'cv2u', 'qe', 'dis', 'fsi',
              'coh', 'nccoh', 'cccoh', 'intf', 'valenciaweight',
              'martini', 'sf_q', 'hf_q', 'crpa', 'edepfsi', 'zexpa']
_EXTRA_KWS = ['extra_xsec', 'minervae2p2h', 'minerva', 'miscinteraction',
              'novastylenon', 'novastyle', 'spplowq2', 'c12toar40',
              'nuenuebar', 'nuenumu']
_G4_KWS = ['reinteraction', 'geant4', 'g4', 'g4reweight']


def _is_univ(c):
    return isinstance(c, tuple) and any('univ' in str(p).lower() for p in c)


def classify_weight_cols(all_cols, verbose=True):
    """Classify universe-weight columns into flux / genie / extra_xsec / g4."""
    univ_cols = [c for c in all_cols if _is_univ(c)]
    groups = defaultdict(list)
    for c in univ_cols:
        groups[c[0]].append(c)

    if verbose:
        print(f"  Systematic groups ({len(groups)} total):")
        for name, cols in sorted(groups.items()):
            print(f"    {name:<60s}: {len(cols)} universes")

    flux_cols, genie_cols, extra_cols, g4_cols = [], [], [], []
    for group_name, cols in groups.items():
        name_lower = str(group_name).lower()
        if any(kw in name_lower for kw in _G4_KWS):
            g4_cols.extend(cols)
        elif any(kw in name_lower for kw in _EXTRA_KWS):
            extra_cols.extend(cols)
        elif any(kw in name_lower for kw in _GENIE_KWS):
            genie_cols.extend(cols)
        elif any(kw in name_lower for kw in _FLUX_KWS):
            flux_cols.extend(cols)
        else:
            warnings.warn(f"Unknown group '{group_name}' ({len(cols)} cols) "
                          f"— assigning to 'extra_xsec'")
            extra_cols.extend(cols)

    result = {'flux': flux_cols, 'genie': genie_cols,
              'extra_xsec': extra_cols, 'g4': g4_cols}
    if verbose:
        print(f"  Classification:")
        for src, cols in result.items():
            print(f"    {src:<12}: {len(cols)} cols")
    return result


def load_and_classify_weights(weights_file, label='MC', verbose=True):
    """Load mcnu weights file and classify columns.

    Returns (mcnu_df, src_cols_dict).
    """
    if verbose:
        print(f'Loading {label} weights ...')
    mcnu_df = load_hdf_key(weights_file, 'mcnu')
    src_cols = classify_weight_cols(list(mcnu_df.columns), verbose=verbose)
    if verbose:
        print(f'{label} weights: {mcnu_df.shape}  index={mcnu_df.index.names}')
    return mcnu_df, src_cols


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Weight alignment
# ══════════════════════════════════════════════════════════════════════════════

def _flat(cols):
    return ['|'.join(str(p) for p in c).strip('|') if isinstance(c, tuple)
            else str(c) for c in cols]


def align_weights(evtdf_file, mcnu_df_in, src_cols_dict, verbose=True):
    """Memory-efficient weight alignment via integer row mapping.

    Returns (wgt_aligned DataFrame, aligned_src_cols dict).
    """
    # Step 1: read mct_index from evtdf
    with pd.HDFStore(evtdf_file, mode='r') as store:
        keys = store.keys()
        if '/split' in keys:
            n = int(store['split']['n_split'].iloc[0])
            evt_keys = [f'/evt_{k}' for k in range(n)]
        else:
            evt_keys = sorted(k for k in keys
                              if k.lstrip('/').startswith('evt'))

    with pd.HDFStore(evtdf_file, mode='r') as store:
        sample_cols = store[evt_keys[0]].columns.tolist()

    mct_col = next((c for c in sample_cols
                    if 'mct_index' in str(c) and 'dlp_true' in str(c)), None)
    if mct_col is None:
        mct_col = next((c for c in sample_cols if 'mct_index' in str(c)), None)
    if verbose:
        print(f'  mct_index column: {mct_col}')

    mct_frames = []
    for ek in tqdm(evt_keys, desc='  reading mct_index', leave=False):
        with pd.HDFStore(evtdf_file, mode='r') as store:
            chunk = store[ek][[mct_col]]
        il = chunk.index.names[:-1] if chunk.index.nlevels > 1 else chunk.index.names
        mct_frames.append(chunk.groupby(level=il).first())
        del chunk
    mct_inter = pd.concat(mct_frames)
    del mct_frames; gc.collect()

    inter_idx = list(mct_inter.index.names)
    full_idx = mct_inter.index.copy()

    # Step 2: integer row mapping
    mcnu_idx = list(mcnu_df_in.index.names)
    mct_r = mct_inter.reset_index()
    mct_r.columns = _flat(mct_r.columns)
    inter_idx_flat = _flat(inter_idx)
    mcnu_idx_flat = _flat(mcnu_idx)
    mct_col_flat = _flat([mct_col])[0]

    mcnu_key = mcnu_df_in.index.to_frame(index=False)
    mcnu_key.columns = mcnu_idx_flat
    mcnu_key['_iloc'] = np.arange(len(mcnu_key), dtype=np.int32)

    merged_map = mct_r.merge(
        mcnu_key,
        left_on=[inter_idx_flat[0], inter_idx_flat[1], mct_col_flat],
        right_on=[mcnu_idx_flat[0], mcnu_idx_flat[1], mcnu_idx_flat[2]],
        how='left')
    row_idx = merged_map['_iloc'].values.astype(np.float64)
    has_match = ~np.isnan(row_idx)
    row_idx_safe = np.where(has_match, row_idx, 0).astype(np.int32)

    del mcnu_key, merged_map, mct_r, mct_inter; gc.collect()
    if verbose:
        print(f'  Row mapping: {has_match.sum()}/{len(has_match)} matched '
              f'({100 * has_match.mean():.1f}%)')

    # Step 3: reindex each source
    aligned_frames = []
    aligned_src_cols = {}
    for src, cols in src_cols_dict.items():
        if not cols:
            aligned_src_cols[src] = []
            continue
        if verbose:
            print(f'  Aligning {src} ({len(cols)} cols) ...', end=' ', flush=True)
        arr = mcnu_df_in[cols].values[row_idx_safe]
        arr[~has_match] = np.nan
        df_part = pd.DataFrame(arr, columns=cols, index=full_idx)
        aligned_frames.append(df_part)
        aligned_src_cols[src] = cols
        if verbose:
            print(f'NaN {np.isnan(arr).mean():.1%}')
        del arr; gc.collect()

    wgt_aligned = pd.concat(aligned_frames, axis=1)
    wgt_aligned.index.names = inter_idx
    del aligned_frames; gc.collect()

    nan_frac = wgt_aligned.isna().mean().mean()
    if verbose:
        print(f'  wgt_aligned: {wgt_aligned.shape}  NaN frac: {nan_frac:.1%}')
    if nan_frac > 0.5:
        warnings.warn('More than 50% NaN — check evtdf_file matches WEIGHTS_FILE')

    return wgt_aligned, aligned_src_cols


# ══════════════════════════════════════════════════════════════════════════════
# 5.  On-beam data loader
# ══════════════════════════════════════════════════════════════════════════════

_m_e = 0.511  # MeV/c²


def patch_vertex_cols(evtdf):
    """Rename vertex I0/I1/I2 → x/y/z for data files."""
    rename_map = {}
    for c in evtdf.columns:
        if isinstance(c, tuple) and 'vertex' in c:
            for old, new in [('I0', 'x'), ('I1', 'y'), ('I2', 'z')]:
                if old in c:
                    parts = list(c)
                    parts[parts.index(old)] = new
                    rename_map[c] = tuple(parts)
    if rename_map:
        evtdf.columns = pd.MultiIndex.from_tuples(
            [rename_map.get(c, c) for c in evtdf.columns])
    return evtdf, len(rename_map)


def extract_reco(evtdf, stage_idx, il):
    """Extract reco KE, costheta, p for leading electron at a given stage."""
    import nue_helpers as nh
    rp = evtdf["rec"]["dlp"]["particles"]
    ke_c = ('ke', '') if ('ke', '') in rp.columns else 'ke'
    pid_c = ('pid', '') if ('pid', '') in rp.columns else 'pid'
    cal_c = ('calo_ke', '') if ('calo_ke', '') in rp.columns else 'calo_ke'

    ele = rp[rp.index.droplevel(-1).isin(stage_idx) & (rp[pid_c] == 1)]
    if ele.empty:
        return {'reco_ke': np.array([]), 'reco_costheta': np.array([]),
                'reco_p': np.array([])}

    ke_src = ele[cal_c] if cal_c in ele.columns else ele[ke_c]
    lead_idx = ke_src.groupby(level=il).idxmax().dropna()
    if lead_idx.empty:
        return {'reco_ke': np.array([]), 'reco_costheta': np.array([]),
                'reco_p': np.array([])}

    lmi = pd.MultiIndex.from_tuples(lead_idx.values, names=ele.index.names)
    ld = ele.loc[lmi]
    ld.index = lead_idx.index
    reco_ke = ld[cal_c if cal_c in ld.columns else ke_c].dropna()
    reco_p = np.sqrt(((reco_ke + _m_e) ** 2 - _m_e ** 2).clip(0))

    try:
        dx = [c for c in rp.columns if 'start_dir' in str(c) and 'I0' in str(c)][0]
        dy = [c for c in rp.columns if 'start_dir' in str(c) and 'I1' in str(c)][0]
        dz = [c for c in rp.columns if 'start_dir' in str(c) and 'I2' in str(c)][0]
        dd = ele.loc[lmi, [dx, dy, dz]]
        dd.index = lead_idx.index
        pm = np.sqrt(dd[dx] ** 2 + dd[dy] ** 2 + dd[dz] ** 2).replace(0, np.nan)
        reco_cos = (dd[dz] / pm).dropna()
    except Exception:
        reco_cos = pd.Series(np.nan, index=reco_ke.index)

    return {
        'reco_ke': reco_ke.values,
        'reco_costheta': reco_cos.dropna().values,
        'reco_p': reco_p.values,
    }


def load_onbeam_data(data_file, target_pot, pot_scale, verbose=True):
    """Load on-beam data, patch vertex cols, run selection, extract reco.

    Returns dict with keys:
        evtdf, total_pot, pot_scale_to_data, cut_flow,
        reco_final, reco_topo, il
    """
    import nue_selection as ns

    evtdf = pd.read_hdf(data_file, key="evt_0")
    with pd.HDFStore(data_file, mode="r") as store:
        potdf = store["pot_0"]
    total_pot = (float(potdf["TOR860"].sum()) if "TOR860" in potdf.columns
                 else float(potdf.select_dtypes("number").iloc[:, 0].sum()))
    mc_sample_pot = target_pot / pot_scale
    pot_scale_to_data = total_pot / mc_sample_pot

    evtdf, n_renamed = patch_vertex_cols(evtdf)
    if verbose:
        print(f'Data: {evtdf.shape}  POT={total_pot:.3e}  '
              f'pot_scale_to_data={pot_scale_to_data:.4f}  '
              f'renamed {n_renamed} vertex cols')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut_flow = ns.shower_qual_cuts(evtdf)

    il = list(range(evtdf.index.nlevels - 1))

    if verbose:
        print('Data cut flow:')
        for stage, d in cut_flow.items():
            print(f'  {stage:<30s}: {len(d["inter_index"]):>8,}')

    final_stage = 'vertex_distance'
    reco_final = extract_reco(evtdf, cut_flow[final_stage]['inter_index'], il)
    reco_topo = extract_reco(evtdf, cut_flow['single_electron']['inter_index'], il)

    return dict(evtdf=evtdf, total_pot=total_pot,
                pot_scale_to_data=pot_scale_to_data,
                cut_flow=cut_flow, reco_final=reco_final,
                reco_topo=reco_topo, il=il)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Off-beam loader
# ══════════════════════════════════════════════════════════════════════════════

BEAM_DUTY_FRACTION = 0.0725


def _count_gates(hdf_file):
    with pd.HDFStore(hdf_file, "r") as store:
        for key in store.keys():
            if "pot" in key.lower():
                df = store[key]
                for col in ["nspills", "n_spills", "NSpills", "Nspills"]:
                    if col in df.columns:
                        return int(df[col].sum())
        return store["hdr_0"][["run", "subrun"]].drop_duplicates().shape[0]


def _count_offbeam_parts(evtdf, idx, il, ptype):
    import nue_helpers as nh
    rp = nh.reco_particles(evtdf)._df
    pid_col = ('pid', '') if ('pid', '') in rp.columns else 'pid'
    parts = rp[rp.index.droplevel(-1).isin(idx)]
    mask = parts[pid_col].isin([0, 1] if ptype == 'shower' else [2, 3, 4])
    return mask.groupby(level=il).sum().reindex(idx, fill_value=0).values


def load_offbeam_data(offbeam_file, data_file, verbose=True):
    """Load off-beam data, run selection, extract reco at all stages.

    Returns dict with keys:
        scale, cut_flow, reco_final, reco_topo, reco_fv,
        vertex, mult, shower_vars, sb_idx, sb_reco, sb_shower_vars
    """
    import nue_selection as nue_sel
    import nue_helpers as nh

    evtdf, _ = patch_vertex_cols(pd.read_hdf(offbeam_file, key="evt_0"))
    n_gates_on = _count_gates(data_file)
    n_gates_off = _count_gates(offbeam_file)
    scale = (1 - BEAM_DUTY_FRACTION) * n_gates_on / n_gates_off

    if verbose:
        print(f'Offbeam: {evtdf.shape}  gates on={n_gates_on} '
              f'off={n_gates_off}  scale={scale:.6f}')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut_flow = nue_sel.shower_qual_cuts(evtdf)

    il = list(range(evtdf.index.nlevels - 1))

    if verbose:
        for stage, d in cut_flow.items():
            n = len(d['inter_index'])
            print(f'  {stage:<30s}: {n:>8,}  →  {n * scale:>7.1f}')

    reco_final = extract_reco(evtdf, cut_flow['vertex_distance']['inter_index'], il)
    reco_topo = extract_reco(evtdf, cut_flow['single_electron']['inter_index'], il)
    reco_fv = extract_reco(evtdf, cut_flow['fiducial']['inter_index'], il)

    # Vertex coords at FV stage
    fv_idx = cut_flow['fiducial']['inter_index']
    ri = evtdf["rec"]["dlp"]
    vertex = {}
    for coord, cands in [('x', [('vertex', 'x', ''), ('vertex', 'x')]),
                         ('y', [('vertex', 'y', ''), ('vertex', 'y')]),
                         ('z', [('vertex', 'z', ''), ('vertex', 'z')])]:
        for cc in cands:
            if cc in ri.columns:
                vals = ri[cc].groupby(level=il).first().reindex(fv_idx).dropna()
                vertex[coord] = vals.values
                break

    # Multiplicity at FV
    mult = {
        'shower': _count_offbeam_parts(evtdf, fv_idx, il, 'shower'),
        'track': _count_offbeam_parts(evtdf, fv_idx, il, 'track'),
    }

    # Shower vars at topology
    topo_idx = cut_flow['single_electron']['inter_index']
    shower_vars = {}
    for col_key in [('primary_scores', 'I1'), ('pid_scores', 'I1'),
                    ('vertex_distance', ''), ('start_dedx', ''), ('calo_ke', '')]:
        vals = nh._get_leading_electron_var(evtdf, col_key, topo_idx, il)
        shower_vars[col_key[0]] = vals.dropna().values if not vals.empty else np.array([])

    # Sideband
    sb_idx, sb_reco, sb_shower_vars = build_sideband_idx_and_reco(
        evtdf, topo_idx, il, extract_reco)

    del evtdf; gc.collect()

    return dict(scale=scale, cut_flow=cut_flow,
                reco_final=reco_final, reco_topo=reco_topo,
                reco_fv=reco_fv, vertex=vertex, mult=mult,
                shower_vars=shower_vars,
                sb_idx=sb_idx, sb_reco=sb_reco,
                sb_shower_vars=sb_shower_vars)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  lowE dirt loader
# ══════════════════════════════════════════════════════════════════════════════

def load_lowE_sample(lowe_file, target_pot, data_total_pot, verbose=True):
    """Load lowE dirt sample, run selection, extract reco.

    Returns dict with keys:
        evtdf, sel, pot_scale, pot_scale_to_data, cut_flow,
        reco_final, reco_topo, vertex, il, reco_ke, reco_cos, reco_p
    """
    import nue_selection as nue_sel
    from mem_optimizer import load_evtdf_slim

    evtdf = load_evtdf_slim(lowe_file, key=None, mode="reco_min_true")
    truth_info = load_hdf_key(lowe_file, 'truth_info')
    histpotdf = load_hdf_key(lowe_file, 'histpotdf')

    _col = ("pot" if "pot" in histpotdf.columns
            else histpotdf.select_dtypes("number").columns[0])
    total_pot = float(histpotdf[_col].sum())
    pot_scale = target_pot / total_pot
    pot_scale_to_data = data_total_pot / total_pot

    if verbose:
        print(f'LowE: {evtdf.shape}  POT={total_pot:.3e}  '
              f'scale={pot_scale:.4f}  scale_to_data={pot_scale_to_data:.6f}')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut_flow = nue_sel.shower_qual_cuts(evtdf)

    il = list(range(evtdf.index.nlevels - 1))

    if verbose:
        for stage, d in cut_flow.items():
            print(f'  {stage:<30s}: {len(d["inter_index"]):>8,}')

    reco_ke = nue_sel.leading_electron_ke(evtdf)
    reco_cos = nue_sel.leading_electron_costheta(evtdf)
    reco_p = nue_sel.leading_electron_p(evtdf)

    # Build mini sel DataFrame
    presel_inter = evtdf.groupby(level=il).size().index
    truth_cat = truth_info['truth_cat'].reindex(presel_inter)

    sel = pd.DataFrame(index=presel_inter)
    sel['truth_cat'] = truth_cat.fillna(7).astype('int8')
    sel['is_sig'] = (sel['truth_cat'] == 0)
    sel['reco_ke'] = reco_ke.reindex(presel_inter).astype('float32')
    sel['reco_costheta'] = reco_cos.reindex(presel_inter).astype('float32')
    sel['reco_p'] = reco_p.reindex(presel_inter).astype('float32')
    for stage_name in cut_flow:
        sel[f'sel_{stage_name}'] = presel_inter.isin(
            cut_flow[stage_name]['inter_index'])
    sel['pot_scale'] = np.float32(pot_scale)

    def _reco_dict(idx):
        return {
            'reco_ke': reco_ke.reindex(idx).dropna().values,
            'reco_costheta': reco_cos.reindex(idx).dropna().values,
            'reco_p': reco_p.reindex(idx).dropna().values,
        }

    final_idx = cut_flow['vertex_distance']['inter_index']
    topo_idx = cut_flow['single_electron']['inter_index']

    # Vertex at FV
    import nue_helpers as nh
    ri = nh.reco_interactions(evtdf)._df
    vertex = {}
    for coord, cands in [('x', [('vertex', 'x', ''), ('vertex', 'x')]),
                         ('y', [('vertex', 'y', ''), ('vertex', 'y')]),
                         ('z', [('vertex', 'z', ''), ('vertex', 'z')])]:
        for cc in cands:
            if cc in ri.columns:
                vals = (ri[cc].groupby(level=il).first()
                        .reindex(cut_flow['fiducial']['inter_index']).dropna())
                vertex[coord] = vals.values
                break

    return dict(evtdf=evtdf, sel=sel, pot_scale=pot_scale,
                pot_scale_to_data=pot_scale_to_data,
                cut_flow=cut_flow, il=il,
                reco_final=_reco_dict(final_idx),
                reco_topo=_reco_dict(topo_idx),
                vertex=vertex,
                reco_ke=reco_ke, reco_cos=reco_cos, reco_p=reco_p)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Sideband builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sideband_idx_and_reco(evtdf, topo_idx, il, extract_reco_fn):
    """Build photon-enriched sideband index and extract reco dict."""
    import nue_selection as _ns
    import nue_helpers as nh

    rp = nh.reco_particles(evtdf)._df
    ke_c = ('ke', '') if ('ke', '') in rp.columns else 'ke'
    pi_c = ('pid', '') if ('pid', '') in rp.columns else 'pid'

    ele = rp[rp.index.droplevel(-1).isin(topo_idx) & (rp[pi_c] == 1)]
    if ele.empty:
        empty = pd.Index([])
        return empty, {k: np.array([]) for k in ['reco_ke', 'reco_costheta', 'reco_p']}, {}

    li = ele[ke_c].groupby(level=il).idxmax().dropna()
    if li.empty:
        empty = pd.Index([])
        return empty, {k: np.array([]) for k in ['reco_ke', 'reco_costheta', 'reco_p']}, {}

    lmi = pd.MultiIndex.from_tuples(li.values, names=ele.index.names)
    ld = ele.loc[lmi].copy()
    ld.index = li.index

    sb = pd.Series(True, index=ld.index)
    vd_c = ('vertex_distance', '')
    ps_c = ('pid_scores', 'I1')
    dx_c = ('start_dedx', '')
    if vd_c in ld.columns:
        sb &= (ld[vd_c] > _ns.THRESH_VERTEX_DIST)
    if ps_c in ld.columns:
        sb &= (ld[ps_c] < _ns.THRESH_PID_SCORE)
    if dx_c in ld.columns:
        sb &= (ld[dx_c] > _ns.THRESH_SB_DEDX_LO) & (ld[dx_c] < _ns.THRESH_SB_DEDX_HI)

    sb_idx = ld.index[sb]
    reco = extract_reco_fn(evtdf, sb_idx, il)

    shower_vars = {}
    for col_key in [('primary_scores', 'I1'), ('pid_scores', 'I1'),
                    ('vertex_distance', ''), ('start_dedx', ''), ('calo_ke', '')]:
        vals = nh._get_leading_electron_var(evtdf, col_key, sb_idx, il)
        shower_vars[col_key[0]] = vals.dropna().values if not vals.empty else np.array([])

    return sb_idx, reco, shower_vars


# ══════════════════════════════════════════════════════════════════════════════
# 9.  lowE auxiliary extractors (mult, shower vars, sideband)
# ══════════════════════════════════════════════════════════════════════════════

def extract_lowE_extras(lowE_data, extract_reco_fn=None):
    """Extract multiplicity, shower vars, and sideband from lowE sample.

    Parameters
    ----------
    lowE_data : dict returned by load_lowE_sample

    Returns updated lowE_data dict with added keys:
        mult, shower_vars, sb_idx, sb_reco, sb_shower_vars
    """
    import nue_helpers as nh

    evtdf = lowE_data['evtdf']
    cut_flow = lowE_data['cut_flow']
    il = lowE_data['il']
    fv_idx = cut_flow['fiducial']['inter_index']
    topo_idx = cut_flow['single_electron']['inter_index']

    if extract_reco_fn is None:
        extract_reco_fn = extract_reco

    # Multiplicity
    mult = {}
    for pt in ['shower', 'track']:
        rp = nh.reco_particles(evtdf)._df
        pid_col = ('pid', '') if ('pid', '') in rp.columns else 'pid'
        parts = rp[rp.index.droplevel(-1).isin(fv_idx)]
        mask = parts[pid_col].isin([0, 1] if pt == 'shower' else [2, 3, 4])
        mult[pt] = mask.groupby(level=il).sum().reindex(fv_idx, fill_value=0).values

    # Shower vars at topology
    shower_vars = {}
    for col_key in [('primary_scores', 'I1'), ('pid_scores', 'I1'),
                    ('vertex_distance', ''), ('start_dedx', ''), ('calo_ke', '')]:
        vals = nh._get_leading_electron_var(evtdf, col_key, topo_idx, il)
        shower_vars[col_key[0]] = vals.dropna().values if not vals.empty else np.array([])

    # Sideband
    sb_idx, sb_reco, sb_shower_vars = build_sideband_idx_and_reco(
        evtdf, topo_idx, il, extract_reco_fn)

    def _reco_dict(idx):
        return {
            'reco_ke': lowE_data['reco_ke'].reindex(idx).dropna().values,
            'reco_costheta': lowE_data['reco_cos'].reindex(idx).dropna().values,
            'reco_p': lowE_data['reco_p'].reindex(idx).dropna().values,
        }

    lowE_data['mult'] = mult
    lowE_data['shower_vars'] = shower_vars
    lowE_data['sb_idx'] = sb_idx
    lowE_data['sb_reco'] = _reco_dict(sb_idx)
    lowE_data['sb_shower_vars'] = sb_shower_vars
    return lowE_data


# ============================================================
# chi2 / p-value utilities
# ============================================================

def chi2_pvalue(observed, expected, cov_matrix=None, stat_errors=None):
    from scipy.stats import chi2 as chi2_dist
    obs  = np.asarray(observed, dtype=float)
    exp  = np.asarray(expected, dtype=float)
    diff = obs - exp
    n    = len(diff)
    if cov_matrix is not None:
        cov = np.asarray(cov_matrix, dtype=float)
        reg = np.eye(n) * max(1e-10, 1e-6 * np.abs(np.diag(cov)).max())
        try:
            cov_inv = np.linalg.inv(cov + reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov + reg)
        chi2_val = float(diff @ cov_inv @ diff)
    else:
        if stat_errors is not None:
            sigma2 = np.asarray(stat_errors, dtype=float) ** 2
        else:
            sigma2 = np.where(exp > 0, exp, 1.0)
        chi2_val = float(np.sum(diff**2 / np.where(sigma2 > 0, sigma2, 1.0)))
    ndof = n
    pval = float(1.0 - chi2_dist.cdf(chi2_val, ndof))
    return chi2_val, ndof, pval


def chi2_text(observed, expected, cov_matrix=None, stat_errors=None, label=""):
    chi2_val, ndof, pval = chi2_pvalue(observed, expected,
                                        cov_matrix=cov_matrix,
                                        stat_errors=stat_errors)
    prefix = f"{label}" if label else ""
    return f"{prefix}$\\chi^2$/ndof = {chi2_val:.1f}/{ndof} (p = {pval:.2f})"


def annotate_chi2(ax, observed, expected, cov_matrix=None, stat_errors=None,
                  label="", loc="upper right", fontsize=10):
    chi2_val, ndof, pval = chi2_pvalue(observed, expected,
                                        cov_matrix=cov_matrix,
                                        stat_errors=stat_errors)
    text = chi2_text(observed, expected, cov_matrix=cov_matrix,
                     stat_errors=stat_errors, label=label)
    ha_map = {"upper right": ("right", 0.97), "upper left": ("left", 0.03),
              "lower right": ("right", 0.97), "lower left": ("left", 0.03)}
    va_map = {"upper right": ("top", 0.88),   "upper left": ("top", 0.88),
              "lower right": ("bottom", 0.05), "lower left": ("bottom", 0.05)}
    ha, x = ha_map.get(loc, ("right", 0.97))
    va, y = va_map.get(loc, ("top", 0.88))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.3))
    return chi2_val, ndof, pval


# ============================================================
# Colour / style palette
# ============================================================

CAT_COLORS = {
    0:  "#56B4E9",   1:  "#0072B2",   2:  "green",     3:  "#009E73",
    4:  "#E69F00",   5:  "#CC79A7",   6:  "#F0E442",   7:  "#BDBDBD",
    9:  "#D2691E",  -1:  "#7F7F7F",
}
CAT_LABELS = {
    0:  r"$\nu_e$ CC (FV)",          1:  r"$\nu_e$ CC (out FV)",
    2:  r"$\nu_\mu$ CC $\pi^0$",     3:  r"$\nu$ NC $\pi^0$",
    4:  r"Other $\nu_\mu$ CC",       5:  r"Other $\nu$ NC",
    6:  r"Cosmic",                   7:  r"Offbeam",
    9:  r"Dirt $\nu$",              -1:  r"Unknown",
}

PID_PHOTON   = 0
PID_ELECTRON = 1
PID_MUON     = 2
PID_PION     = 3
PID_PROTON   = 4

PID_NAMES  = {0:"Photon",   1:"Electron", 2:"Muon",    3:"Pion",    4:"Proton"}
PID_COLORS = {0:"#CC79A7",  1:"#9B8EC4",  2:"#56C8C8", 3:"#88CCAA", 4:"#AADDAA"}


# ============================================================
# ParticleView / InteractionView wrappers
# ============================================================

class ParticleView:
    def __init__(self, df: pd.DataFrame):
        self._df = df
    def __getattr__(self, name: str):
        try:    return self._df[name]
        except KeyError: raise AttributeError(f"No column '{name}' in ParticleView") from None
    def __getitem__(self, key):
        if isinstance(self._df.index, pd.MultiIndex):
            try:    return self._df.xs(key, level=0)
            except KeyError: return self._df.loc[key]
        return self._df.loc[key]
    def __len__(self):           return len(self._df)
    def __repr__(self):          return f"ParticleView({len(self._df):,} rows)"
    @property
    def df(self):                return self._df
    @property
    def index(self):             return self._df.index
    @property
    def columns(self):           return self._df.columns
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)


class InteractionView:
    def __init__(self, df: pd.DataFrame):
        self._df = df
    def __getattr__(self, name: str):
        try:    return self._df[name]
        except KeyError: raise AttributeError(f"No column '{name}' in InteractionView") from None
    def __getitem__(self, key):  return self._df[key]
    def __len__(self):           return len(self._df)
    def __repr__(self):          return f"InteractionView({len(self._df):,} rows)"
    @property
    def df(self):                return self._df
    @property
    def index(self):             return self._df.index
    def groupby(self, *a, **kw): return self._df.groupby(*a, **kw)


def reco_particles(evtdf)    -> ParticleView:    return ParticleView(evtdf["rec"]["dlp"]["particles"])
def true_particles(evtdf)    -> ParticleView:    return ParticleView(evtdf["rec"]["dlp_true"]["particles"])
def reco_interactions(evtdf) -> InteractionView: return InteractionView(evtdf["rec"]["dlp"])
def true_interactions(evtdf) -> InteractionView: return InteractionView(evtdf["rec"]["dlp_true"])


# ============================================================
# POT / index utilities
# ============================================================

def compute_pot_scale(histpotdf, target_pot: float = 6.6e20):
    total_pot = histpotdf["TotalPOT"].sum()
    return total_pot, target_pot / total_pot

def inter_levels(evtdf) -> list:
    return list(range(evtdf.index.nlevels - 1))

def collapse_to_interactions(evtdf):
    return evtdf.groupby(level=inter_levels(evtdf)).first()


# ============================================================
# Selection DataFrame I/O
# ============================================================

_SEL_HDF_KEY = "sel_0"

def build_and_save_selection(path, presel_inter, truth_cat_presel,
                              reco_ke, reco_cos, true_ke, true_cos,
                              stage_idx_dict, cut_names, pot_scale,
                              reco_p=None, true_p=None):
    import os, warnings
    sel = pd.DataFrame(index=presel_inter)
    sel["truth_cat"]    = truth_cat_presel.reindex(presel_inter).astype("int8")
    sel["is_sig"]       = (sel["truth_cat"] == 0)
    sel["reco_ke"]      = reco_ke.reindex(presel_inter).astype("float32")
    sel["reco_costheta"]= reco_cos.reindex(presel_inter).astype("float32")
    if reco_p is not None:
        sel["reco_p"]   = reco_p.reindex(presel_inter).astype("float32")
    if true_ke is not None:
        sel["true_ke"]  = true_ke.reindex(presel_inter).astype("float32")
    if true_cos is not None:
        sel["true_costheta"] = true_cos.reindex(presel_inter).astype("float32")
    if true_p is not None:
        sel["true_p"]   = true_p.reindex(presel_inter).astype("float32")
    for name in cut_names:
        if name in stage_idx_dict:
            sel[f"sel_{name}"] = presel_inter.isin(stage_idx_dict[name])
        else:
            warnings.warn(f"build_and_save_selection: stage {name!r} missing")
    sel["pot_scale"] = np.float32(pot_scale)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel.to_hdf(path, key=_SEL_HDF_KEY, mode="w", complevel=1, complib="blosc")
    size_mb = os.path.getsize(path) / 1024**2
    print(f"Saved selection df → {path}  shape={sel.shape}  size={size_mb:.1f} MB")
    return sel

def load_selection(path):
    return pd.read_hdf(path, key=_SEL_HDF_KEY)


# ============================================================
# Purity / efficiency
# ============================================================

def calc_pur_eff(sel_mask, sig_mask, weights=None):
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w = np.ones(len(sel_mask)) if weights is None else np.asarray(weights)
    n_sel      = w[sel_mask].sum()
    n_true_pos = w[sel_mask & sig_mask].sum()
    n_sig      = w[sig_mask].sum()
    pur = float(n_true_pos / n_sel) if n_sel > 0 else np.nan
    eff = float(n_true_pos / n_sig) if n_sig > 0 else np.nan
    return pur, eff

def pur_eff_cut_flow(cut_flow_masks, sig_mask, weights=None):
    return zip(*[calc_pur_eff(m, sig_mask, weights) for m in cut_flow_masks])

def pur_eff_binned(values, bins, sel_mask, sig_mask, weights=None):
    values   = np.asarray(values, dtype=float)
    sel_mask = np.asarray(sel_mask, dtype=bool)
    sig_mask = np.asarray(sig_mask, dtype=bool)
    w        = np.ones(len(values)) if weights is None else np.asarray(weights)
    bins     = np.asarray(bins)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    purs = np.full(len(centers), np.nan)
    effs = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin     = (values >= lo) & (values < hi)
        counts[i]  = w[in_bin & sig_mask].sum()
        purs[i], effs[i] = calc_pur_eff(sel_mask & in_bin, sig_mask & in_bin, w)
    return centers, purs, effs, counts


# ============================================================
# Internal helpers used by nue_plotter (kept here for compat)
# ============================================================

def _get_interaction_var(evtdf, col_key, stage_idx, il):
    ri_df = reco_interactions(evtdf)._df
    if col_key not in ri_df.columns:
        for c in ri_df.columns:
            if isinstance(c, tuple) and all(k in c for k in col_key if k):
                col_key = c; break
    return ri_df[col_key].groupby(level=il).first().reindex(stage_idx).dropna()


def _get_leading_electron_var(evtdf, col_key, stage_idx, il):
    rp_df   = reco_particles(evtdf)._df
    ke_col  = ('ke','')  if ('ke','')  in rp_df.columns else 'ke'
    pid_col = ('pid','') if ('pid','') in rp_df.columns else 'pid'
    ele = rp_df[rp_df.index.droplevel(-1).isin(stage_idx) & (rp_df[pid_col] == PID_ELECTRON)]
    if ele.empty or col_key not in ele.columns:
        return pd.Series(dtype=float)
    lead_idx = ele[ke_col].groupby(level=il).idxmax().dropna()
    if lead_idx.empty: return pd.Series(dtype=float)
    lead_mi = pd.MultiIndex.from_tuples(lead_idx.values, names=ele.index.names)
    vals = ele.loc[lead_mi, col_key]; vals.index = lead_idx.index
    return vals


def _by_cat_generic(var_series, truth_cat_series, cat):
    mask   = (truth_cat_series == cat)
    common = var_series.index.intersection(mask.index)
    return var_series.loc[common][mask.loc[common]].dropna().values


# ============================================================
# Backward-compatible re-exports from nue_plotter
# ============================================================
# So existing code doing `from nue_helpers import plot_stacked_hist` still works.

try:
    from nue_plotter import (
        plot_stacked_hist,
        plot_unblinding_var,
        plot_differential_slices,
        plot_extra_selection_vars,
        plot_fracunc,
        plot_heatmap_custom,
        format_heatmap_value,
        plot_hist2d_frac_err,
        plot_leading_e_resolution,
        threshold_plots,
        threshold_plots_cumulative,
        plot_eff_pur_vs_bin,
        create_purity_efficiency_table,
        plot_matrix_binned,
        plot_pid_fraction_vs_ke,
        save_plot,
        plot_lynn_comparison,
        SRC_LABEL, COLORS,
        _mat,
    )
except ImportError:
    pass  # nue_plotter not yet on path