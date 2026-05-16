import uproot
import numpy as np
import pandas as pd
import awkward as ak

# Read weight data in small chunks to avoid loading the entire
# rec.mc.nu.wgt.univ array (~5-10 GB) into memory at once.
CHUNK_SIZE = "10 MB"


def _report_entry_start(report):
    """Get global entry start from an uproot iterate report object."""
    for attr in ("global_entry_start", "entry_start", "start"):
        if hasattr(report, attr):
            return int(getattr(report, attr))
    if isinstance(report, dict):
        for key in ("global_entry_start", "entry_start", "start"):
            if key in report:
                return int(report[key])
    return 0


def _read_wgt_metadata(f):
    wgt_names = list(f["globalTree"]["global/wgts/wgts.name"].arrays(library="np")["wgts.name"][0])
    wgt_types = f["globalTree"]["global/wgts/wgts.type"].arrays(library="np")["wgts.type"][0]
    wgt_nuniv = f["globalTree"]["global/wgts/wgts.nuniv"].arrays(library="np")["wgts.nuniv"][0]
    return wgt_names, wgt_types, wgt_nuniv


def _build_isyst_map(wgt_nuniv):
    isyst_map = pd.Series(np.repeat(list(range(len(wgt_nuniv))), wgt_nuniv), name="isyst")
    isyst_map.index.name = "iwgt"
    return isyst_map


def _iter_filtered_wgts(rec_tree, nuniv, isyst_map, needed_syst_indices, step_size):
    """
    Yield filtered weight DataFrames chunk-by-chunk to cap peak memory.

    Each yielded chunk contains only the rows belonging to needed_syst_indices,
    with entry indices corrected to the global (file-level) position.
    """
    for batch, report in rec_tree.iterate(
        filter_name=["rec.mc.nu.wgt.univ"],
        library="ak",
        step_size=step_size,
        report=True,
    ):
        entry_offset = _report_entry_start(report)
        wgts = ak.to_dataframe(batch["rec.mc.nu.wgt.univ"], how=None)[0]
        wgts["inu"] = wgts.index.get_level_values(1) // nuniv
        wgts["iwgt"] = wgts.index.get_level_values(1) % nuniv
        wgts = wgts.reset_index()
        wgts["entry"] = wgts["entry"] + entry_offset   # local → global entry index
        wgts = wgts.set_index(["entry", "inu", "iwgt"]).drop(columns="subentry")
        wgts.columns = ["wgt"]
        wgts = wgts.join(isyst_map)
        # Keep only the systematics we actually need
        wgts = wgts[wgts.isyst.isin(needed_syst_indices)]
        if len(wgts):
            yield wgts


def _slim_mul(systs_slim, col, wgt_series):
    """
    Multiply systs_slim[col] by wgt_series in-place, safely handling the case
    where wgt_series.index is a SUPERSET of systs_slim.index (i.e. wgt_series
    has rows for non-preselected nus that don't exist in systs_slim).

    Lynn's original code used `systs_slim.loc[wgt_series.index, col]` directly,
    which raises a KeyError when any label in wgt_series.index is absent from
    systs_slim.index.  This helper intersects first to avoid that error.
    """
    shared = wgt_series.index.intersection(systs_slim.index)
    if len(shared) == 0:
        return
    systs_slim.loc[shared, col] = (
        systs_slim.loc[shared, col].values * wgt_series.loc[shared].values
    )


def getsyst(f, systematics, nuind, multisim_nuniv=100, slim=False, slimname="slim"):
    """
    Compute systematic weight universes for the given nu indices.

    Parameters
    ----------
    f          : uproot file handle
    systematics: list of systematic names to compute
    nuind      : pd.Series — index = (entry, ...) MultiIndex,
                 values  = per-event integer nu index.
                 Can be a SUBSET of all nus in the file.
    multisim_nuniv: number of multisim universes to keep (default 100)
    slim       : if True, multiply all systs into a single N-universe product
    slimname   : column prefix for slim output

    Memory profile
    --------------
    Weight data is read in CHUNK_SIZE pieces (default 10 MB) so peak memory
    from the weight branch is bounded regardless of file size.
    The slim systs_slim DataFrame scales with len(nuind), not with the full
    file size.

    Subset safety
    -------------
    nuind may contain only a subset of all nus in the file (e.g. preselected
    interactions only).  All internal updates use _slim_mul() which intersects
    indices before multiplying, so non-preselected nus are cleanly skipped.
    """
    if "globalTree" not in f:
        return pd.DataFrame(index=nuind.index)

    # nuidx: (entry, per-event-nu-index) — must match the (entry, inu) indexing
    # used inside getsyst when reading rec.mc.nu.wgt.univ.
    nuidx = pd.MultiIndex.from_arrays([nuind.index.get_level_values(0), nuind])

    wgt_names, wgt_types, wgt_nuniv = _read_wgt_metadata(f)
    isyst_map = _build_isyst_map(wgt_nuniv)
    nuniv = wgt_nuniv.sum()

    needed_syst_indices = {wgt_names.index(s) for s in systematics if s in wgt_names}
    if len(needed_syst_indices) != len(systematics):
        missing = [s for s in systematics if s not in wgt_names]
        raise Exception("Missing systematic weights in file: %s" % ", ".join(missing))

    if slim:
        cols = pd.MultiIndex.from_product(
            [[slimname], [f"univ_{i}" for i in range(multisim_nuniv)]]
        )
        # Initialized to 1.0; rows will be multiplied in-place during iteration.
        # Size: len(nuind) × multisim_nuniv × 8 bytes — negligible for subsets.
        systs_slim = pd.DataFrame(1.0, index=nuidx, columns=cols)
    else:
        systs_slim = None

    file_id = id(f)
    syst_chunks = {}

    for wgts in _iter_filtered_wgts(
        f["recTree"], nuniv, isyst_map, needed_syst_indices, CHUNK_SIZE
    ):
        for s in systematics:
            isyst = wgt_names.index(s)
            this_systs = []
            filtered = wgts[wgts.isyst == isyst]
            if filtered.empty:
                continue

            if wgt_types[isyst] == 3 and wgt_nuniv[isyst] == 1:  # morph unisim
                s_morph = filtered.wgt.groupby(level=[0, 1]).first()
                s_morph.name = (s, "morph")

                if slim:
                    for i in range(multisim_nuniv):
                        seed_input = s + str(i) + str(file_id)
                        np.random.seed(hash(seed_input) % (2**32))
                        wgt = 1 + (s_morph - 1) * 2 * np.abs(np.random.normal(0, 1))
                        wgt = np.maximum(wgt, 0)
                        col = (slimname, f"univ_{i}")
                        _slim_mul(systs_slim, col, wgt)   # ← safe subset multiply
                else:
                    this_systs.append(s_morph)

            elif wgt_types[isyst] == 3 and wgt_nuniv[isyst] > 1:  # +/- sigma unisim
                nwgt = filtered.wgt.groupby(level=[0, 1]).size().values[0]
                nsigma = nwgt // 2

                pivoted = (
                    filtered.wgt
                    .reset_index(level=2)
                    .pivot_table(values="wgt", index=["entry", "inu"], columns="iwgt")
                )

                for isigma in range(nsigma):
                    s_ps = pivoted.iloc[:, 2 * isigma].rename((s, "ps%i" % (isigma + 1)))
                    s_ms = pivoted.iloc[:, 2 * isigma + 1].rename((s, "ms%i" % (isigma + 1)))

                    if slim and isigma == 0:  # use ps1
                        for i in range(multisim_nuniv):
                            seed_input = s + str(i) + str(file_id)
                            np.random.seed(hash(seed_input) % (2**32))
                            wgt = 1 + (s_ps - 1) * np.random.normal(0, 1)
                            wgt = np.maximum(wgt, 0)
                            col = (slimname, f"univ_{i}")
                            _slim_mul(systs_slim, col, wgt)   # ← safe subset multiply
                    else:
                        this_systs.append(s_ps)
                        this_systs.append(s_ms)

                if nwgt % 2 != 0:
                    s_cv = pivoted.iloc[:, nwgt - 1].rename((s, "cv"))
                    this_systs.append(s_cv)
                else:
                    this_systs.append(pd.Series(1, index=pivoted.index, name=(s, "cv")))

            elif wgt_types[isyst] == 0:  # multisim
                this_wgts = filtered.wgt.groupby(level=[0, 1]).head(multisim_nuniv)
                this_wgts = this_wgts.reset_index(level=2)
                this_wgts = this_wgts.pivot_table(
                    values="wgt", index=["entry", "inu"], columns="iwgt"
                )
                this_wgts.columns = pd.MultiIndex.from_tuples(
                    [(s, "univ_%i" % i) for i in range(len(this_wgts.columns))]
                )

                if slim:
                    for i in range(multisim_nuniv):
                        col = (slimname, f"univ_{i}")
                        src = (s, f"univ_{i}")
                        if src in this_wgts.columns:
                            _slim_mul(systs_slim, col, this_wgts[src])   # ← safe subset multiply

                for c in this_wgts.columns:
                    this_systs.append(this_wgts[c])
            else:
                raise Exception("Cannot decode systematic uncertainty: %s" % s)

            for syst in this_systs:
                syst_chunks.setdefault(syst.name, []).append(syst)

    if slim:
        s_idx = systs_slim.index.get_indexer(nuidx)
        systs_slim.loc[s_idx < 0, :] = 1.0
        systs_slim.index = nuind.index
        return systs_slim

    if not syst_chunks:
        return pd.DataFrame(index=nuind.index)

    systs = []
    for _, chunks in syst_chunks.items():
        if len(chunks) == 1:
            systs.append(chunks[0])
        else:
            merged = pd.concat(chunks, axis=0)
            merged.name = chunks[0].name
            systs.append(merged)

    systs = pd.concat(systs, axis=1)
    if not systs.index.is_unique:
        raise Exception("Internal error: non-unique (entry, inu) index after chunk merge")

    s_idx = systs.index.get_indexer(nuidx)
    systs_match = systs.iloc[s_idx]
    systs_match.loc[s_idx < 0, :] = 1.0
    systs_match.index = nuind.index
    return systs_match


def print_syst_all(f):
    if "globalTree" not in f:
        print("no globalTree")
        return
    wgt_names, wgt_types, wgt_nuniv = _read_wgt_metadata(f)
    for i in range(len(wgt_names)):
        print(f"Index: {i:<3} | Name: {wgt_names[i]:<60} | Type: {wgt_types[i]:<5} | Univ: {wgt_nuniv[i]}")