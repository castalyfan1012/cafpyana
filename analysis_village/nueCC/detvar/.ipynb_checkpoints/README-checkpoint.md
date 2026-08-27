# nueCC Detector Systematics

## Current status (August 2026)

**Placeholder in use.** Official Gen1 wiremod samples are pending.
The test files (`sbnd_wiremod_x_thetaxw`, `sbnd_wiremod_y_z`) used a
Gen2-like SPINE config — incompatible with the Gen1 nominal MC. Do not use
them.

Det-sys uncertainty is currently a **5% flat placeholder** applied to all
analysis bins, conservative relative to Pandora wiremod results (~2-3%).

---

## File layout

```
analysis_village/nueCC/
├── detvar/
│   ├── build_detsys_covariances.py   ← main script (run after Gen1 files arrive)
│   ├── filelists/
│   │   ├── detvar_wiremodxtheta.txt  ← update paths when Gen1 files arrive
│   │   └── detvar_wiremodyz.txt
│   └── README.md

nue_sel/production_files/detvar/
├── make_detvar_filelist.sh           ← generate xrootd filelists from /pnfs/
├── filelists/                        ← xrootd file lists
└── detvar_dfs/                       ← output of run_df_maker.py (populated in Step 2)
```

---

## Full workflow (run when Gen1 samples arrive)

### Step 0 — Get Gen1 file paths from Jacob

Ask Jacob Zettle for `/pnfs/` paths of official Gen1 wiremod det-var CAFs.
Update `filelists/detvar_wiremodxtheta.txt` and `detvar_wiremodyz.txt` with
the correct paths (one xrootd URL per line):

```
root://fndcadoor.fnal.gov:1094//pnfs/sbnd/persistent/.../<file>.flat.caf.root
```

Or generate from a `/pnfs/` directory on a GPVM with xrootd access:

```bash
bash make_detvar_filelist.sh /pnfs/sbnd/persistent/.../wiremodxtheta/ \
    > filelists/detvar_wiremodxtheta.txt
```

---

### Step 1 — Process det-var CAFs through run_df_maker.py

Run on EAF (or submit to grid for large samples). Uses `make_nuecc_evtdf_data`
since det-var CAFs may not have truth branches.

```bash
cd ~/cafpyana

# Test locally with 1 file
python3 run_df_maker.py \
    -c analysis_village/nueCC/makedf/nueCC_detvar.py \
    -l nue_sel/production_files/detvar/filelists/detvar_wiremodxtheta.txt \
    -o nue_sel/production_files/detvar/detvar_dfs/wiremodxtheta/detvar_wiremodxtheta \
    -nfile 1

# Repeat for wiremodyz
python3 run_df_maker.py \
    -c analysis_village/nueCC/makedf/nueCC_detvar.py \
    -l nue_sel/production_files/detvar/filelists/detvar_wiremodyz.txt \
    -o nue_sel/production_files/detvar/detvar_dfs/wiremodyz/detvar_wiremodyz \
    -nfile 1
```

The config `makedf/nueCC_detvar.py`:
```python
from make_nueCC_df import make_nuecc_evtdf_data
from makedf.makedf import make_hdrdf, make_potdf_bnb
DFS   = [make_nuecc_evtdf_data, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "hdr", "pot"]
```

If the Gen1 det-var files have truth branches (MC mode), use `nueCC_mc.py`
instead — this enables truth matching and gives more reliable selection.

---

### Step 2 — Sanity check the .df files

```bash
cd ~/cafpyana/analysis_village/nueCC/detvar

python3 build_detsys_covariances.py --check-detvar wiremodxtheta
python3 build_detsys_covariances.py --check-detvar wiremodyz
```

**Expected output (Gen1, compatible with nominal):**

| Quantity | Nominal | Det-var (expected) |
|----------|---------|-------------------|
| Primary electron rate | ~4.7% | ~4-5% (±few %) |
| KE median of primary e | ~84 MeV | ~80-90 MeV |
| Final selected events | ~5,700 | ~5,000-6,500 |

If you see rates ~0.7% and KE median ~400 MeV, the files are the wrong SPINE
config — do not proceed.

---

### Step 3 — Build covariance matrices

```bash
python3 build_detsys_covariances.py
```

Outputs to `saved_syst/{reco_ke,reco_costheta}/detsys_cov_matrices.npz`.
Fractional shifts should be a few percent per bin, not tens of percent.

---

### Step 4 — Enable in notebook

In `nue_syst_v2.ipynb` Cell 8.1, change:

```python
DETSYS_METHOD = "placeholder"   # ← currently
```
to:
```python
DETSYS_METHOD = "npz"           # ← after Step 3 gives sensible results
```

---

## Notebook Cell 8.1 (copy into nue_syst_v2.ipynb after Cell 8)

```python
# ── Cell 8.1 — Detector systematics ─────────────────────────────────────────
# DETSYS_METHOD = "npz"         # use when Gen1 det-var NPZ files are ready
DETSYS_METHOD = "placeholder"   # 5% flat placeholder — pending Gen1 samples
DETSYS_PLACEHOLDER_FRAC = 0.05

print('=== Detector systematics ===')
print(f'  Method: {DETSYS_METHOD}')

if DETSYS_METHOD == "npz":
    for vcfg in VAR_CFGS:
        path = f'{SYST_DIR}/{vcfg.name}/detsys_cov_matrices.npz'
        if not os.path.exists(path):
            print(f'  [{vcfg.name}] NPZ not found: {path}'); continue
        d  = np.load(path)
        nb = len(vcfg.bins) - 1
        if d['cov_ms_ms'].shape != (nb, nb):
            print(f'  [{vcfg.name}] shape mismatch — skipping'); continue
        cov_results[vcfg.name]['detsys'] = {k: d[k] for k in
            ['cov_ms_ms', 'cov_bs_bs', 'cov_ms_bs', 'cov_bs_ms']}
        print(f'  [{vcfg.name}] loaded ({nb} bins, '
              f'variations: {list(d["variations"])})')

elif DETSYS_METHOD == "placeholder":
    warnings.warn(
        f"Det-sys uses {DETSYS_PLACEHOLDER_FRAC*100:.0f}% flat placeholder. "
        "Pending Gen1 SPINE wiremod samples.", UserWarning)
    for vcfg in VAR_CFGS:
        cv = cv_results[vcfg.name]
        ds = DETSYS_PLACEHOLDER_FRAC * cv['sig_cv']
        db = DETSYS_PLACEHOLDER_FRAC * cv['bkg_cv']
        cov_results[vcfg.name]['detsys'] = {
            'cov_ms_ms': np.outer(ds, ds), 'cov_bs_bs': np.outer(db, db),
            'cov_ms_bs': np.outer(ds, db), 'cov_bs_ms': np.outer(db, ds),
        }
        print(f'  [{vcfg.name}] placeholder {DETSYS_PLACEHOLDER_FRAC*100:.0f}% flat')

SRC_LABEL['detsys'] = 'Detector' + (' (placeholder)' if DETSYS_METHOD == 'placeholder' else '')
COLORS['detsys']    = '#8B0000'

print('\nTotal covariance (updated):')
for vcfg in VAR_CFGS:
    cv = cv_results[vcfg.name]
    ts = sum(_mat(cov_results[vcfg.name][s]['cov_ms_ms'])
             for s in cov_results[vcfg.name])
    tf = np.sqrt(np.diag(ts).sum()) / cv['sig_cv'].sum() if cv['sig_cv'].sum() > 0 else 0
    print(f'  {vcfg.name}: total signal frac unc = {tf*100:.2f}%')
```