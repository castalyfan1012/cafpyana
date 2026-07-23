# nueCC Detector Systematics Pipeline


## File Layout

```
analysis_village/nueCC/
├── scripts/
│   ├── nuecc_detsys_config.py         ← config (paths, det-var list)
│   ├── find_detvar_cafs.py            ← Step 0: check SPINE availability
│   ├── make_detvar_filelists.sh       ← Step 0b: generate xrootd file lists
│   ├── run_detvar_dfmaker.sh          ← Step 1: run df maker on det-vars
│   ├── merge_detvar_dfs.py            ← Step 1b: merge per-job .df files
│   ├── match_detvar_events.py         ← Step 2: truth-level event matching
│   ├── build_detvar_covariances.py    ← Step 3: covariance matrices + plots
│   ├── integrate_detsys_cells.py      ← Step 4: notebook integration cells
│   └── filelists/
│       ├── detvar_wiremodxtheta.txt   ← xrootd URLs (auto-generated)
│       └── detvar_wiremodyz.txt
├── makedf/
│   ├── nueCC_mc.py                    ← YOUR existing config
│   └── make_nueCC_df.py               ← YOUR existing df maker
├── nue_selection.py
└── nue_syst_v2.ipynb                  ← add det-sys cells here
```

## Complete Workflow

### Step 0: Check det-var CAFs for SPINE branches
In the following Step 0 instruction:
scripts/ = /exp/sbnd/data/users/castalyf/nue_sel/production_files/detvar

```bash
# From a node that can access /pnfs/ (not EAF — use a GPVM)
python scripts/find_detvar_cafs.py --check \
    /pnfs/sbnd/persistent/users/jzettle/sbnd_wiremod_x_thetaxw/combined_wiremod_x_thetaxw.flat.caf.root

# Or scan production areas
python scripts/find_detvar_cafs.py --scan --check-all
```

If no SPINE branches → talk to Afro about options.

### Step 0b: Generate xrootd file lists

```bash
# Edit make_detvar_filelists.sh to add your det-var /pnfs/ paths
# (wiremodxtheta and wiremodyz are already filled in from jzettle)
# Then run from a node with /pnfs/ access:
bash scripts/make_detvar_filelists.sh

# Verify:
head scripts/filelists/detvar_wiremodxtheta.txt
# → root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/persistent/users/jzettle/...
```

### Step 1: Run your existing df maker on det-var file lists

```bash
# Source your environment
source /exp/sbnd/app/users/castalyf/setup_cafpyana.sh
cd ~/cafpyana/analysis_village/nueCC

# Test with 1 file locally
bash scripts/run_detvar_dfmaker.sh --var wiremodxtheta --nfile 1

# Generate grid scripts for all available variations
bash scripts/run_detvar_dfmaker.sh --var all --grid --njobs 20

# Submit grid jobs
source /exp/sbnd/data/users/castalyf/nue_sel/detsys/gen1/grid_scripts/submit_wiremodxtheta.sh
source /exp/sbnd/data/users/castalyf/nue_sel/detsys/gen1/grid_scripts/submit_wiremodyz.sh
```

### Step 1b: Merge per-job outputs

```bash
# After grid jobs finish
python scripts/merge_detvar_dfs.py

# Or single variation
python scripts/merge_detvar_dfs.py --var wiremodxtheta
```

### Step 2: Match nominal ↔ variation events

```bash
# This can run on EAF (reads from /exp/, not /pnfs/)
python scripts/match_detvar_events.py
```

### Step 3: Build covariance matrices

```bash
# All variables, all variations
python scripts/build_detvar_covariances.py

# Specific variables only
python scripts/build_detvar_covariances.py --variables electron_ke electron_pid_score
```

### Step 4: Add to notebooks

Paste cells from `integrate_detsys_cells.py` into `nue_syst_v2.ipynb`:

```python
DETSYS_COV_DIR = "/exp/sbnd/data/users/castalyf/nue_sel/detsys/gen1/v1/detsys"
data = np.load(f"{DETSYS_COV_DIR}/detsys_electron_ke_cov.npz")
cov_det = data["cov_total"]
cov_total = cov_flux + cov_genie + cov_extra_xsec + cov_mcstat + cov_pot + cov_nt + cov_det
```

## Adding New Variables

Add to `_get_variable_specs()` in `build_detvar_covariances.py`:

```python
{
    "name": "electron_pid_score",
    "column": "electron_pid_score",  # column name in your evtdf
    "bins": np.linspace(0, 1, 11),
    "xlabel": "Electron PID score",
    "is_xsec": False,
},
```

Then rerun Step 3. No grid jobs needed.

## Where to find more det-var samples

You have wiremod from jzettle. For the other 5 variations (pmtgain, pmtqe,
pmtspe, nosce, twicesce), ask:
- Bear (brindenc) — he has them for Pandora at his data paths
- Afro — she may know the SPINE-processed versions
- SBND production group — check the production wiki
- The det-var CAFs may be at: /pnfs/sbnd/persistent/sbndpro/*/detvar/