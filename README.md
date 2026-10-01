# AFNO_BOB v2.0: Fourier neural operators for Bay of Bengal surface ocean forecasting

This repository contains the code and configurations for:

> J. Rishi and D. N. Subramani. *Adaptive Fourier neural operators for ensemble forecasting of the Bay of Bengal surface ocean.* Department of Computational and Data Sciences, Indian Institute of Science, Bengaluru.

The models forecast daily sea surface temperature (SST), salinity (SSS), height (SSH), and zonal and meridional currents in the Bay of Bengal (4–23°N, 77–99°E). Each model maps the ocean state on day *t* and the ERA5 atmospheric forcing on day *t*+1 to the ocean state on day *t*+1. Multi-day forecasts are produced by feeding predictions back autoregressively. Four Fourier-family neural operators (AFNO, FNO, TFNO, U-NO) are each trained in a single-step (1T) and a three-step rollout (RT) regime, and are compared with persistence and a ConvLSTM baseline.

Version 1.0 of this repository (tag `v1.0`, June 2025; https://doi.org/10.5281/zenodo.15630879) accompanied an earlier version of the manuscript. Version 2.0 is the code used for the current manuscript. It adds:
- the FNO, TFNO, U-NO and ConvLSTM benchmarks;
- rollout training;
- evaluation over 2020–2025;
- TIGGE-driven ensembles and the seasonal and physical-consistency analyses.

The grid-resolution and hyperparameter sensitivity tests in the paper were run with the v1.0 code (tag `v1.0`).

## Trained models, data and checkpoints

The trained models, ancillary data, training checkpoints and evaluation outputs are too large for this repository. Download them from Dropbox:

https://www.dropbox.com/scl/fo/wyumtkxabjn8ixbid0s8p/ALOTXIX5SQKfXVHFgXyK3XE?rlkey=083hvwkma5atozlhku2beir80&dl=0

To download the whole folder as a single zip file instead, use https://www.dropbox.com/scl/fo/wyumtkxabjn8ixbid0s8p/ALOTXIX5SQKfXVHFgXyK3XE?rlkey=083hvwkma5atozlhku2beir80&dl=1.

| Archive | Size | Contents |
|---|---|---|
| `AFNO_BOB-v2.0-models.zip` | 518 MB | Trained weights in `results/models/`: the nine paper models and the five rollout-length ablation models |
| `AFNO_BOB-v2.0-checkpoints.zip` | 1.5 GB | Full training checkpoints (`results/models/checkpoint_*.pth`, model, optimizer and scheduler states) for resuming training |
| `AFNO_BOB-v2.0-ancillary-data.zip` | 136 MB | `data/1993_2020/mean/` normalization statistics (1993–2018) and `data/eof_perturbations/` seasonal EOFs for ensemble initial-state perturbations |
| `AFNO_BOB-v2.0-results.zip` | 3.4 MB | Metrics files behind the paper's tables and figures, training-loss histories, rollout ablation results |
| `SHA256SUMS.txt` | — | Checksums; verify with `sha256sum -c SHA256SUMS.txt` |

Unzip the archives in the repository root. The paths inside them are relative to that root. The normalization statistics in the ancillary archive are needed for any inference or training.

### Trained models

| Paper label | File | Configuration | Training |
|---|---|---|---|
| AFNO 1T | `AFNO_BoB_Surf_E13.pth` | `config/afno_bob_surf_e13.yaml` | single step |
| AFNO RT | `AFNO_BoB_Surf_E14.pth` | `config/afno_bob_surf_e14.yaml` | three-step rollout |
| FNO 1T / RT | `FNO_BoB_Surf_E03.pth` / `E04` | `config/fno_bob_surf_e03.yaml` / `e04` | single step / rollout |
| TFNO 1T / RT | `TFNO_BoB_Surf_E03.pth` / `E04` | `config/tfno_bob_surf_e03.yaml` / `e04` | single step / rollout |
| U-NO 1T / RT | `UNO_BoB_Surf_E03.pth` / `E04` | `config/uno_bob_surf_e03.yaml` / `e04` | single step / rollout |
| ConvLSTM | `ConvLSTM_BoB_Surf_E01.pth` | `config/convlstm_bob_config.yaml` | single step |
| Rollout ablation | `AFNO_Ablation_rs{1..5}.pth` | `config/afno_bob_surf_e11p1.yaml` | 15 epochs, *R* = 1–5 |

The `.pth` files hold the model `state_dict` for inference.

## Installation

```bash
conda env create -f environment.yml
conda activate afno_bob
```

The results were produced with Python 3.12, PyTorch 2.6.0 (CUDA 12.4) and neuraloperator 2.0.0 on NVIDIA RTX A6000 GPUs. `requirements.txt` pins every package version. Run all scripts from the repository root.

## Input data

The input data are public but are not redistributed here. Download and assemble them with `scripts/prepare_data.py`. This requires a Copernicus Marine account (`copernicusmarine login`) and a Copernicus Climate Data Store key in `~/.cdsapirc`.

```bash
# Ocean: GLORYS12V1, GLOBAL_MULTIYEAR_PHY_001_030, uppermost level (0.494 m)
python scripts/prepare_data.py glorys --start 1993-01-01 --end 2020-12-31 --out data/1993_2020/ocean.nc
python scripts/prepare_data.py glorys --start 2021-01-01 --end 2025-09-16 --out data/2021_2025/ocean_2021_2025.nc

# Atmosphere: ERA5 daily means of hourly data (UTC), six variables
python scripts/prepare_data.py era5 --start-year 1993 --end-year 2025 --workdir data/raw_era5
python scripts/prepare_data.py assemble-era5 --workdir data/raw_era5 --start 1993-01-01 --end 2020-12-31 --out data/1993_2020/atm.nc
python scripts/prepare_data.py assemble-era5 --workdir data/raw_era5 --start 2021-01-01 --end 2025-11-26 --out data/2021_2025/atm_2021_2025.nc

# OSTIA SST for 2020 (ensemble figures only)
python scripts/prepare_data.py ostia --start 2020-01-01 --end 2020-12-31 --out data/1993_2020/ostia_2020.nc
```

Expected layout:

```
data/1993_2020/ocean.nc, atm.nc, ostia_2020.nc, mean/   (mean/ is in the ancillary archive)
data/2021_2025/ocean_2021_2025.nc, atm_2021_2025.nc
data/eof_perturbations/                                  (ancillary archive)
data/tigge_ensemble/                                     (see below)
data/argo/{jan,apr,jun,oct}2020/YYYYMMDD_prof.nc          (see below)
```

The code indexes each file by day from its first date (1993-01-01 or 2021-01-01), so each file must be daily and gap-free. ERA5 variables must have dimensions (time, depth=1, latitude, longitude), with latitude ascending. `assemble-era5` produces this layout.

**TIGGE ensemble forecasts.** These drive the ensemble experiments: the ECMWF ensemble, 00 UTC, 50 perturbed members, initialized on 14 Jan, 14 Apr, 14 Jun and 14 Oct 2020. The TIGGE terms of use do not allow redistribution. Accept the TIGGE licence on the ECMWF data store, then run `src/data_pipeline/download_tigge.py --dates 14-01-2020,14-04-2020,14-06-2020,14-10-2020 --ensemble` and `src/data_pipeline/prepare_tigge_ensemble.py` (see the module docstrings).

**Argo profiles.** Daily profile files for the four 9-day ensemble windows are read from `data/argo/`. `src/inference/run_argo_det_comparison.py --download_argo` downloads missing files from the NOAA NCEI Argo mirror.

## Reproducing the paper

`REPRODUCE.md` lists, for each table and figure, the commands that produce it and the output files. Training commands are also listed there. Training the nine paper models takes about 13–41 GPU-hours each on an RTX A6000.

## Citation

If you use this code or the trained models, please cite the paper and the Zenodo archive of this repository (see `CITATION.cff`).

## Funding

This work was supported by the Ministry of Earth Sciences (MoES), Government of India, through grant MoES/36/OOIS/Extra/84/2022.

## Contact

Deepak N. Subramani, Department of Computational and Data Sciences, Indian Institute of Science, Bengaluru 560012, India (deepakns@iisc.ac.in).
