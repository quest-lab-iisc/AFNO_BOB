# Reproducing the paper

All commands run from the repository root, with the environment from `environment.yml` and the data laid out as in `README.md`. Figure and table numbers follow the PLOS Climate version of the manuscript. Outputs go to `results/<model name>/`, `results/tables/` and `results/figures/`. Each script's module docstring documents all of its options.

Set this once:

```bash
EXTRA="--extra_data_dir data/2021_2025 --extra_num_days 1711 --combined_output"
```

## 1. Training (optional; trained weights are in the models archive on Dropbox, see README)

| Model | Command |
|---|---|
| AFNO 1T (E13) | `python src/training/scripts/train_afno.py --config_file afno_bob_surf_e13.yaml` |
| AFNO RT (E14) | `python src/training/scripts/train_afno_multistep.py --config_file afno_bob_surf_e14.yaml --opt.epochs 100`, then resume from the best checkpoint for 150 more epochs: `python src/training/scripts/train_afno_multistep.py --config_file afno_bob_surf_e14.yaml --results.resume_checkpoint results/models/checkpoint_AFNO_BoB_Surf_E14.pth --opt.epochs 150` |
| FNO 1T / RT | `python src/training/scripts/train_FNO.py --config_file fno_bob_surf_e03.yaml` / `python src/training/scripts/train_FNO_multistep.py --config_file fno_bob_surf_e04.yaml` |
| TFNO 1T / RT | `python src/training/scripts/train_TFNO.py --config_file tfno_bob_surf_e03.yaml` / `python src/training/scripts/train_TFNO_multistep.py --config_file tfno_bob_surf_e04.yaml` |
| U-NO 1T / RT | `python src/training/scripts/train_UNO.py --config_file uno_bob_surf_e03.yaml` / `python src/training/scripts/train_UNO_multistep.py --config_file uno_bob_surf_e04.yaml` |
| ConvLSTM | `python src/training/scripts/train_convlstm.py` |
| Rollout ablation | `python src/training/scripts/train_afno_rollout_ablation.py --rollout_steps R` for R = 1..5 |

Rollout (RT) training unrolls three steps and feeds each prediction back detached from the computational graph. Training writes the best weights to `results/models/<name>.pth`, the full checkpoint to `results/models/checkpoint_<name>.pth`, and the loss history to `results/experiments/logs/<name>_train_val_losses.csv`. Training uses seed 42; GPU non-determinism can still cause small differences from the released weights.

## 2. Deterministic evaluation, 2020–2025 (Tables 3–4, Figs 3–5)

```bash
python src/inference/run_metrics.py --name AFNO_BoB_Surf_E13 --config_file afno_bob_surf_e13.yaml --model_path results/models/AFNO_BoB_Surf_E13.pth $EXTRA
python src/inference/run_metrics.py --name AFNO_BoB_Surf_E14 --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth $EXTRA
for e in E03 E04; do
  python src/inference/run_metrics_fno.py  --config_file fno_bob_surf_${e,,}.yaml  --name FNO_BoB_Surf_$e  --model_path results/models/FNO_BoB_Surf_$e.pth  $EXTRA
  python src/inference/run_metrics_tfno.py --config_file tfno_bob_surf_${e,,}.yaml --name TFNO_BoB_Surf_$e --model_path results/models/TFNO_BoB_Surf_$e.pth --device cuda:0 $EXTRA
  python src/inference/run_metrics_uno.py  --config_file uno_bob_surf_${e,,}.yaml  --name UNO_BoB_Surf_$e  --model_path results/models/UNO_BoB_Surf_$e.pth  --device cuda:0 $EXTRA
done
python src/inference/run_metrics_convlstm.py --name ConvLSTM_BoB_Surf_E01 --extra_data_dir data/2021_2025 --extra_num_days 1711 --combined_output
python src/inference/run_persistence_baseline.py --extra_data_dir data/2021_2025 --extra_num_days 1711 --combined_output
```

Each run writes `results/<name>/forecast_metrics_combined.txt`, which holds RMSE and Pearson correlation for each variable at lead days 1–9, averaged over the 2060 forecasts (357 starting in 2020 and 1703 in 2021–2025). With `--extra_num_days 1711`, the evaluation scripts use the 1703 start dates whose nine-day forecasts end within the 2021–2025 data.

**Table 3** (lead-averaged comparison):
```bash
python src/inference/run_model_comparison_table.py --style transposed --output results/tables/model_comparison_table.tex --models \
  "Persistence:results/Persistence_Baseline" "ConvLSTM:results/ConvLSTM_BoB_Surf_E01" \
  "FNO 1T:results/FNO_BoB_Surf_E03" "FNO RT:results/FNO_BoB_Surf_E04" \
  "UNO 1T:results/UNO_BoB_Surf_E03" "UNO RT:results/UNO_BoB_Surf_E04" \
  "TFNO 1T:results/TFNO_BoB_Surf_E03" "TFNO RT:results/TFNO_BoB_Surf_E04" \
  "AFNO 1T:results/AFNO_BoB_Surf_E13" "AFNO RT:results/AFNO_BoB_Surf_E14"
```

**Fig 3** (skill against lead day): `python src/visualization/plot_model_comparison_trends.py --models` with the same model list minus ConvLSTM, followed by `--leads 1 2 3 4 5 6 7 8 9 --output results/figures/fig_model_comparison_trends --dpi 300`.

**Fig 4** (AFNO RT error distributions):
```bash
python src/analysis/compute_error_histograms.py --model_type afno --model_name AFNO_BoB_Surf_E14 --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth --extra_data_dir data/2021_2025 --extra_num_days 1703 --pdf
```

**Fig 5** (per-forecast RMSE distributions): run `src/analysis/compute_rmse_histograms.py --model_type {afno,fno,tfno,uno} --model_name <name> --config_file <config> --model_path results/models/<name>.pth --extra_data_dir data/2021_2025 --extra_num_days 1703 --pdf` for each of the eight operators. The histogram scripts take every start date in the window, so `--extra_num_days 1703` gives the same 2060 forecasts as the tables. Then run:
```bash
python src/analysis/plot_model_cdf_comparison.py --output_fig results/figures/fig_model_cdf_comparison --pdf --models \
  "AFNO 1T:results/AFNO_BoB_Surf_E13" "AFNO RT:results/AFNO_BoB_Surf_E14" "FNO 1T:results/FNO_BoB_Surf_E03" "FNO RT:results/FNO_BoB_Surf_E04" \
  "TFNO 1T:results/TFNO_BoB_Surf_E03" "TFNO RT:results/TFNO_BoB_Surf_E04" "UNO 1T:results/UNO_BoB_Surf_E03" "UNO RT:results/UNO_BoB_Surf_E04"
```

**Table 4** (seasonal skill of AFNO RT):
```bash
python src/inference/run_seasonal_metrics_table.py --config_file afno_bob_surf_e14.yaml --name AFNO_BoB_Surf_E14 \
  --model_path results/models/AFNO_BoB_Surf_E14.pth --extra_data_dir data/2021_2025 --extra_num_days 1711
```

## 3. Seasonal maps (Figs 6–11)

**Figs 6–7** (one-day forecasts, 15 Jan/Apr/Jul/Oct 2020):
```bash
python src/visualization/plot_seasonal_forecast.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth \
  --dates "15-01-2020" "15-04-2020" "15-07-2020" "15-10-2020" --season_labels "Winter" "Pre-monsoon" "Monsoon" "Post-monsoon" \
  --output results/figures/fig_seasonal_sst_sss
python src/visualization/plot_seasonal_uvssh.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth \
  --dates "15-01-2020" "15-04-2020" "15-07-2020" "15-10-2020" --season_labels "Winter" "Pre-monsoon" "Monsoon" "Post-monsoon" \
  --output results/figures/fig_seasonal_ssh_uv
```

**Figs 8–11** (nine-day rollouts). The start dates are the best of each season, found with `src/inference/find_best_ic_dates.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth --extra_data_dir data/2021_2025`. That script writes `results/AFNO_BoB_Surf_E14/best_ic_dates.csv`.
```bash
for d in 24-02-2023 22-04-2021 14-06-2022 04-10-2020; do
  python src/visualization/plot_forecast_evolution.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth \
    --init_date $d --lead_days 3 5 7 9 --style error_bg --extra_data_dir data/2021_2025 --output results/figures/fig_rollout_$d --pdf
done
```

## 4. Sensitivity to grid and hyperparameters (Tables 5–6)

These tables come from the single-step AFNO configuration evaluated with the v1.0 code (tag `v1.0` of this repository, https://doi.org/10.5281/zenodo.15630879). The rollout-length ablation in the Preprocessing and training section uses `train_afno_rollout_ablation.py` (step 1). Its per-lead results are in `results/AFNO_Ablation_rs{1..5}/forecast_metrics.txt`, summarized in `results/rollout_ablation_findings.md`.

## 5. Ensemble case studies (Figs 12–19)

```bash
# Seasonal EOFs of 1993-2018 GLORYS fields (also provided in the ancillary-data archive on Dropbox)
python src/inference/compute_ocean_eofs.py --data_dir data/1993_2020 --mean_dir data/1993_2020/mean --output_dir data/eof_perturbations

# TIGGE forcing (see README), then 50-member ensembles for each start date
for m in Jan:14-01-2020 Apr:14-04-2020 Jun:14-06-2020 Oct:14-10-2020; do
  python src/inference/run_ensemble_inference_ic.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth \
    --input_date ${m#*:} --eof_dir data/eof_perturbations --n_members 50 --tigge_dir data/tigge_ensemble --month ${m%%:*}
done
```

The ensemble members are drawn with random seed 42.

**Figs 12–15** (ensemble distributions against Argo):
```bash
E=results/AFNO_BoB_Surf_E14
python src/visualization/plot_ensemble_pdf_validation.py \
  --jan_dir $E/ensemble_ic_Jan_14012020 --apr_dir $E/ensemble_ic_Apr_14042020 --jun_dir $E/ensemble_ic_Jun_14062020 --oct_dir $E/ensemble_ic_Oct_14102020 \
  --argo_jan data/argo/jan2020 --argo_apr data/argo/apr2020 --argo_jun data/argo/jun2020 --argo_oct data/argo/oct2020 \
  --mean_dir data/1993_2020/mean --ocean_file data/1993_2020/ocean.nc --pdf
```

**Figs 16–19** (ensemble time series against GLORYS and OSTIA):
```bash
python src/visualization/plot_ensemble_glorys_timeseries.py \
  --jan_dir $E/ensemble_ic_Jan_14012020 --apr_dir $E/ensemble_ic_Apr_14042020 --jun_dir $E/ensemble_ic_Jun_14062020 --oct_dir $E/ensemble_ic_Oct_14102020 \
  --mean_dir data/1993_2020/mean --ocean_file data/1993_2020/ocean.nc --ostia_file data/1993_2020/ostia_2020.nc --n_locs 6 --seed 42 --pdf
```

## 6. Physical consistency (Fig 20)

```bash
python src/visualization/plot_conservation_laws.py --config_file afno_bob_surf_e14.yaml --model_path results/models/AFNO_BoB_Surf_E14.pth \
  --name AFNO_BoB_Surf_E14 --output results/figures/fig_physical_consistency
```

The script uses twelve 2020 start dates (the 15th of each month) and the 1 July 2020 forecast for the front map.

## Figures without code

Fig 1 (study area) and Fig 2 (architecture schematic) are illustrations. Tables 1–2 (datasets, model configurations) are compiled from the configs and the training logs in `results/experiments/logs/`.
