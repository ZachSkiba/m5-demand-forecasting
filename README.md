# M5 Demand Forecasting and Inventory Planning

An end-to-end demand forecasting and inventory-planning framework built on the [M5 retail dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (30,490 SKUs). Instead of fitting one model to everything, the project classifies each SKU by its demand pattern and routes it to a forecasting method suited to that pattern. Forecasts and demand uncertainty then feed a service-level inventory policy, and everything is evaluated on future periods in a time-series backtest.

**Live app:** <!-- TODO: replace with your Streamlit URL --> `https://YOUR-APP-NAME.streamlit.app`

<!-- Optional: add a screenshot -->
<!-- ![App overview](docs/overview.png) -->

## The approach

```text
Historical demand
      ↓
Demand-pattern features (ADI, CV²)
      ↓
Regime: Smooth / Erratic / Intermittent / Lumpy
      ↓
Regime-specific forecast
      ↓
Demand uncertainty and service-level planning
      ↓
Inventory policy
      ↓
Future-period evaluation
```

### Demand regimes

Each SKU is classified on training data using two standard measures:

- **ADI** (average demand interval): how often non-zero demand occurs.
- **CV²** (squared coefficient of variation of non-zero demand): how variable demand size is when demand occurs.

The final Fold 3 thresholds are **ADI = 1.32** and **CV² = 0.49**.

| Regime | Pattern | SKUs | Method |
|---|---|---:|---|
| Smooth | Frequent demand, stable size | 8,111 | LightGBM Tweedie |
| Erratic | Frequent demand, highly variable size | 906 | LightGBM Tweedie |
| Intermittent | Infrequent demand, stable size | 17,303 | TSB (α = 0.5, β = 0.5) |
| Lumpy | Infrequent demand, highly variable size | 4,170 | Historical policy |

- **Smooth / Erratic:** LightGBM with a Tweedie objective, stored as daily forecast trajectories.
- **Intermittent:** Teunter-Syntetos-Babai (TSB), a causal weekly walk-forward forecast that models demand occurrence and demand size separately. Parameters were selected on earlier validation and locked for Fold 3.
- **Lumpy:** a historical planning policy, not a predictive model. It appears in the app as a planning proxy, and its evaluation is labeled a policy-level comparison, not model accuracy.

### Uncertainty

Intermittent uncertainty comes from a frozen moving-block bootstrap of 7-day lead-time demand (quantiles q50 to q99). These are demand quantiles, not confidence intervals. Lumpy has no stored uncertainty, and none is fabricated.

### Inventory planning

Inventory planning is a separate frozen layer. It converts demand uncertainty and a service-level scenario (q80, q90, q95, q99) into lead-time demand, safety buffer, reorder point and order quantity, using a 7-day lead time and a 1-week review period. q80 is the validated primary scenario; the other levels are scenario analyses.

### Validation

Evaluation is chronological, never a random split: the past is used for training and model selection, the future for evaluation. Methods and parameters were chosen on earlier folds. **Fold 3 is the final locked evaluation** and was not retuned after seeing its results. The full methodology is in [`notebooks/07_fold3_final_evaluation.ipynb`](notebooks/07_fold3_final_evaluation.ipynb).

## About the app

The Streamlit app is a presentation layer over **frozen artifacts**. It does not retrain models, run model inference, generate replacement forecasts, simulate inventory, or load the raw M5 files.

- **Portfolio overview:** weekly aggregate actual demand vs forecast, grouped by regime, state, store, department or category.
- **Product explorer:** search any of the 30,490 SKUs and see historical demand, the evaluation-period actuals, the regime's forecast or policy level, and its demand uncertainty where one exists.
- **Evaluation:** MAE, WAPE and Bias computed over the selected display window (7, 28, 90 or 365 days). Smooth and Erratic use daily values; Intermittent and Lumpy use complete 7-day periods anchored at the 2015-02-01 forecast origin.
- **Inventory planning:** frozen reorder point, order quantity and safety buffer by service level.

The data is a historical snapshot covering February 2015 to January 2016.

## Run locally

```bash
git clone https://github.com/ZachSkiba/m5-demand-forecasting.git
cd m5-demand-forecasting
python -m venv .venv
.venv\Scripts\activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`data/processed/predictions/app/app_forecasts.parquet` is stored with [Git LFS](https://git-lfs.com/). Install Git LFS before cloning, or run `git lfs pull` afterwards.

## Repository layout

```text
streamlit_app.py                     Streamlit app (single file)
requirements.txt
.streamlit/config.toml               Theme
notebooks/
  07_fold3_final_evaluation.ipynb    Fold 3 final evaluation
data/processed/predictions/app/      Frozen app artifacts (Parquet)
  app_sku_metadata.parquet           Catalog and regime for all 30,490 SKUs
  app_historical_demand.parquet      Historical context, 2014-02-01 to 2015-01-31
  app_forecasts.parquet              Daily LightGBM trajectories (Git LFS)
  app_actuals.parquet                Daily actuals for those SKUs
  app_regime_trajectories.parquet    Weekly TSB / policy trajectories (Intermittent, Lumpy)
  app_regime_actuals.parquet         Weekly actuals for Intermittent and Lumpy
  app_inventory_policy.parquet       Frozen inventory policy by service level
```

## Data

Sales data comes from the M5 Forecasting competition (Walmart retail sales), available on [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy). The raw files are not included in this repository.

## Author

Zach Skiba