# Solar Plant Intelligence Dashboard

A complete AI4003 Applied Machine Learning project for solar AC-power prediction using manually implemented linear regression with NumPy.

## Model

The dashboard uses **Set B (Public Weather)** for live prediction:

- `sw_radiation` — Shortwave radiation (W/m²)
- `temp_2m` — 2 m air temperature (°C)
- `cloud_cover` — Cloud cover (%)
- `sin_hour`, `cos_hour` — cyclic hour features
- intercept

The three continuous variables are standardized using the **training-set mean and standard deviation saved with the trained model**. The prediction is `max(0, X @ theta)` and is displayed in kW.

## Training results with the supplied data

| Solver | Features | All-hours RMSE (kW) | Daytime RMSE (kW) |
|---|---:|---:|---:|
| Normal equation | Set A | 544.223 | 710.134 |
| Batch GD (alpha=0.0001, iters=20000) | Set A | 544.337 | 710.278 |
| Stochastic GD (alpha=0.01, epochs=50) | Set A | 561.935 | 734.627 |
| Normal equation | Set B | 3167.816 | 3989.990 |
| Batch GD (alpha=0.0001, iters=20000) | Set B | 3167.816 | 3989.990 |
| Stochastic GD (alpha=0.01, epochs=50) | Set B | 2985.071 | 3829.018 |

Set A Normal Equation theta:

| feature | theta |
|---|---:|
| intercept | 6803.188733 |
| irradiation | 8187.254344 |
| module_temp | 5.329879 |
| ambient_temp | -65.130812 |
| sin_hour | -86.115151 |
| cos_hour | -606.321145 |

The supplied training run produces the requested values exactly (to the displayed precision). Set B Batch GD and Normal Equation differ by approximately `1.02e-10` in maximum absolute theta difference.

## Data files

Place the supplied CSV files in `data/`. The included final package already contains the prepared/model artifacts generated from the supplied files.

## Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Retrain

From the project root:

```bash
python src/prepare.py
python src/fetch_weather.py
python src/train_eval.py
```

Or use **Model Training** inside the Streamlit dashboard.

## Important implementation detail

`plant1_hourly_openmeteo.csv` may already contain `ac_power` and on-site columns. The Set B loader deliberately selects only the three public-weather predictors before merging the plant target/irradiation. This prevents duplicate `ac_power_x/ac_power_y` columns and ensures AC power remains the target, not an input feature.
