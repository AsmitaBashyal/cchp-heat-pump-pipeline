# CCHP Digital Twin – UI prototype (simulated data)

Interactive dashboard on the v3 simulated dataset (R-454B + humidity) and the
pipeline outputs in `Outputs_v3/`. Runs locally; nothing is sent anywhere.

## Run
Double-click `Run_Dashboard.bat`, or from the project folder:

    python -m streamlit run Dashboard/app.py

It opens at http://localhost:8501 (first load ~10 s while the data is cached).
Needs `streamlit` and `plotly` (`python -m pip install streamlit plotly`).

## What it shows
| Sidebar | Site picker; time-period buttons (1 day / 3 days / 1 week / 1 month / Season), Earlier / Later paging, "Jump to model test period", drag-to-select range slider, time resolution (raw 5-min / hourly / daily). |
|---|---|
| Header | Window KPIs: mean outdoor temperature and RH, system COP, kWh used, defrost cycles, data coverage. |
| Sensor readings | Any sensors or physics-derived channels as stacked panels on one time axis, defrost cycles shaded, summary stats, CSV export; regime timeline (simulator truth vs HDBSCAN clusters). |
| Sensor relationships | Scatter of any two channels (one-click physics presets), coloured by regime / site / cluster / RH / temperature, Pearson r, correlation matrix. |
| Model accuracy | XGBoost or graph-attention COP vs measured (or simulator-true) COP with 90 % conformal band; RMSE / MAE / R² / MAPE / coverage for the selected window; parity plot; RMSE by regime; residuals vs a chosen sensor; LOSO table and SHAP importance. |

Predictions exist only for the held-out test period (last ~15 % of each site's
season, from 27 Feb 2027). Missing 5-min intervals are shown as gaps, never filled.

## Moving to real sensor data
Point `DATA_DIR` in `app.py` at a folder with the same files/columns
(`with_regimes.parquet`, `test_predictions.parquet`, `conformal_predictions.parquet`,
`loso_results.csv`, `shap_global_importance.csv`).
