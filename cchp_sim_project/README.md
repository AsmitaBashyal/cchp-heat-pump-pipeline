# CCHP Interim Data Simulator — Starter Scaffold

**Status: v1 development scaffold, tested end-to-end.** This generates a
SIMULATED cold-climate heat pump dataset (8 confirmed sensor streams + 5
instrumentation-gap streams from Table 5.1 of the proposal) so you can build
and test your full pipeline — ingestion, HDBSCAN regime discovery, baseline
models, the physics-informed graph transformer, SHAP, conformal prediction —
before the TRU lab/field data are ready.

**This is not field data and must never be presented as such.** Every output
file is prefixed `SIMULATED_`. A parallel `_TRUTH_DO_NOT_TRAIN_ON.csv` file
carries the ground-truth regime label, true proxy COP, etc. — useful for
validating whether HDBSCAN recovers the regimes actually built into the
simulation, but it must never be fed into a model as a feature or used as
if it were a measured target.

## Quick start

```bash
pip install CoolProp pandas numpy --break-system-packages   # if not already installed

# Fastest way to get something running today (synthetic Kamloops-normals weather):
python run_simulation.py --weather synthetic --days 150 --refrigerant R410A --out data/raw --run-tag v1

# Recommended within the next few days: swap in REAL weather.
# 1. Go to https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
# 2. Search "Kamloops", pick a station (e.g. Kamloops A), download hourly/daily CSV
#    for your target date range (a past winter, e.g. Dec 2023 - Mar 2024)
# 3. Run:
python run_simulation.py --weather real --weather-csv path/to/kamloops.csv --out data/raw --run-tag v2
```

Output: one CSV per sensor stream (`SIMULATED_<tag>_<sensor>.csv`) plus a
combined table (`SIMULATED_<tag>_combined.csv`), all on the project's 5-minute
grid, with realistic sensor noise and randomly injected dropout gaps
(flagged via `data_quality_flag`, not silently filled — consistent with the
proposal's §5.7 QA principle).

## What's implemented (v1)

- `cchp_simulator/weather.py` — real-weather CSV loader + a clearly-labeled
  synthetic fallback based on Kamloops 1991–2020 climate normals.
- `cchp_simulator/refrigerant.py` — CoolProp wrapper for R-410A (predefined)
  and R-454B (approximated as an R-32/R-1234yf HEOS mixture — flag this
  approximation explicitly if you ever cite R-454B results).
- `cchp_simulator/cycle_model.py` — steady-flow energy balance cycle model:
  capacity derating vs. outdoor temp, bivalent-temperature supplemental-heat
  logic, superheat/subcooling setpoints, defrost state machine, and the 5
  instrumentation-gap variables (compressor frequency, EEV position, airflow,
  defrost flag, refrigerant type) so you can build the full graph-transformer
  architecture now.
- `cchp_simulator/noise_and_export.py` — sensor noise (per Table 6.1 accuracy
  specs), dropout injection, per-sensor CSV export matching your existing
  file layout, and a truth/observed split.

## What's simplified (be upfront about these if this ever appears in a report)

- Single-zone building model (UA-based, no thermal mass/dynamics).
- Fixed compressor isentropic efficiency, not a manufacturer compressor map.
- Capacity-derate curve is a representative shape, not a specific unit's
  published performance data.
- Superheat/subcooling are control setpoints, not solved from a full
  heat-exchanger model.
- R-454B properties are an approximate mixture, not a validated proprietary
  correlation.

## Suggested next steps (see the step-by-step plan in chat for full detail)

1. Run the synthetic version today, confirm the pipeline runs end to end.
2. Swap in real ECCC Kamloops weather within the week.
3. Build the ingestion/synchronization layer (§5.7) reading these CSVs into
   a unified table — reuse this schema unchanged when real field data starts.
4. Run HDBSCAN on the *observed* columns only, then check recovered clusters
   against `_truth_regime` in the truth file — this is a real validation
   exercise you can't do with field data alone.
5. Train the Mitacs-scope baselines (linear regression, RF, XGBoost) +
   tree-based SHAP.
6. Prototype the physics-informed graph transformer using the now-available
   EEV position / compressor frequency node features.
