Cold-Climate Heat Pump: Interim Data Pipeline (Simulated Data)

Interim data pipeline and baseline analysis built while real lab/field data collection for my MSc research project (Thompson Rivers University) is still pending. This repo demonstrates the pipeline — simulation, exploratory analysis, unsupervised regime detection, and a baseline predictive model with explainability — running end-to-end on physics-based simulated data.

This is not the thesis research itself. It is a working, tested prototype of the pipeline that will later run on real cold-climate heat pump sensor data once field/lab deployment begins. See "Status and scope" below for exactly what this is and isn't.

What's in this repo
cchp_simulator/ — physics-based CCHP simulator (steady-flow energy balance, CoolProp thermodynamic properties, defrost-cycle logic) that generates realistic sensor-style data
run_simulation.py — generates the simulated dataset
eda_beginner.py — exploratory data analysis and basic charts
regime_discovery_beginner.py — HDBSCAN clustering to detect operating regimes (steady-state heating vs. defrost), validated against the simulator's known ground truth
shap_beginner.py — XGBoost baseline model predicting electrical power draw, with SHAP feature-importance analysis
Status and scope
Data: 100% simulated, generated from documented thermodynamic relationships — not real sensor or field measurements. No lab or field data has been collected yet for this project.
Models: standard, non-physics-informed baselines (XGBoost). Not the physics-informed / graph-based models planned for the full thesis.
Purpose: validate that the pipeline (ingestion → EDA → regime detection → modeling → explainability) works correctly end-to-end before real data is available.
Authorship note

I am a data science student with no prior coding background before this project. To be specific about how this repo was built:

Written by AI (Claude): all the Python code in this repo — the simulator physics, the analysis scripts, and this README.
Done by me: installing the required tools (Python, VS Code), running every script, reading and interpreting the output, deciding what to try next, catching and reporting errors, and validating that the results made sense (e.g., checking the HDBSCAN clusters against known ground truth).

I'm including this note deliberately rather than presenting the code as self-written, since accurate attribution matters to me both academically and professionally.

Requirements
pip install CoolProp pandas numpy matplotlib hdbscan scikit-learn xgboost shap openpyxl
Running it
python run_simulation.py --run-tag v1
python eda_beginner.py
python regime_discovery_beginner.py
python shap_beginner.py

Charts are saved to plots/.

Next steps
Integrate real historical weather data (Environment and Climate Change Canada) in place of the synthetic weather profile
Re-target modeling to proxy COP rather than raw electrical power
Replace simulated data with real lab/field measurements as they become available
Extend to the physics-informed and regime-aware modeling framework described in the full thesis proposal.
