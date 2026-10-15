# Cold-Climate Heat Pump: Interim Data Pipeline Using Simulated Data

This repository contains an initial data pipeline for my MSc research project at Thompson Rivers University. I developed this prototype while waiting for real data from the laboratory and field sensors to become available.

The project demonstrates the main steps of a data science pipeline, including data simulation, exploratory data analysis, operating regime detection, and a baseline prediction model with explainability. The pipeline uses simulated data based on thermodynamic relationships.

**This is not the final thesis research.** It is a working prototype that I can test and improve before using real sensor data from a cold-climate heat pump. The *Status and Scope* section explains the current limitations of this work.

## What's in this repository

- `cchp_simulator/`: A physics-based heat pump simulator that uses energy balance equations, CoolProp thermodynamic properties, and defrost cycle logic to generate sensor-like data.
- `run_simulation.py`: Generates the simulated dataset.
- `eda_beginner.py`: Explores the data and creates basic charts.
- `regime_discovery_beginner.py`: Uses HDBSCAN clustering to identify different operating conditions, such as normal heating and defrost cycles. The results are compared with the known operating conditions from the simulator.
- `shap_beginner.py`: Uses an XGBoost model to predict electrical power consumption and SHAP to understand which features are important for the predictions.

## Status and scope

**Data:** All data in this repository is simulated. It is generated using documented thermodynamic relationships and does not come from real sensor readings or field measurements. Real laboratory and field data collection has not started yet.

**Models:** The current prediction model uses XGBoost, which is a standard machine learning method. It is not a physics-informed or graph-based model. These more advanced methods are planned for the full thesis research.

**Purpose:** The main goal is to check whether the complete pipeline works correctly, from data generation and exploration to operating regime detection, prediction, and explainability. This will help prepare the pipeline for real data when it becomes available.

## Authorship and AI use

I am a data science student, and I did not have a coding background before starting this project. I want to be clear about how I developed this repository.

Claude AI wrote all the Python code, including the simulator, analysis scripts, and this README file.

My contribution was setting up the required tools, including Python and VS Code, running each script, studying the outputs, and deciding what to test next. I also identified and reported errors and checked whether the results made sense. For example, I compared the HDBSCAN clustering results with the operating conditions already known from the simulator.

I have included this explanation because I believe it is important to be honest about my work and the role of AI in developing it, both academically and professionally.

## Requirements

Install the required Python packages using:

```bash
pip install CoolProp pandas numpy matplotlib hdbscan scikit-learn xgboost shap openpyxl
```

## How to run

Run the following commands in order:

```bash
python run_simulation.py --run-tag v1
python eda_beginner.py
python regime_discovery_beginner.py
python shap_beginner.py
```

The generated charts are saved in the `plots/` directory.

## Next steps

1. Add historical weather data from Environment and Climate Change Canada instead of using the current simulated weather profile.
2. Change the prediction target from electrical power consumption to an estimated coefficient of performance (COP).
3. Replace simulated data with real laboratory and field sensor measurements when they become available.
4. Develop the physics-informed and operating-regime-aware models proposed for the full thesis research.
