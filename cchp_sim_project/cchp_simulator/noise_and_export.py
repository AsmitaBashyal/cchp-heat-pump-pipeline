"""
noise_and_export.py

Adds instrument-realistic noise (per Table 6.1 accuracy specs) and occasional
dropped intervals, then exports per-sensor files in a structure compatible
with the existing lab-data ingestion pipeline, plus one combined "truth-free"
table (no _truth_* columns) that is what your ML pipeline should actually
train on -- exactly like a real deployment, where you don't have ground-truth
regime labels or true COP handed to you.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# (mean=0, std) noise levels drawn from Table 6.1 sensor accuracy specs
NOISE_STD = {
    "high_side_pressure_psig": 0.5,      # ~0.5% FS-ish, simplified to a flat psig std
    "suction_pressure_psig": 0.3,
    "liquid_line_temp_c": 0.5,
    "suction_line_temp_c": 0.5,
    "outdoor_temp_c": 0.3,
    "return_air_temp_c": 0.3,
    "supply_air_temp_c": 0.3,
    "top_of_outdoor_unit_temp_c": 1.0,
    "electrical_power_w": 15.0,          # ~2-5% of reading, applied as flat + scaled below
}


def add_sensor_noise(df: pd.DataFrame, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col, std in NOISE_STD.items():
        if col in out.columns:
            out[col] = out[col] + rng.normal(0, std, size=len(out))
    if "electrical_power_w" in out.columns:
        out["electrical_power_w"] = out["electrical_power_w"] * (1 + rng.normal(0, 0.02, size=len(out)))
    return out


def inject_dropouts(df: pd.DataFrame, drop_prob=0.003, seed=1) -> pd.DataFrame:
    """Randomly null out short stretches to emulate wifi/connectivity gaps
    (Sec 5.7/6.13) -- your pipeline should flag these, not silently interpolate.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    mask = rng.random(n) < drop_prob
    gap_len = rng.integers(1, 6, size=n)  # 1-5 missed 5-min intervals
    sensor_cols = [c for c in out.columns if not c.startswith("_truth_") and c not in
                   ("defrost_flag", "refrigerant_type")]
    idx_positions = np.where(mask)[0]
    for pos in idx_positions:
        end = min(pos + gap_len[pos], n)
        out.iloc[pos:end, out.columns.get_indexer(sensor_cols)] = np.nan
    return out


def split_truth_and_observed(df: pd.DataFrame):
    truth_cols = [c for c in df.columns if c.startswith("_truth_")]
    observed_cols = [c for c in df.columns if not c.startswith("_truth_")]
    return df[observed_cols].copy(), df[truth_cols].copy()


def export_per_sensor_csvs(observed_df: pd.DataFrame, out_dir: str, run_tag: str):
    """Writes one CSV per sensor stream, mirroring the existing per-sensor file
    layout (e.g. 'High side pressure.xlsx' style), but tagged SIMULATED and in
    CSV form for easy pipeline consumption. Also writes one combined file.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    sensor_file_map = {
        "high_side_pressure_psig": "high_side_pressure",
        "suction_pressure_psig": "suction_pressure",
        "liquid_line_temp_c": "liquid_temperature",
        "suction_line_temp_c": "suction_temperature",
        "outdoor_temp_c": "outdoor_temperature",
        "return_air_temp_c": "return_air_furnace",
        "supply_air_temp_c": "supply_air_furnace",
        "top_of_outdoor_unit_temp_c": "top_of_outdoor_unit",
        "electrical_power_w": "electrical_power",
        "supplemental_heat_power_w": "supplemental_heat_power",
        "compressor_frequency_hz": "compressor_frequency",
        "eev_position_pct": "eev_position",
        "airflow_cfm": "airflow",
        "defrost_flag": "defrost_flag",
        "refrigerant_type": "refrigerant_type",
    }

    for col, fname in sensor_file_map.items():
        if col in observed_df.columns:
            sub = observed_df[[col]].reset_index()
            sub["data_quality_flag"] = np.where(sub[col].isna(), "MISSING", "OK")
            sub.to_csv(f"{out_dir}/SIMULATED_{run_tag}_{fname}.csv", index=False)

    combined = observed_df.reset_index()
    combined.to_csv(f"{out_dir}/SIMULATED_{run_tag}_combined.csv", index=False)
