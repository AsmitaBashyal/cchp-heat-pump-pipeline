"""
Stage 1: Data preparation and feature engineering.

Reads the raw simulated CCHP sensor data (pipeline-development data only --
see proposal Section 3.2 / Appendix A; NOT real field or lab data) and
produces a cleaned, feature-engineered dataset used by the rest of the
pipeline (regime detection, predictive modelling, SHAP, conformal prediction).

Ground-truth columns in Truth_Reference_DO_NOT_USE are used ONLY to validate
pipeline outputs (e.g. checking whether HDBSCAN recovers the known regime
structure) -- they are never used as model input features, per the raw
workbook's own README warning.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
RAW_PATH = PROJECT_DIR / "Raw_Simulated_Data_v2.xlsx"
OUT_DIR = PROJECT_DIR / "Outputs"
OUT_DIR.mkdir(exist_ok=True)

SENSOR_COLS = [
    "high_side_pressure_psig",
    "suction_pressure_psig",
    "liquid_line_temp_c",
    "suction_line_temp_c",
    "outdoor_temp_c",
    "relative_humidity_pct",
    "return_air_temp_c",
    "supply_air_temp_c",
    "top_of_outdoor_unit_temp_c",
    "electrical_power_w",
    "supplemental_heat_power_w",
    "compressor_frequency_hz",
    "eev_position_pct",
    "airflow_cfm",
]

# Air-side heat balance constants (standard air, SI units)
AIR_DENSITY_KG_M3 = 1.225
AIR_CP_J_PER_KGK = 1006.0
CFM_TO_M3S = 0.00047194745


def load_raw():
    obs = pd.read_excel(RAW_PATH, sheet_name="Combined_Observed")
    truth = pd.read_excel(RAW_PATH, sheet_name="Truth_Reference_DO_NOT_USE")
    obs["timestamp"] = pd.to_datetime(obs["timestamp"])
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    return obs, truth


def clean(obs: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing sensor values, gap-flagged rather than imputed
    (Section 3.3). Sort by time and keep the flag/refrigerant columns."""
    before = len(obs)
    df = obs.dropna(subset=SENSOR_COLS).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} rows with missing sensor values "
          f"({dropped/before:.2%} of {before}).")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Physics-derived features (Section 3.3, 3.6) ---
    df["pressure_ratio"] = df["high_side_pressure_psig"] / df["suction_pressure_psig"]

    # Approximate superheat / subcooling proxies (no refrigerant-property
    # lookup available offline -- these are simple sensor-temperature-based
    # proxies for pipeline development, not thermodynamically exact values).
    df["suction_superheat_proxy_c"] = df["suction_line_temp_c"] - df["outdoor_temp_c"]
    df["liquid_subcool_proxy_c"] = df["liquid_line_temp_c"] - df["return_air_temp_c"]

    df["total_electrical_power_w"] = (
        df["electrical_power_w"] + df["supplemental_heat_power_w"]
    )

    # --- Outdoor temperature bins ---
    bins = [-40, -20, -15, -10, -5, 0, 5, 100]
    labels = ["<-20", "-20to-15", "-15to-10", "-10to-5", "-5to0", "0to5", ">5"]
    df["outdoor_temp_bin"] = pd.cut(df["outdoor_temp_c"], bins=bins, labels=labels)

    # --- Rolling features (5-min sampling -> 6 steps = 30 min window) ---
    roll_cols = ["outdoor_temp_c", "electrical_power_w", "high_side_pressure_psig"]
    for c in roll_cols:
        df[f"{c}_roll30min_mean"] = df[c].rolling(window=6, min_periods=1).mean()
    df["defrost_flag_roll30min_frac"] = (
        df["defrost_flag"].astype(int).rolling(window=6, min_periods=1).mean()
    )

    # --- Air-side heat balance: delivered heat, measured-style COP (Sec 3.5) ---
    airflow_m3s = df["airflow_cfm"] * CFM_TO_M3S
    delta_t = df["supply_air_temp_c"] - df["return_air_temp_c"]
    df["delivered_heat_w"] = airflow_m3s * AIR_DENSITY_KG_M3 * AIR_CP_J_PER_KGK * delta_t

    with np.errstate(divide="ignore", invalid="ignore"):
        cop = df["delivered_heat_w"] / df["total_electrical_power_w"]
    # Physical bound: COP must be positive and finite; guard against the
    # total-electrical-power denominator being non-positive.
    cop = cop.where(df["total_electrical_power_w"] > 0)
    df["measured_cop"] = cop.clip(lower=0, upper=8)

    return df


def attach_truth_for_validation(df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Left-join truth columns for VALIDATION ONLY. Callers must not use
    these as model input features."""
    keep = ["timestamp", "_truth_proxy_cop", "_truth_regime",
            "_truth_delivered_capacity_w", "_truth_compressor_speed_frac"]
    return df.merge(truth[keep], on="timestamp", how="left")


def main():
    obs, truth = load_raw()
    df = clean(obs)
    df = engineer_features(df)
    df = attach_truth_for_validation(df, truth)

    print(df[["timestamp", "measured_cop", "_truth_proxy_cop", "_truth_regime"]].head(10))
    print("\nmeasured_cop describe:\n", df["measured_cop"].describe())
    print("\nCorrelation(measured_cop, truth_proxy_cop) on steady-state rows:")
    ss = df[df["_truth_regime"] == "steady_state_heating"]
    print(ss["measured_cop"].corr(ss["_truth_proxy_cop"]))

    out_path = OUT_DIR / "prepared_data.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved prepared dataset: {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
