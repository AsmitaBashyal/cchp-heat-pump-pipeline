"""
Stage 0: Update the simulated raw dataset to R-454B and add a relative
humidity sensor channel, producing Raw_Simulated_Data_v2.xlsx.

Refrigerant swap (R-410A -> R-454B):
  R-454B is not a pure fluid in CoolProp's default library, so it is
  represented as its published nominal composition, a near-azeotropic
  HFO/HFC blend of 68.9% R-32 / 31.1% R-1234yf by mass (AHRI/ASHRAE
  designation R-454B). Rather than re-deriving the whole simulator (whose
  internal logic is not available in this project folder -- only its
  output workbook is), this script performs a physically grounded
  refrigerant substitution: it holds the underlying cycle temperatures
  (condensing/evaporating) fixed -- these come from the original R-410A
  simulation and are not refrigerant-specific -- and re-derives the
  refrigerant-side pressures a system running R-454B would show at those
  same temperatures, using each fluid's saturation pressure-temperature
  curve (bubble point). This reproduces the well-documented result that
  R-454B runs at somewhat lower head pressure than R-410A for the same
  operating temperatures (roughly 5-12% lower over typical CCHP operating
  range, verified against CoolProp below). Air-side and electrical sensor
  channels (temperatures, power, airflow, compressor frequency, EEV
  position) are left unchanged, since in this synthetic pipeline-dev
  dataset they are not deterministic functions of refrigerant identity;
  this simplification is documented in the output README sheet.

New sensor: relative_humidity_pct (outdoor), synthesized with a plausible
BC-winter profile (Section 3.2 weather driver note): a slow seasonal/
day-to-day base level, a mild diurnal cycle (higher overnight/early
morning), a weak inverse relationship with outdoor temperature (colder air
snaps tend to run more humid at this site), and autocorrelated noise
(bounded 35-98%) so 5-minute-to-5-minute values are smooth rather than iid.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
# Source copy (the v1 workbook may be open in Excel and locked for reading
# directly from OneDrive); this scratch copy was made before this script ran.
SRC_COPY = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\C--Users-Asus-OneDrive---Thompson-Rivers-University-Desktop-CCHP-Masters-Project-Claude-Code\e80a29f0-b07c-407a-b50b-e9a1f4150c87\scratchpad\Raw_Simulated_Data_v1_copy.xlsx")
OUT_PATH = PROJECT_DIR / "Raw_Simulated_Data_v2.xlsx"

R454B_COOLPROP = "HEOS::R32[0.689]&R1234yf[0.311]"
PSI_TO_PA = 6894.757293168
PA_TO_PSI = 1 / PSI_TO_PA
ATM_PSI = 14.6959

RNG = np.random.default_rng(42)


def build_saturation_curve(fluid, t_min_c=-55, t_max_c=60, step_c=0.5):
    """Bubble-point (Q=0) saturation pressure [Pa] over a temperature grid."""
    t_grid_c = np.arange(t_min_c, t_max_c + step_c, step_c)
    t_grid_k = t_grid_c + 273.15
    p_grid_pa = np.array([PropsSI("P", "T", t, "Q", 0, fluid) for t in t_grid_k])
    return t_grid_c, p_grid_pa


def psig_to_temp_c(psig, t_grid_c, p_grid_pa):
    """Invert a fluid's saturation curve: gauge pressure [psig] -> saturation temp [C]."""
    p_pa = (psig + ATM_PSI) * PSI_TO_PA
    # p_grid_pa is monotonically increasing with t_grid_c
    return np.interp(p_pa, p_grid_pa, t_grid_c)


def temp_c_to_psig(temp_c, t_grid_c, p_grid_pa):
    """Fluid's saturation curve: saturation temp [C] -> gauge pressure [psig]."""
    p_pa = np.interp(temp_c, t_grid_c, p_grid_pa)
    return p_pa * PA_TO_PSI - ATM_PSI


def convert_pressures_r410a_to_r454b(df):
    t_r410a, p_r410a = build_saturation_curve("R410A")
    t_r454b, p_r454b = build_saturation_curve(R454B_COOLPROP)

    out = df.copy()
    for col in ["high_side_pressure_psig", "suction_pressure_psig"]:
        valid = out[col].notna()
        sat_temp_c = psig_to_temp_c(out.loc[valid, col].to_numpy(), t_r410a, p_r410a)
        new_psig = temp_c_to_psig(sat_temp_c, t_r454b, p_r454b)
        out.loc[valid, col] = new_psig
    return out


def synthesize_relative_humidity(df):
    n = len(df)
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_index = (ts - ts.iloc[0]).dt.total_seconds() / 86400.0

    # Slow day-to-day base level drifting between synoptic-system extremes.
    n_days = int(np.ceil(day_index.max())) + 2
    daily_base = 68 + 14 * np.sin(np.linspace(0, 6 * np.pi, n_days) + 0.7) \
        + RNG.normal(0, 4, n_days).cumsum() * 0.15
    daily_base = np.clip(daily_base, 45, 92)
    base_series = np.interp(day_index, np.arange(n_days), daily_base)

    # Mild diurnal cycle: higher overnight/early morning, lower mid-afternoon.
    diurnal = 6 * np.cos((hour - 5) / 24 * 2 * np.pi)

    # Weak inverse relationship with outdoor temperature (colder -> more humid
    # at this site), centered on the dataset's own mean outdoor temperature.
    temp_effect = -0.6 * (df["outdoor_temp_c"] - df["outdoor_temp_c"].mean())

    # Autocorrelated (AR(1)) noise so 5-minute steps are smooth, not iid.
    noise = np.zeros(n)
    phi = 0.97
    sigma = 1.2
    innovations = RNG.normal(0, sigma, n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + innovations[i]

    rh = base_series + diurnal + temp_effect.to_numpy() + noise
    rh = np.clip(rh, 35, 98)
    return np.round(rh, 1)


def main():
    obs = pd.read_excel(SRC_COPY, sheet_name="Combined_Observed")
    truth = pd.read_excel(SRC_COPY, sheet_name="Truth_Reference_DO_NOT_USE")
    readme = pd.read_excel(SRC_COPY, sheet_name="README")

    # --- Sanity check: report the pressure shift at representative temps ---
    t_r410a, p_r410a = build_saturation_curve("R410A")
    t_r454b, p_r454b = build_saturation_curve(R454B_COOLPROP)
    print("Representative saturation pressure shift, R410A -> R454B:")
    for t_c in [-20, -10, 0, 10, 20, 40]:
        p1 = temp_c_to_psig(t_c, t_r410a, p_r410a)
        p2 = temp_c_to_psig(t_c, t_r454b, p_r454b)
        print(f"  {t_c:>4} C sat temp: R410A={p1:7.1f} psig   R454B={p2:7.1f} psig   ratio={p2/p1:.3f}")

    obs2 = convert_pressures_r410a_to_r454b(obs)
    obs2["refrigerant_type"] = "R454B"
    obs2["relative_humidity_pct"] = synthesize_relative_humidity(obs2)

    # Reorder: keep humidity next to the other outdoor-condition sensor.
    cols = list(obs2.columns)
    cols.remove("relative_humidity_pct")
    insert_at = cols.index("outdoor_temp_c") + 1
    cols.insert(insert_at, "relative_humidity_pct")
    obs2 = obs2[cols]

    print(f"\nHigh-side pressure: R410A mean={obs['high_side_pressure_psig'].mean():.1f} psig "
          f"-> R454B mean={obs2['high_side_pressure_psig'].mean():.1f} psig")
    print(f"Suction pressure:   R410A mean={obs['suction_pressure_psig'].mean():.1f} psig "
          f"-> R454B mean={obs2['suction_pressure_psig'].mean():.1f} psig")
    print(f"\nrelative_humidity_pct summary:\n{obs2['relative_humidity_pct'].describe()}")

    readme2 = readme.copy()
    def set_field(field, value):
        readme2.loc[readme2["Field"] == field, "Value"] = value

    set_field("Dataset", "SIMULATED CCHP sensor data, run tag v2 (R-454B refrigerant, "
                          "relative humidity sensor added)")
    set_field("Refrigerant", "R-454B (modelled as 68.9% R-32 / 31.1% R-1234yf by mass, "
                              "per AHRI/ASHRAE nominal composition)")
    new_rows = pd.DataFrame({
        "Field": ["Refrigerant conversion method", "New sensor: relative_humidity_pct",
                  "Derived from"],
        "Value": [
            "high_side_pressure_psig and suction_pressure_psig were recomputed from "
            "Raw_Simulated_Data_v1 by holding the underlying cycle temperatures fixed "
            "and remapping through each refrigerant's CoolProp saturation pressure-"
            "temperature curve (R410A -> R454B). All temperature, power, airflow, "
            "compressor-frequency, and EEV-position channels are unchanged from v1.",
            "Synthetic outdoor relative humidity (%), not derived from v1; generated "
            "with a seasonal base level, mild diurnal cycle, weak inverse relationship "
            "with outdoor temperature, and autocorrelated noise, bounded 35-98%. "
            "Illustrative for pipeline development only, per Section 1.6.",
            "Raw_Simulated_Data_v1.xlsx (Combined_Observed sheet), refrigerant "
            "substituted and relative_humidity_pct added",
        ],
    })
    readme2 = pd.concat([readme2, new_rows], ignore_index=True)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        readme2.to_excel(writer, sheet_name="README", index=False)
        obs2.to_excel(writer, sheet_name="Combined_Observed", index=False)
        truth.to_excel(writer, sheet_name="Truth_Reference_DO_NOT_USE", index=False)

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
