"""
cycle_model.py

A simplified but thermodynamically consistent vapor-compression heat pump
simulator, implementing the relations from the proposal's Chapter 2
(steady-flow energy balance, COP, superheat/subcooling, Carnot bound) using
CoolProp for real refrigerant properties (refrigerant.py).

This is NOT a substitute for manufacturer performance data or the real field
deployment. It is a physically-grounded data generator for pipeline and
architecture development while the TRU lab/field data accumulate. Every run
should be tagged with its config so simulated output is traceable and never
silently mixed with real sensor data.

Simplifications made explicitly (document these in any methods write-up):
  - Single-zone building model (UA-based static heat loss, no thermal mass).
  - Compressor modeled with a fixed isentropic efficiency (not a manufacturer
    compressor map).
  - Superheat/subcooling are control setpoints (representing a working EEV
    controller), not solved from a full heat-exchanger model.
  - Airflow is assumed constant except during defrost.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import refrigerant as ref


@dataclass
class CCHPConfig:
    refrigerant: str = "R410A"          # "R410A" or "R454B"
    indoor_setpoint_c: float = 21.0
    building_ua_w_per_c: float = 350.0  # heat loss coefficient (W/°C), typical mid-size home order-of-magnitude
    rated_capacity_w: float = 10_500.0  # nominal heating capacity at ~8.3C, order-of-magnitude residential CCHP
    rated_cop_at_8_3c: float = 3.6
    bivalent_temp_c: float = -15.0      # below this, supplemental resistance heat engages
    min_compressor_speed_frac: float = 0.25
    isentropic_efficiency: float = 0.70
    motor_efficiency: float = 0.90
    superheat_setpoint_c: float = 8.0
    subcooling_setpoint_c: float = 5.0
    evap_approach_c: float = 6.0        # evaporating temp below outdoor temp
    cond_approach_c: float = 15.0       # condensing temp above supply air target
    supply_air_target_c: float = 45.0   # nominal supply air temp at full load
    fan_cfm_nominal: float = 1200.0
    defrost_trigger_temp_c: float = -2.0   # defrost most likely in this band (per proposal Sec 2.4)
    defrost_trigger_temp_high_c: float = 4.0
    defrost_rh_threshold_pct: float = 70.0
    defrost_min_interval_min: float = 45.0
    defrost_duration_min: float = 8.0
    supplemental_heat_w: float = 8_000.0
    random_seed: int = 42


def capacity_derate_curve(t_outdoor_c: float) -> float:
    """Fraction of rated capacity available at a given outdoor temperature.
    Representative CCHP-style piecewise curve (Section 2.3): near-100% capacity
    down to ~-8C, then increasingly steep falloff. This is a representative
    shape, NOT a specific manufacturer's published curve — replace with real
    submittal data for a specific unit once one is selected for the field study.
    """
    t = t_outdoor_c
    if t >= 8.3:
        return 1.05
    elif t >= -8.3:
        # linear from 1.05 at 8.3C to 0.75 at -8.3C
        return 1.05 - (8.3 - t) / (8.3 - (-8.3)) * 0.30
    elif t >= -25.0:
        # steeper falloff below -8.3C, down to ~0.35 at -25C
        return 0.75 - (-8.3 - t) / (-8.3 - (-25.0)) * 0.40
    else:
        return 0.30


def _carnot_cop_heating(t_outdoor_k, t_indoor_k):
    return t_indoor_k / max(t_indoor_k - t_outdoor_k, 1e-3)


def simulate_timestep(t_outdoor_c, rh_pct, cfg: CCHPConfig, defrost_active: bool):
    """Compute one 5-minute timestep of the cycle. Returns a dict of raw
    'sensor' readings plus derived truth-quantities (kept separate so you can
    hide the truth columns from your ML pipeline and only feed it what a real
    sensor deployment would actually provide).
    """
    t_indoor_c = cfg.indoor_setpoint_c
    load_w = max(cfg.building_ua_w_per_c * (t_indoor_c - t_outdoor_c), 0.0)

    derate = capacity_derate_curve(t_outdoor_c)
    available_capacity_w = cfg.rated_capacity_w * derate

    below_bivalent = t_outdoor_c < cfg.bivalent_temp_c
    if below_bivalent:
        # Compressor still runs near max, supplemental heat tops up the gap.
        compressor_speed_frac = 1.0
        hp_delivered_w = available_capacity_w
        supplemental_active = True
        supplemental_w = max(load_w - hp_delivered_w, 0.0)
        supplemental_w = min(supplemental_w, cfg.supplemental_heat_w)
    else:
        target_w = min(load_w, available_capacity_w)
        speed_frac = target_w / cfg.rated_capacity_w if cfg.rated_capacity_w > 0 else 0.0
        compressor_speed_frac = float(np.clip(speed_frac, cfg.min_compressor_speed_frac, 1.0))
        hp_delivered_w = compressor_speed_frac * cfg.rated_capacity_w * (derate / 1.0)
        hp_delivered_w = min(hp_delivered_w, available_capacity_w)
        supplemental_active = False
        supplemental_w = 0.0

    if defrost_active:
        # Reverse-cycle defrost: no useful heating delivered to the space,
        # indoor fan suspended, supplemental heat may engage to offset.
        hp_delivered_w = 0.0
        compressor_speed_frac = max(compressor_speed_frac, 0.6)
        supplemental_active = True
        supplemental_w = min(load_w, cfg.supplemental_heat_w)

    # --- Refrigerant-side state points ---
    t_evap_c = t_outdoor_c - cfg.evap_approach_c if not defrost_active else t_outdoor_c + 20.0
    t_cond_target_c = cfg.supply_air_target_c * (0.5 + 0.5 * compressor_speed_frac) + cfg.cond_approach_c
    t_cond_c = t_cond_target_c if not defrost_active else t_outdoor_c + 5.0

    p_suction_pa = ref.sat_pressure_from_temp(ref.c_to_k(t_evap_c), cfg.refrigerant)
    p_high_pa = ref.sat_pressure_from_temp(ref.c_to_k(t_cond_c), cfg.refrigerant)

    t_suction_c = t_evap_c + cfg.superheat_setpoint_c
    t_liquid_c = t_cond_c - cfg.subcooling_setpoint_c

    h_suction = ref.enthalpy_superheated(p_suction_pa, ref.c_to_k(t_suction_c), cfg.refrigerant)
    h_liquid = ref.enthalpy_subcooled_liquid(p_high_pa, ref.c_to_k(t_liquid_c), cfg.refrigerant)
    h_isentropic_out = ref.isentropic_outlet_enthalpy(
        p_suction_pa, ref.c_to_k(t_suction_c), p_high_pa, cfg.refrigerant
    )
    h_discharge = h_suction + (h_isentropic_out - h_suction) / cfg.isentropic_efficiency

    # Mass flow solved from required condenser heat rejection ~= delivered capacity
    # (ignores compressor motor heat added to refrigerant vs. casing losses — a
    # standard simplification for a proxy-COP-level simulator).
    delta_h_cond = max(h_discharge - h_liquid, 1.0)
    if hp_delivered_w > 0:
        mdot_kg_s = hp_delivered_w / delta_h_cond
    else:
        mdot_kg_s = 0.0

    compressor_work_w = mdot_kg_s * (h_discharge - h_suction)
    electrical_power_w = compressor_work_w / cfg.motor_efficiency
    fan_power_w = 150.0 * (0.4 + 0.6 * compressor_speed_frac) if not defrost_active else 0.0
    total_power_w = electrical_power_w + fan_power_w + (0 if not supplemental_active else 0)  # supplemental metered separately below

    proxy_cop = hp_delivered_w / electrical_power_w if electrical_power_w > 1e-3 else np.nan

    carnot_cop = _carnot_cop_heating(ref.c_to_k(t_outdoor_c), ref.c_to_k(t_indoor_c))

    superheat_c = t_suction_c - ref.k_to_c(ref.sat_temp_from_pressure(p_suction_pa, cfg.refrigerant))
    subcooling_c = ref.k_to_c(ref.sat_temp_from_pressure(p_high_pa, cfg.refrigerant)) - t_liquid_c

    # --- Air-side ---
    airflow_cfm = cfg.fan_cfm_nominal * (0.4 + 0.6 * compressor_speed_frac) if not defrost_active else 0.0
    if airflow_cfm > 1 and hp_delivered_w > 0:
        # Q(W) = 1.23 * CFM * dT(C) is a standard air-side sensible-heat approx (SI-adjusted constant)
        supply_delta_t = hp_delivered_w / (1.23 * airflow_cfm)
    else:
        supply_delta_t = 0.0
    return_air_c = t_indoor_c + np.random.normal(0, 0.3)
    supply_air_c = return_air_c + supply_delta_t if not defrost_active else return_air_c - 2.0

    # --- Frost / top-of-unit indicator ---
    if defrost_active:
        top_of_unit_c = t_outdoor_c + 15.0  # hot gas melting frost
    else:
        frosting_conditions = (cfg.defrost_trigger_temp_c <= t_outdoor_c <= cfg.defrost_trigger_temp_high_c
                                and rh_pct >= cfg.defrost_rh_threshold_pct)
        top_of_unit_c = t_outdoor_c - (3.0 if frosting_conditions else 0.5)

    # --- Instrumentation-gap variables (Table 5.1 targets) ---
    compressor_freq_hz = 30 + compressor_speed_frac * 60  # e.g. 30-90 Hz variable-speed range
    eev_position_pct = float(np.clip(50 + (superheat_c - cfg.superheat_setpoint_c) * 5, 0, 100))
    refrigerant_type = cfg.refrigerant

    return {
        "high_side_pressure_psig": p_high_pa / 6894.76 - 14.7,
        "suction_pressure_psig": p_suction_pa / 6894.76 - 14.7,
        "liquid_line_temp_c": t_liquid_c,
        "suction_line_temp_c": t_suction_c,
        "outdoor_temp_c": t_outdoor_c,
        "return_air_temp_c": return_air_c,
        "supply_air_temp_c": supply_air_c,
        "top_of_outdoor_unit_temp_c": top_of_unit_c,
        "electrical_power_w": electrical_power_w + fan_power_w,
        "supplemental_heat_power_w": supplemental_w,
        # instrumentation-gap sensors (simulated ahead of real Phase 2 install)
        "compressor_frequency_hz": compressor_freq_hz if hp_delivered_w > 0 or defrost_active else 0.0,
        "eev_position_pct": eev_position_pct,
        "airflow_cfm": airflow_cfm,
        "defrost_flag": bool(defrost_active),
        "refrigerant_type": refrigerant_type,
        # ground-truth / derived quantities -- keep separate from "sensor" columns
        "_truth_proxy_cop": proxy_cop,
        "_truth_carnot_cop": carnot_cop,
        "_truth_superheat_c": superheat_c,
        "_truth_subcooling_c": subcooling_c,
        "_truth_delivered_capacity_w": hp_delivered_w,
        "_truth_compressor_speed_frac": compressor_speed_frac,
        "_truth_regime": _label_regime(defrost_active, below_bivalent, supplemental_active, compressor_speed_frac),
    }


def _label_regime(defrost_active, below_bivalent, supplemental_active, speed_frac):
    if defrost_active:
        return "defrost"
    if below_bivalent or supplemental_active:
        return "supplemental_heat_engaged"
    if speed_frac <= 0.3:
        return "low_load_steady_state"
    return "steady_state_heating"


def run_simulation(weather_df: pd.DataFrame, cfg: CCHPConfig) -> pd.DataFrame:
    """Run the cycle model over a full weather time series, with defrost-cycle
    state machine layered on top (timer + threshold trigger per Section 2.4).
    """
    np.random.seed(cfg.random_seed)
    rows = []
    minutes_since_defrost = cfg.defrost_min_interval_min
    defrost_remaining_min = 0.0
    step_min = 5.0

    for ts, row in weather_df.iterrows():
        t_outdoor_c = float(row["outdoor_temp_c"])
        rh_pct = float(row.get("outdoor_rh_pct", 70.0))

        defrost_active = defrost_remaining_min > 0
        if defrost_active:
            defrost_remaining_min -= step_min
        else:
            minutes_since_defrost += step_min
            frosting_conditions = (cfg.defrost_trigger_temp_c <= t_outdoor_c <= cfg.defrost_trigger_temp_high_c
                                    and rh_pct >= cfg.defrost_rh_threshold_pct)
            if frosting_conditions and minutes_since_defrost >= cfg.defrost_min_interval_min:
                defrost_active = True
                defrost_remaining_min = cfg.defrost_duration_min
                minutes_since_defrost = 0.0

        result = simulate_timestep(t_outdoor_c, rh_pct, cfg, defrost_active)
        result["timestamp"] = ts
        rows.append(result)

    df = pd.DataFrame(rows).set_index("timestamp")
    return df
