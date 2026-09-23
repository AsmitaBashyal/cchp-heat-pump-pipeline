"""
weather.py

Loads real historical outdoor-temperature/humidity data to drive the simulator,
per the interim-plan decision: weather should be REAL, not invented, since it is
the dominant driver of everything downstream (capacity, COP, defrost frequency,
regime structure).

Two supported sources:

1. A user-supplied CSV (recommended) — e.g. exported from Environment and
   Climate Change Canada's historical climate data tool for a Kamloops station:
   https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
   Expected columns (case-insensitive, flexible): a timestamp column and a
   temperature column (°C). Relative humidity is optional; if absent, a
   plausible seasonal RH profile is synthesized (flagged as such).

2. A synthetic fallback generator (`synthesize_kamloops_winter`) that produces a
   physically plausible winter temperature trace using Kamloops' published
   1991-2020 climate normals (mean daily temps by month) plus diurnal and
   random variation, for use ONLY when no real station CSV is available yet.
   This is clearly less rigorous than option 1 and should be treated as a
   placeholder to be replaced by real data before any evaluation is reported.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# Kamloops YKA station, 1991-2020 climate normals, mean daily temperature (°C)
# Source: Environment and Climate Change Canada climate normals (public data).
# Used only as a fallback shape generator — replace with real station data ASAP.
KAMLOOPS_MONTHLY_MEAN_C = {
    1: -3.4, 2: 0.3, 3: 5.0, 4: 9.6, 5: 14.6, 6: 18.7,
    7: 21.8, 8: 21.5, 9: 16.1, 10: 9.3, 11: 2.2, 12: -3.1,
}
KAMLOOPS_MONTHLY_STD_C = {
    1: 5.5, 2: 4.8, 3: 4.2, 4: 3.8, 5: 3.5, 6: 3.0,
    7: 2.8, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4.4, 12: 5.2,
}


def load_from_csv(path: str, timestamp_col: str = None, temp_col: str = None,
                   rh_col: str = None, resample_to: str = "5min") -> pd.DataFrame:
    """Load a real weather CSV and resample onto the project's 5-minute grid
    via linear interpolation (weather changes slowly relative to 5 min, so
    interpolation is a reasonable and clearly-documented choice — unlike
    sensor gap-filling, which the proposal deliberately does NOT interpolate).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if timestamp_col is None:
        candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if not candidates:
            raise ValueError("Could not auto-detect a timestamp column; pass timestamp_col=")
        timestamp_col = candidates[0]
    if temp_col is None:
        candidates = [c for c in df.columns if "temp" in c.lower()]
        if not candidates:
            raise ValueError("Could not auto-detect a temperature column; pass temp_col=")
        temp_col = candidates[0]

    df["timestamp"] = pd.to_datetime(df[timestamp_col])
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)
    out["outdoor_temp_c"] = pd.to_numeric(df[temp_col], errors="coerce")

    if rh_col and rh_col in df.columns:
        out["outdoor_rh_pct"] = pd.to_numeric(df[rh_col], errors="coerce")
    else:
        # Simple, clearly-synthetic seasonal RH placeholder if not provided.
        month = out.index.month
        out["outdoor_rh_pct"] = 70 + 10 * np.sin((month - 1) / 12 * 2 * np.pi)

    out = out.resample(resample_to).interpolate(method="time")
    out.attrs["source"] = f"REAL:{path}"
    return out


def synthesize_kamloops_winter(start="2024-11-01", periods_days=150,
                                freq="5min", seed=None) -> pd.DataFrame:
    """FALLBACK ONLY. Generates a plausible but SYNTHETIC winter outdoor
    temperature/RH trace for Kamloops from published monthly climate normals.
    Clearly tag any output derived from this as simulated/synthetic weather,
    not measured weather, in every downstream artifact.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=int(periods_days * 24 * 60 / 5), freq=freq)

    months = idx.month
    means = np.array([KAMLOOPS_MONTHLY_MEAN_C[m] for m in months])
    stds = np.array([KAMLOOPS_MONTHLY_STD_C[m] for m in months])

    # Diurnal cycle: coldest ~06:00, warmest ~15:00, amplitude ~4C
    hour_frac = idx.hour + idx.minute / 60
    diurnal = -4 * np.cos((hour_frac - 6) / 24 * 2 * np.pi)

    # Slow-moving weather-system noise (autocorrelated random walk) + fast noise
    n = len(idx)
    slow_noise = np.cumsum(rng.normal(0, 0.05, n))
    slow_noise -= pd.Series(slow_noise).rolling(2000, min_periods=1).mean().values
    fast_noise = rng.normal(0, 0.3, n)

    outdoor_temp_c = means + diurnal + slow_noise + fast_noise
    outdoor_temp_c = np.clip(outdoor_temp_c, means - 3 * stds, means + 3 * stds)

    rh_pct = np.clip(75 + 10 * np.sin((months - 1) / 12 * 2 * np.pi) + rng.normal(0, 5, n), 30, 100)

    out = pd.DataFrame({"outdoor_temp_c": outdoor_temp_c, "outdoor_rh_pct": rh_pct}, index=idx)
    out.index.name = "timestamp"
    out.attrs["source"] = "SYNTHETIC:kamloops_climate_normals_fallback"
    return out
