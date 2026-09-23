"""
run_simulation.py

Entry point: generate a SIMULATED CCHP field-analog dataset for pipeline and
model development, per the interim-plan discussion.

Usage:
    # Fallback synthetic weather (use only until you have a real weather CSV):
    python run_simulation.py --weather synthetic --days 150 --refrigerant R410A --out data/raw

    # Real weather (recommended) -- download a CSV from
    # https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
    # for a Kamloops station, then:
    python run_simulation.py --weather real --weather-csv path/to/kamloops_weather.csv --out data/raw
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from cchp_simulator import weather, cycle_model, noise_and_export


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weather", choices=["real", "synthetic"], default="synthetic")
    p.add_argument("--weather-csv", type=str, default=None)
    p.add_argument("--days", type=int, default=150)
    p.add_argument("--refrigerant", choices=["R410A", "R454B"], default="R410A")
    p.add_argument("--out", type=str, default="data/raw")
    p.add_argument("--run-tag", type=str, default="phase0_dev")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.weather == "real":
        if not args.weather_csv:
            raise SystemExit("--weather-csv is required when --weather real is selected")
        wx = weather.load_from_csv(args.weather_csv)
        print(f"Loaded REAL weather from {args.weather_csv}: {wx.attrs.get('source')}")
    else:
        wx = weather.synthesize_kamloops_winter(periods_days=args.days, seed=args.seed)
        print(f"Using SYNTHETIC fallback weather: {wx.attrs.get('source')}")
        print("  -> Replace with real station data before reporting any results.")

    cfg = cycle_model.CCHPConfig(refrigerant=args.refrigerant, random_seed=args.seed)
    print(f"Running cycle simulation over {len(wx)} timesteps "
          f"({len(wx) * 5 / 60 / 24:.1f} days at 5-min resolution)...")
    raw = cycle_model.run_simulation(wx, cfg)

    noisy = noise_and_export.add_sensor_noise(raw, seed=args.seed)
    noisy = noise_and_export.inject_dropouts(noisy, seed=args.seed + 1)

    observed, truth = noise_and_export.split_truth_and_observed(noisy)

    noise_and_export.export_per_sensor_csvs(observed, args.out, args.run_tag)
    truth_path = f"{args.out}/SIMULATED_{args.run_tag}_TRUTH_DO_NOT_TRAIN_ON.csv"
    truth.reset_index().to_csv(truth_path, index=False)

    print(f"\nDone. Observed (train-on) sensor files written to: {args.out}/SIMULATED_{args.run_tag}_*.csv")
    print(f"Ground-truth reference (for validating HDBSCAN/models, NOT a model input): {truth_path}")
    print("\nRegime distribution (ground truth, for validation only):")
    print(truth["_truth_regime"].value_counts())


if __name__ == "__main__":
    main()
