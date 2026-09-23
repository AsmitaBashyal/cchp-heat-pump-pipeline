"""
Stage 5: Conformal prediction, calculated per operating mode (Section 3.7, RQ5).

Uses split conformal prediction (MAPIE) on top of the Stage-3 XGBoost model:
the validation split is used as the conformalization (calibration) set, and
prediction intervals are then produced for the test split. Because Section
3.7 calls for per-regime calibration, a separate conformal calibration is
fit within each HDBSCAN regime rather than pooling all regimes together,
and empirical coverage + interval width are reported per regime as well as
overall (Table 2 metrics: empirical coverage probability, mean interval
width).

Standard split-conformal coverage assumes exchangeability between the
calibration and test sets. Here calibration and test are drawn from the
same simulated run (no cross-site shift yet, since this is single-run
synthetic data), so this checks the calibration procedure itself works
correctly; cross-site shift-robust calibration (Section 3.7) applies once
multi-site field data exists.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from mapie.regression import SplitConformalRegressor

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
OUT_DIR = PROJECT_DIR / "Outputs"
CONFIDENCE_LEVEL = 0.90
TARGET = "measured_cop"


def time_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def main():
    bundle = joblib.load(OUT_DIR / "stage3_xgb_model.joblib")
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

    df = pd.read_parquet(OUT_DIR / "with_regimes.parquet")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    train_df, val_df, test_df = time_split(df)  # same split as Stage 3

    overall_rows = []
    per_regime_rows = []
    test_out = test_df.copy()
    test_out["cop_lower"] = np.nan
    test_out["cop_upper"] = np.nan

    regimes = sorted(df["hdbscan_cluster"].unique())
    for regime in regimes:
        val_r = val_df[val_df["hdbscan_cluster"] == regime]
        test_r = test_df[test_df["hdbscan_cluster"] == regime]
        if len(val_r) < 30 or len(test_r) < 10:
            print(f"Skipping regime {regime}: insufficient rows (val={len(val_r)}, test={len(test_r)})")
            continue

        Xv = scaler.transform(val_r[features])
        Xt = scaler.transform(test_r[features])

        mapie_reg = SplitConformalRegressor(
            estimator=model, confidence_level=CONFIDENCE_LEVEL, prefit=True,
        )
        mapie_reg.conformalize(Xv, val_r[TARGET])
        point_pred, intervals = mapie_reg.predict_interval(Xt)
        lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]

        idx = test_r.index
        test_out.loc[idx, "cop_lower"] = lower
        test_out.loc[idx, "cop_upper"] = upper
        test_out.loc[idx, "cop_point_pred"] = point_pred

        covered = (test_r[TARGET].to_numpy() >= lower) & (test_r[TARGET].to_numpy() <= upper)
        coverage = covered.mean()
        width = (upper - lower).mean()
        label = test_r["_truth_regime"].mode().iat[0]
        per_regime_rows.append(dict(
            hdbscan_cluster=regime, true_regime=label, n_test=len(test_r),
            target_coverage=CONFIDENCE_LEVEL, empirical_coverage=coverage,
            mean_interval_width=width,
        ))
        print(f"Regime {regime} ({label}): n_test={len(test_r)}, "
              f"empirical coverage={coverage:.3f} (target {CONFIDENCE_LEVEL}), "
              f"mean width={width:.4f}")

    per_regime_df = pd.DataFrame(per_regime_rows)

    # Overall coverage across all regimes combined (pooled test rows that got an interval)
    valid = test_out.dropna(subset=["cop_lower", "cop_upper"])
    covered_all = (valid[TARGET] >= valid["cop_lower"]) & (valid[TARGET] <= valid["cop_upper"])
    overall = dict(
        target_coverage=CONFIDENCE_LEVEL,
        empirical_coverage=covered_all.mean(),
        mean_interval_width=(valid["cop_upper"] - valid["cop_lower"]).mean(),
        n_test=len(valid),
    )
    print("\n=== Overall conformal calibration (pooled across regimes) ===")
    print(overall)

    per_regime_df.to_csv(OUT_DIR / "conformal_per_regime.csv", index=False)
    pd.DataFrame([overall]).to_csv(OUT_DIR / "conformal_overall.csv", index=False)
    test_out.to_parquet(OUT_DIR / "conformal_predictions.parquet", index=False)

    print(f"\nSaved: {OUT_DIR / 'conformal_per_regime.csv'}")
    print(f"Saved: {OUT_DIR / 'conformal_overall.csv'}")
    print(f"Saved: {OUT_DIR / 'conformal_predictions.parquet'}")


if __name__ == "__main__":
    main()
