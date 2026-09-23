"""
Stage 4: SHAP explainability, applied per operating mode (Section 3.7, RQ4).

Uses the Stage-3 XGBoost model (sensors + regime + physics features) since
tree SHAP is exact and fast for XGBoost; explanations are computed
separately within each HDBSCAN regime so they reflect what is actually
driving predictions during, e.g., defrost specifically, rather than an
average blurred across mixed conditions.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
OUT_DIR = PROJECT_DIR / "Outputs"


def main():
    bundle = joblib.load(OUT_DIR / "stage3_xgb_model.joblib")
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

    test_df = pd.read_parquet(OUT_DIR / "test_predictions.parquet")
    X_test_scaled = scaler.transform(test_df[features])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=test_df.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in features], index=test_df.index)

    global_importance = (
        shap_df.abs().mean().sort_values(ascending=False)
        .rename("mean_abs_shap_value").reset_index()
        .rename(columns={"index": "feature"})
    )
    global_importance["feature"] = global_importance["feature"].str.replace("shap_", "", regex=False)
    print("=== Global SHAP importance (all regimes) ===")
    print(global_importance.to_string(index=False))

    # --- Per-regime SHAP importance ---
    per_regime_rows = []
    for regime, group_idx in test_df.groupby("hdbscan_cluster").groups.items():
        sub = shap_df.loc[group_idx].abs().mean()
        for feat, val in sub.items():
            per_regime_rows.append(dict(
                hdbscan_cluster=regime,
                true_regime=test_df.loc[group_idx, "_truth_regime"].mode().iat[0],
                feature=feat.replace("shap_", ""),
                mean_abs_shap_value=val,
                n_rows=len(group_idx),
            ))
    per_regime_df = pd.DataFrame(per_regime_rows)
    per_regime_df = per_regime_df.sort_values(["hdbscan_cluster", "mean_abs_shap_value"], ascending=[True, False])

    print("\n=== Top 5 SHAP features per regime ===")
    for regime, g in per_regime_df.groupby("hdbscan_cluster"):
        label = g["true_regime"].iat[0]
        print(f"\nRegime {regime} ({label}, n={g['n_rows'].iat[0]}):")
        print(g.head(5)[["feature", "mean_abs_shap_value"]].to_string(index=False))

    out = pd.concat([test_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
    out.to_parquet(OUT_DIR / "shap_values.parquet", index=False)
    global_importance.to_csv(OUT_DIR / "shap_global_importance.csv", index=False)
    per_regime_df.to_csv(OUT_DIR / "shap_per_regime_importance.csv", index=False)

    print(f"\nSaved: {OUT_DIR / 'shap_values.parquet'}")
    print(f"Saved: {OUT_DIR / 'shap_global_importance.csv'}")
    print(f"Saved: {OUT_DIR / 'shap_per_regime_importance.csv'}")


if __name__ == "__main__":
    main()
