"""
Stage 6: Consolidate all pipeline outputs into one results workbook and a
set of summary figures, mirroring the structure of Processed_Data_v1.xlsx
but covering the full methodology chapter (regime detection, ablation
study, SHAP, conformal prediction).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
OUT_DIR = PROJECT_DIR / "Outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def make_figures():
    with_regimes = pd.read_parquet(OUT_DIR / "with_regimes.parquet")
    ablation = pd.read_csv(OUT_DIR / "ablation_results.csv")
    shap_global = pd.read_csv(OUT_DIR / "shap_global_importance.csv")
    conformal = pd.read_parquet(OUT_DIR / "conformal_predictions.parquet")

    # 1. COP vs outdoor temperature, colored by regime
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, g in with_regimes.dropna(subset=["measured_cop"]).groupby("hdbscan_cluster"):
        label = g["_truth_regime"].mode().iat[0]
        ax.scatter(g["outdoor_temp_c"], g["measured_cop"], s=3, alpha=0.3, label=f"regime {regime} ({label})")
    ax.set_xlabel("Outdoor temperature (C)")
    ax.set_ylabel("Measured COP proxy")
    ax.set_title("COP vs outdoor temperature by HDBSCAN regime (simulated data)")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cop_vs_outdoor_temp_by_regime.png", dpi=150)
    plt.close(fig)

    # 2. Ablation study bar chart (test R2 by stage, best model per stage)
    test_res = ablation[ablation["split"] == "test"]
    best_per_stage = test_res.loc[test_res.groupby("stage")["R2"].idxmax()].sort_values("stage")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(best_per_stage["stage"], best_per_stage["R2"], color="#3b6ea5")
    for i, (s, r2, m) in enumerate(zip(best_per_stage["stage"], best_per_stage["R2"], best_per_stage["model"])):
        ax.text(i, r2 + 0.0005, f"{m}\nR2={r2:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("Test R-squared (best model per stage)")
    ax.set_title("Ablation study: incremental contribution of regime + physics features")
    ax.set_ylim(min(best_per_stage["R2"]) - 0.005, 1.001)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_test_r2.png", dpi=150)
    plt.close(fig)

    # 3. SHAP global importance
    fig, ax = plt.subplots(figsize=(7, 6))
    top = shap_global.head(12).sort_values("mean_abs_shap_value")
    ax.barh(top["feature"], top["mean_abs_shap_value"], color="#5a9367")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP feature importance (Stage-3 XGBoost model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_global_importance.png", dpi=150)
    plt.close(fig)

    # 4. Conformal prediction intervals, sample of test rows over time
    sample = conformal.dropna(subset=["cop_lower", "cop_upper"]).sort_values("timestamp").iloc[:400]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(sample["timestamp"], sample["cop_lower"], sample["cop_upper"],
                     color="#cbd8e8", label="90% conformal interval")
    ax.plot(sample["timestamp"], sample["measured_cop"], color="#1f3b57", lw=1, label="observed COP")
    ax.set_ylabel("COP")
    ax.set_title("Conformal prediction intervals vs observed COP (test set sample)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conformal_intervals_timeseries.png", dpi=150)
    plt.close(fig)

    # 5. Relative humidity: distribution by regime (new sensor, Section 3.2/3.3)
    fig, ax = plt.subplots(figsize=(7, 5))
    data_by_regime = [
        g["relative_humidity_pct"].dropna()
        for _, g in with_regimes.groupby("hdbscan_cluster")
    ]
    labels = [
        f"regime {r} ({g['_truth_regime'].mode().iat[0]})"
        for r, g in with_regimes.groupby("hdbscan_cluster")
    ]
    ax.boxplot(data_by_regime, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Relative humidity (%)")
    ax.set_title("Outdoor relative humidity by HDBSCAN regime (R-454B run, simulated)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "relative_humidity_by_regime.png", dpi=150)
    plt.close(fig)

    print(f"Saved 5 figures to {FIG_DIR}")


def build_workbook():
    with_regimes = pd.read_parquet(OUT_DIR / "with_regimes.parquet")
    regime_metrics = pd.read_csv(OUT_DIR / "regime_clustering_metrics.csv")
    ablation = pd.read_csv(OUT_DIR / "ablation_results.csv")
    shap_global = pd.read_csv(OUT_DIR / "shap_global_importance.csv")
    shap_per_regime = pd.read_csv(OUT_DIR / "shap_per_regime_importance.csv")
    conf_per_regime = pd.read_csv(OUT_DIR / "conformal_per_regime.csv")
    conf_overall = pd.read_csv(OUT_DIR / "conformal_overall.csv")

    readme = pd.DataFrame({
        "Field": [
            "Dataset", "Source", "Refrigerant", "Status", "Pipeline stages", "Target variable",
            "Important note",
        ],
        "Value": [
            "Full pipeline outputs -- regime detection, ablation study, SHAP, conformal prediction",
            "Derived from Raw_Simulated_Data_v2.xlsx (Combined_Observed sheet)",
            "R-454B (converted from the v1 R-410A run via matched-saturation-temperature "
            "pressure remapping; see Raw_Simulated_Data_v2.xlsx README for method). "
            "relative_humidity_pct added as a new synthetic sensor channel.",
            "SIMULATED -- pipeline/methodology development only, per proposal Section 1.6. "
            "Not field-measured data; all final performance claims will use real lab/field data.",
            "1) Data prep + feature engineering, 2) HDBSCAN regime detection "
            "(benchmarked vs k-means/GMM), 3) Ablation study (sensors -> +regime -> "
            "+physics features -> physics-informed graph attention model), "
            "4) SHAP explainability per regime, 5) Split conformal prediction per regime",
            "measured_cop: air-side heat-balance proxy COP computed from airflow, "
            "supply/return air temperature and total electrical power (Section 3.5)",
            "Ground-truth columns from Truth_Reference_DO_NOT_USE were used only to "
            "validate clustering/regime recovery, never as a model input feature.",
        ],
    })

    cluster_ct = pd.crosstab(with_regimes["hdbscan_cluster"], with_regimes["_truth_regime"])

    out_path = OUT_DIR / "Pipeline_Results_v2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        regime_metrics.to_excel(writer, sheet_name="Regime_Clustering_Metrics", index=False)
        cluster_ct.to_excel(writer, sheet_name="Regime_vs_Truth_Crosstab")
        ablation.to_excel(writer, sheet_name="Ablation_Study", index=False)
        shap_global.to_excel(writer, sheet_name="SHAP_Global_Importance", index=False)
        shap_per_regime.to_excel(writer, sheet_name="SHAP_Per_Regime", index=False)
        conf_per_regime.to_excel(writer, sheet_name="Conformal_Per_Regime", index=False)
        conf_overall.to_excel(writer, sheet_name="Conformal_Overall", index=False)

    print(f"Saved consolidated workbook: {out_path}")


def main():
    make_figures()
    build_workbook()


if __name__ == "__main__":
    main()
