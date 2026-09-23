"""
Stage 2: Regime detection (Section 3.4, RQ1).

Runs HDBSCAN on standardized sensor features to identify operating regimes
without predefined labels, then benchmarks it against k-means and a Gaussian
mixture model using standard internal clustering-quality metrics (silhouette,
Davies-Bouldin, Calinski-Harabasz -- Table 2), and cross-checks the resulting
clusters against the simulator's known ground-truth regime labels
(validation only, per the raw data README -- truth labels are never used as
a clustering input).
"""
import numpy as np
import pandas as pd
from pathlib import Path

import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
OUT_DIR = PROJECT_DIR / "Outputs"

REGIME_FEATURES = [
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
    "pressure_ratio",
]

RNG = 42
SAMPLE_N = 8000  # subsample for clustering-quality metrics (O(n^2) cost)


def cluster_metrics(X, labels, sample_n=SAMPLE_N, rng=RNG):
    """Silhouette/DBI/CH on a subsample (these metrics are expensive at
    full scale); excludes HDBSCAN noise points (label == -1)."""
    mask = labels != -1
    X_use, lab_use = X[mask], labels[mask]
    if len(np.unique(lab_use)) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    if len(X_use) > sample_n:
        idx = np.random.RandomState(rng).choice(len(X_use), sample_n, replace=False)
        X_use, lab_use = X_use[idx], lab_use[idx]
    return dict(
        silhouette=silhouette_score(X_use, lab_use),
        davies_bouldin=davies_bouldin_score(X_use, lab_use),
        calinski_harabasz=calinski_harabasz_score(X_use, lab_use),
    )


def main():
    df = pd.read_parquet(OUT_DIR / "prepared_data.parquet")
    X_raw = df[REGIME_FEATURES].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    results = {}

    # --- HDBSCAN (primary method) ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=150, min_samples=25)
    hdb_labels = clusterer.fit_predict(X)
    df["hdbscan_cluster"] = hdb_labels
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    noise_frac = (hdb_labels == -1).mean()
    m = cluster_metrics(X, hdb_labels)
    m.update(n_clusters=n_clusters, noise_fraction=noise_frac)
    results["HDBSCAN"] = m
    print(f"HDBSCAN: {n_clusters} clusters, {noise_frac:.2%} noise")
    print(pd.crosstab(df["hdbscan_cluster"], df["_truth_regime"]))

    # --- Benchmarks: k-means and GMM at the same k as HDBSCAN found ---
    k = max(n_clusters, 2)
    km = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit(X)
    df["kmeans_cluster"] = km.labels_
    m = cluster_metrics(X, km.labels_)
    m.update(n_clusters=k, noise_fraction=0.0)
    results["KMeans"] = m

    gmm = GaussianMixture(n_components=k, random_state=RNG, n_init=3).fit(X)
    gmm_labels = gmm.predict(X)
    df["gmm_cluster"] = gmm_labels
    m = cluster_metrics(X, gmm_labels)
    m.update(n_clusters=k, noise_fraction=0.0)
    results["GMM"] = m

    # --- Agreement with ground-truth regime (validation only, synthetic data) ---
    truth = df["_truth_regime"].astype("category").cat.codes
    for name, labels in [("HDBSCAN", hdb_labels), ("KMeans", km.labels_), ("GMM", gmm_labels)]:
        mask = labels != -1
        results[name]["ARI_vs_truth"] = adjusted_rand_score(truth[mask], labels[mask])
        results[name]["NMI_vs_truth"] = normalized_mutual_info_score(truth[mask], labels[mask])

    metrics_df = pd.DataFrame(results).T
    print("\nClustering comparison:\n", metrics_df)

    df.to_parquet(OUT_DIR / "with_regimes.parquet", index=False)
    metrics_df.to_csv(OUT_DIR / "regime_clustering_metrics.csv")
    print(f"\nSaved: {OUT_DIR / 'with_regimes.parquet'}")
    print(f"Saved: {OUT_DIR / 'regime_clustering_metrics.csv'}")


if __name__ == "__main__":
    main()
