"""
regime_discovery_beginner.py

Your first "regime discovery" run using HDBSCAN -- the unsupervised clustering
method from your proposal (Chapter 7, Section 7.2). This script:

  1. Loads your simulated sensor data.
  2. Picks the sensor columns that describe the heat pump's operating state.
  3. Standardizes them (puts every sensor on the same numeric scale, since
     HDBSCAN is sensitive to raw units -- pressure in psig is a much bigger
     number than temperature in Celsius, and without standardizing, pressure
     would unfairly dominate the clustering).
  4. Runs HDBSCAN to automatically group similar time periods into clusters
     ("candidate operating regimes") -- WITHOUT being told in advance how
     many regimes exist or what they mean.
  5. Because this is SIMULATED data, we also have a "truth" file that tells
     us what regime the simulator actually intended (steady-state heating vs.
     defrost). This script compares HDBSCAN's discovered clusters against
     that truth -- a real field dataset would never let you do this check,
     so make the most of it now.
  6. Saves a couple of charts so you can see the clusters visually.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
import os

print("=" * 60)
print("STEP 1: Load the data")
print("=" * 60)

observed = pd.read_csv("data/raw/SIMULATED_v1_combined.csv")
truth = pd.read_csv("data/raw/SIMULATED_v1_TRUTH_DO_NOT_TRAIN_ON.csv")
print(f"Observed (sensor) data: {observed.shape[0]} rows, {observed.shape[1]} columns")
print(f"Truth reference data:   {truth.shape[0]} rows (for checking our work only)")

# ---- Step 2: Pick the features HDBSCAN will use to find regimes ----
# These are exactly the kind of physically-meaningful features your proposal
# specifies (Section 5.8) -- pressures, temperatures, and power draw.
feature_columns = [
    "high_side_pressure_psig",
    "suction_pressure_psig",
    "liquid_line_temp_c",
    "suction_line_temp_c",
    "outdoor_temp_c",
    "return_air_temp_c",
    "supply_air_temp_c",
    "top_of_outdoor_unit_temp_c",
    "electrical_power_w",
]

# Drop any rows with missing values (remember: these come from the simulated
# "dropout" gaps -- a real pipeline would flag and investigate these rather
# than just dropping them, but for this first pass, dropping is fine).
data = observed[feature_columns].dropna()
print(f"\nUsing {len(feature_columns)} features on {len(data)} complete rows "
      f"(dropped {len(observed) - len(data)} rows with missing sensor values).")

# ---- Step 3: Standardize ----
print("\n" + "=" * 60)
print("STEP 2: Standardizing the features")
print("=" * 60)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Done. Each sensor now has mean 0 and standard deviation 1.")

# ---- Step 4: Run HDBSCAN ----
print("\n" + "=" * 60)
print("STEP 3: Running HDBSCAN (this may take 30-90 seconds)")
print("=" * 60)

# min_cluster_size: the smallest number of 5-minute readings that counts as
# a real regime. At 5-min intervals, 12 readings = 1 hour -- a reasonable
# floor for "this is a real, sustained operating state" rather than noise.
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
cluster_labels = clusterer.fit_predict(data_scaled)

data = data.copy()
data["cluster"] = cluster_labels

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = int((cluster_labels == -1).sum())
print(f"HDBSCAN found {n_clusters} cluster(s), plus {n_noise} points labeled 'noise' "
      f"({n_noise / len(data) * 100:.1f}% of the data).")
print("\n'Noise' points are readings that didn't fit cleanly into any stable "
      "regime -- often transitions between states (e.g. cold-start, mid-defrost).")

print("\nCluster sizes:")
print(data["cluster"].value_counts().sort_index())

# ---- Step 5: Compare against the SIMULATED ground truth ----
print("\n" + "=" * 60)
print("STEP 4: Comparing HDBSCAN's clusters against the known truth")
print("=" * 60)
print("(This check is only possible because the data is simulated with a")
print(" known regime label built in -- a real field dataset can't do this.)\n")

# Align the truth labels to the same rows we kept after dropping missing values
truth_aligned = truth.loc[data.index, "_truth_regime"]
comparison = pd.crosstab(data["cluster"], truth_aligned)
print(comparison)

print("\nHow to read this table: each row is one of HDBSCAN's discovered")
print("clusters (-1 means 'noise'). Each column is the TRUE regime the")
print("simulator actually used. If HDBSCAN is doing a good job, each row")
print("should be dominated by mostly ONE column -- meaning that cluster")
print("corresponds cleanly to one real physical regime.")

# ---- Step 6: Save visual charts ----
print("\n" + "=" * 60)
print("STEP 5: Saving charts to the 'plots' folder")
print("=" * 60)
os.makedirs("plots", exist_ok=True)

# Chart: outdoor temp vs power, colored by HDBSCAN's discovered cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    data["outdoor_temp_c"], data["electrical_power_w"],
    c=data["cluster"], cmap="tab10", alpha=0.3, s=5
)
plt.title("HDBSCAN Discovered Clusters\n(color = cluster HDBSCAN found on its own)")
plt.xlabel("Outdoor Temperature (Celsius)")
plt.ylabel("Electrical Power (Watts)")
plt.colorbar(scatter, label="Cluster ID (-1 = noise)")
plt.tight_layout()
plt.savefig("plots/05_hdbscan_clusters.png")
plt.close()
print("Saved: plots/05_hdbscan_clusters.png")

# Chart: same axes, but colored by the TRUE regime, for direct comparison
truth_codes = truth_aligned.astype("category").cat.codes
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(
    data["outdoor_temp_c"], data["electrical_power_w"],
    c=truth_codes, cmap="tab10", alpha=0.3, s=5
)
plt.title("TRUE Regimes (from the simulator)\n(color = the actual regime the simulator used)")
plt.xlabel("Outdoor Temperature (Celsius)")
plt.ylabel("Electrical Power (Watts)")
plt.tight_layout()
plt.savefig("plots/06_true_regimes.png")
plt.close()
print("Saved: plots/06_true_regimes.png")

print("\nAll done! Compare plots/05 and plots/06 side by side in VS Code --")
print("if HDBSCAN worked well, the color patterns should roughly match.")
