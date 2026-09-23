"""
shap_beginner.py

Your first predictive model + SHAP explanation. This script:

  1. Trains a simple, well-established model (XGBoost -- a "gradient boosted
     tree" model, one of the three baseline models your proposal specifies
     in the Mitacs Phase 1 scope) to predict electrical power draw from the
     other sensor readings.
  2. Checks how accurate the model is on data it never saw during training
     (this is called a "train/test split" -- a core data science practice:
     always evaluate on data the model hasn't memorized).
  3. Uses SHAP to explain WHY the model makes its predictions -- which
     sensors matter most, overall and for individual predictions.
  4. Saves a SHAP summary chart you can open and interpret.

Model choice note: XGBoost was picked here (rather than the physics-informed
graph transformer from your proposal) because it's simple, fast, has a
built-in exact SHAP calculation, and is explicitly one of your proposal's
Mitacs Phase 1 baseline models -- a legitimate deliverable on its own, not
just a stepping stone.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import shap
import os

print("=" * 60)
print("STEP 1: Load the data and choose what to predict")
print("=" * 60)

df = pd.read_csv("data/raw/SIMULATED_v1_combined.csv")

# We'll predict electrical power draw (the TARGET) from the other sensors
# (the FEATURES). This mirrors your proposal's proxy-COP / capacity
# prediction task, using power as a simpler, well-understood first target.
target_column = "electrical_power_w"
feature_columns = [
    "high_side_pressure_psig",
    "suction_pressure_psig",
    "liquid_line_temp_c",
    "suction_line_temp_c",
    "outdoor_temp_c",
    "return_air_temp_c",
    "supply_air_temp_c",
    "top_of_outdoor_unit_temp_c",
]

data = df[feature_columns + [target_column]].dropna()
X = data[feature_columns]
y = data[target_column]
print(f"Predicting '{target_column}' from {len(feature_columns)} sensor features, "
      f"using {len(data)} complete rows.")

# ---- Step 2: Split into training data and test data ----
print("\n" + "=" * 60)
print("STEP 2: Splitting into training data and test data")
print("=" * 60)
# We hold back 20% of the data and NEVER let the model see it during
# training. This is how we honestly check whether the model actually
# learned something useful, rather than just memorizing the training data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows "
      "the model has never seen.")

# ---- Step 3: Train the model ----
print("\n" + "=" * 60)
print("STEP 3: Training the XGBoost model")
print("=" * 60)
model = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)
print("Training complete.")

# ---- Step 4: Check accuracy on the unseen test data ----
print("\n" + "=" * 60)
print("STEP 4: Checking accuracy on data the model never saw")
print("=" * 60)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.1f} Watts")
print("  -> On average, predictions are off by about this many Watts.")
print(f"R-squared (R2): {r2:.4f}")
print("  -> 1.0 would mean perfect predictions; 0.0 would mean no better")
print("     than just guessing the average every time. Closer to 1 is better.")

# ---- Step 5: Explain the model with SHAP ----
print("\n" + "=" * 60)
print("STEP 5: Explaining the model with SHAP")
print("=" * 60)
print("Computing SHAP values (this attributes each prediction to the")
print("sensors that drove it)...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Average absolute SHAP value per feature = overall importance ranking
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=feature_columns).sort_values(ascending=False)
print("\nOverall feature importance (bigger = matters more to predictions):")
for feat, val in importance.items():
    print(f"  {feat:35s} {val:8.2f}")

# ---- Step 6: Save SHAP charts ----
print("\n" + "=" * 60)
print("STEP 6: Saving SHAP charts to the 'plots' folder")
print("=" * 60)
os.makedirs("plots", exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("plots/07_shap_summary.png", bbox_inches="tight")
plt.close()
print("Saved: plots/07_shap_summary.png")

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("plots/08_shap_importance_bar.png", bbox_inches="tight")
plt.close()
print("Saved: plots/08_shap_importance_bar.png")

print("\nAll done! Open plots/07 and plots/08 to see which sensors matter most.")
print("\nHow to read plot 08 (the bar chart): longer bars = that sensor has a")
print("bigger average effect on the model's predictions. This should broadly")
print("match physical intuition -- outdoor temperature and pressures should")
print("rank high, since they directly drive compressor work in the physics.")
print("\nHow to read plot 07 (the summary/beeswarm chart): each dot is one")
print("test-data reading. Red = that sensor had a high value for that")
print("reading, blue = a low value. Position left/right shows whether that")
print("reading pushed the power prediction up or down. E.g. if outdoor_temp_c")
print("shows red dots (high, i.e. WARM) clustered on the left (pushing power")
print("DOWN), that matches the physics: warmer outside -> less power needed.")
