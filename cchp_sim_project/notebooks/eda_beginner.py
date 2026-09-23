"""
eda_beginner.py

A first, beginner-friendly look at the simulated CCHP dataset.
This script:
  1. Loads the combined sensor data file into a table (called a "DataFrame").
  2. Prints basic facts about it (how many rows/columns, column names).
  3. Prints simple statistics (average, min, max) for each sensor.
  4. Saves a few charts as image files you can open and look at.

You don't need to understand every line yet -- just run it and look at the
output. Each section below is commented in plain English so you can start
connecting the code to what it does.
"""

# "import" brings in extra tools other people have already written.
# pandas handles tables of data. matplotlib draws charts.
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets this script save chart images without needing a screen popup
import matplotlib.pyplot as plt
import os

# ---- Step 1: Load the data ----
# This reads the CSV file into a table we can work with, called "df"
# (short for "DataFrame" -- just means "a spreadsheet-like table in memory").
data_path = "data/raw/SIMULATED_v1_combined.csv"
df = pd.read_csv(data_path)

print("=" * 60)
print("STEP 1: Basic facts about your data")
print("=" * 60)
print(f"Number of rows (5-minute readings): {len(df)}")
print(f"Number of columns (sensors/fields): {len(df.columns)}")
print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

# ---- Step 2: Look at the first few rows ----
print("\n" + "=" * 60)
print("STEP 2: First 5 rows of the data (a quick peek)")
print("=" * 60)
print(df.head())

# ---- Step 3: Simple statistics for the important sensors ----
print("\n" + "=" * 60)
print("STEP 3: Basic statistics (average, min, max) for key sensors")
print("=" * 60)
key_columns = [
    "outdoor_temp_c",
    "electrical_power_w",
    "high_side_pressure_psig",
    "suction_pressure_psig",
    "supply_air_temp_c",
]
print(df[key_columns].describe())

# ---- Step 4: Save some charts as image files ----
print("\n" + "=" * 60)
print("STEP 4: Saving charts to the 'plots' folder...")
print("=" * 60)

os.makedirs("plots", exist_ok=True)

# Chart 1: Outdoor temperature over time -- shows the simulated winter unfold
df["timestamp"] = pd.to_datetime(df["timestamp"])
plt.figure(figsize=(12, 4))
plt.plot(df["timestamp"], df["outdoor_temp_c"])
plt.title("Outdoor Temperature Over Time (Simulated)")
plt.xlabel("Date")
plt.ylabel("Outdoor Temperature (Celsius)")
plt.tight_layout()
plt.savefig("plots/01_outdoor_temperature_over_time.png")
plt.close()
print("Saved: plots/01_outdoor_temperature_over_time.png")

# Chart 2: Electrical power over time
plt.figure(figsize=(12, 4))
plt.plot(df["timestamp"], df["electrical_power_w"], color="orange")
plt.title("Electrical Power Draw Over Time (Simulated)")
plt.xlabel("Date")
plt.ylabel("Power (Watts)")
plt.tight_layout()
plt.savefig("plots/02_electrical_power_over_time.png")
plt.close()
print("Saved: plots/02_electrical_power_over_time.png")

# Chart 3: How electrical power relates to outdoor temperature (a scatter plot)
plt.figure(figsize=(8, 6))
plt.scatter(df["outdoor_temp_c"], df["electrical_power_w"], alpha=0.1, s=5)
plt.title("Electrical Power vs Outdoor Temperature")
plt.xlabel("Outdoor Temperature (Celsius)")
plt.ylabel("Power (Watts)")
plt.tight_layout()
plt.savefig("plots/03_power_vs_outdoor_temp.png")
plt.close()
print("Saved: plots/03_power_vs_outdoor_temp.png")

# Chart 4: Distribution of outdoor temperatures (a histogram)
plt.figure(figsize=(8, 5))
plt.hist(df["outdoor_temp_c"], bins=40, color="skyblue", edgecolor="black")
plt.title("Distribution of Outdoor Temperatures (Simulated Winter)")
plt.xlabel("Outdoor Temperature (Celsius)")
plt.ylabel("Number of 5-minute readings")
plt.tight_layout()
plt.savefig("plots/04_outdoor_temp_histogram.png")
plt.close()
print("Saved: plots/04_outdoor_temp_histogram.png")

print("\nAll done! Open the 'plots' folder in VS Code to view the charts.")
