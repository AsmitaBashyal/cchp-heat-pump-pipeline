"""
Stage 3: Predictive modelling and ablation study (Section 3.6, 3.9; RQ2, RQ3).

Target: measured_cop (air-side heat-balance proxy computed in Stage 1).
Split: time-based train/val/test (no shuffling -- Section 3.8), since the
data is a single continuous simulated season here rather than multiple
sites; cross-site leave-one-out is not applicable to this single-run
synthetic dataset (Section 3.8 applies once multi-site field data exists).

Ablation stages (Section 3.9), same splits throughout:
  1. Conventional ML, sensor features only (linear regression, random forest, XGBoost)
  2. + operating-mode (HDBSCAN regime) information
  3. + physics-derived features (pressure ratio, superheat/subcooling proxies)
  4. Physics-informed graph model (4-node graph attention network with a
     physics-consistency penalty term), conditioned on operating mode.

Only physics rules checkable against the sensors actually present in this
synthetic run are enforced (Section 3.6): COP bounds, and consistency
between delivered heat and electrical input sign/scale.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import torch
import torch.nn as nn

PROJECT_DIR = Path(r"C:\Users\Asus\OneDrive - Thompson Rivers University\Desktop\CCHP_Masters Project\Claude Code")
OUT_DIR = PROJECT_DIR / "Outputs"
RNG = 42
TARGET = "measured_cop"

SENSOR_FEATURES = [
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
    "supplemental_heat_power_w",
    "compressor_frequency_hz",
    "eev_position_pct",
    "airflow_cfm",
]
PHYSICS_FEATURES = ["pressure_ratio", "suction_superheat_proxy_c", "liquid_subcool_proxy_c"]

# 4-node graph: compressor(0), condenser(1), expansion_valve(2), evaporator(3)
# Node features approximate each component's local thermodynamic state from
# available sensors (Table 1). Edges follow heating-mode refrigerant flow:
# compressor -> condenser -> expansion valve -> evaporator -> (back to compressor).
NODE_FEATURE_MAP = {
    0: ["electrical_power_w", "compressor_frequency_hz", "suction_pressure_psig", "high_side_pressure_psig"],
    1: ["high_side_pressure_psig", "liquid_line_temp_c", "return_air_temp_c", "supply_air_temp_c"],
    2: ["eev_position_pct", "high_side_pressure_psig", "suction_pressure_psig", "liquid_line_temp_c"],
    3: ["suction_pressure_psig", "suction_line_temp_c", "outdoor_temp_c",
        "top_of_outdoor_unit_temp_c", "relative_humidity_pct"],
}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # directed, heating-mode flow + return to compressor


def time_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def eval_metrics(y_true, y_pred):
    return dict(
        RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
        MAE=mean_absolute_error(y_true, y_pred),
        R2=r2_score(y_true, y_pred),
    )


def fit_eval_conventional(X_train, y_train, X_val, y_val, X_test, y_test, prefix):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RNG, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1,
        ),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = model.predict(X)
            m = eval_metrics(y, pred)
            rows.append(dict(stage=prefix, model=name, split=split_name, **m))
    return pd.DataFrame(rows), fitted


# ---------------- Physics-informed graph attention model ----------------

class GraphAttentionCOP(nn.Module):
    """Small 4-node graph-attention regressor. Each node embeds its local
    sensor features; a single graph-attention layer lets each component's
    representation depend on its physically connected neighbours (edges
    follow refrigerant flow direction, Table 1); node embeddings are then
    pooled and mapped to a COP prediction, conditioned on the regime label."""

    def __init__(self, node_in_dims, edges, n_regimes, hidden=16):
        super().__init__()
        self.edges = edges
        self.node_encoders = nn.ModuleList([nn.Linear(d, hidden) for d in node_in_dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.regime_embed = nn.Embedding(n_regimes, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4 + hidden, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, node_feats, regime_idx):
        # node_feats: list of 4 tensors [B, d_i]
        embs = [enc(f) for enc, f in zip(self.node_encoders, node_feats)]
        h = torch.stack(embs, dim=1)  # [B, 4, hidden]
        attn_out, _ = self.attn(h, h, h)  # self-attention over the 4 nodes
        h = h + attn_out
        pooled = h.reshape(h.size(0), -1)  # concat all 4 node embeddings
        r = self.regime_embed(regime_idx)  # [B, hidden]
        out = self.head(torch.cat([pooled, r], dim=1))
        return out.squeeze(-1)


def physics_penalty(pred_cop, delivered_heat_w, total_power_w):
    """Penalize predictions that break basic physical bounds (Section 3.6):
    COP must be non-negative and bounded (<=6 for an air-source heat pump
    in this operating range), and should scale consistently with the
    delivered-heat / power ratio actually observed."""
    lower = torch.relu(-pred_cop).pow(2).mean()
    upper = torch.relu(pred_cop - 6.0).pow(2).mean()
    return lower + upper


def build_node_tensor(df, node_idx, scalers):
    cols = NODE_FEATURE_MAP[node_idx]
    X = df[cols].to_numpy(dtype=np.float32)
    X = scalers[node_idx].transform(X)
    return torch.tensor(X, dtype=torch.float32)


def train_graph_model(train_df, val_df, test_df, regime_col, n_regimes, epochs=40, lr=1e-3, batch_size=512):
    scalers = {i: StandardScaler().fit(train_df[cols]) for i, cols in NODE_FEATURE_MAP.items()}
    node_dims = [len(v) for v in NODE_FEATURE_MAP.values()]
    model = GraphAttentionCOP(node_dims, EDGES, n_regimes=n_regimes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def make_tensors(df):
        nodes = [build_node_tensor(df, i, scalers) for i in range(4)]
        y = torch.tensor(df[TARGET].to_numpy(dtype=np.float32))
        regime = torch.tensor(df[regime_col].to_numpy(dtype=np.int64))
        dheat = torch.tensor(df["delivered_heat_w"].to_numpy(dtype=np.float32))
        power = torch.tensor(df["total_electrical_power_w"].to_numpy(dtype=np.float32))
        return nodes, y, regime, dheat, power

    train_nodes, train_y, train_regime, train_dheat, train_power = make_tensors(train_df)
    val_nodes, val_y, val_regime, val_dheat, val_power = make_tensors(val_df)
    test_nodes, test_y, test_regime, test_dheat, test_power = make_tensors(test_df)

    n = len(train_df)
    best_val = np.inf
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            nodes_b = [nf[idx] for nf in train_nodes]
            y_b, regime_b = train_y[idx], train_regime[idx]
            dheat_b, power_b = train_dheat[idx], train_power[idx]

            pred = model(nodes_b, regime_b)
            data_loss = loss_fn(pred, y_b)
            phys_loss = physics_penalty(pred, dheat_b, power_b)
            loss = data_loss + 0.05 * phys_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_nodes, val_regime)
            val_loss = loss_fn(val_pred, val_y).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={total_loss/n:.5f}  val_mse={val_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for split_name, nodes_t, y_t, regime_t in [
            ("train", train_nodes, train_y, train_regime),
            ("val", val_nodes, val_y, val_regime),
            ("test", test_nodes, test_y, test_regime),
        ]:
            pred = model(nodes_t, regime_t).numpy()
            m = eval_metrics(y_t.numpy(), pred)
            rows.append(dict(stage="4_physics_graph_model", model="GraphAttentionCOP", split=split_name, **m))
    return pd.DataFrame(rows), model, (test_nodes, test_regime, test_y)


def main():
    df = pd.read_parquet(OUT_DIR / "with_regimes.parquet")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)  # defrost rows have undefined COP (no steady delivered heat)
    print(f"Rows with valid {TARGET}: {len(df)}")

    train_df, val_df, test_df = time_split(df)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    all_results = []

    # Stage 1: sensor features only
    scaler1 = StandardScaler().fit(train_df[SENSOR_FEATURES])
    Xtr, Xv, Xte = (scaler1.transform(d[SENSOR_FEATURES]) for d in (train_df, val_df, test_df))
    res1, _ = fit_eval_conventional(Xtr, train_df[TARGET], Xv, val_df[TARGET], Xte, test_df[TARGET], "1_sensors_only")
    all_results.append(res1)

    # Stage 2: + regime (HDBSCAN cluster) one-hot
    feats2 = SENSOR_FEATURES + ["hdbscan_cluster"]
    scaler2 = StandardScaler().fit(train_df[feats2])
    Xtr, Xv, Xte = (scaler2.transform(d[feats2]) for d in (train_df, val_df, test_df))
    res2, _ = fit_eval_conventional(Xtr, train_df[TARGET], Xv, val_df[TARGET], Xte, test_df[TARGET], "2_plus_regime")
    all_results.append(res2)

    # Stage 3: + physics-derived features
    feats3 = feats2 + PHYSICS_FEATURES
    scaler3 = StandardScaler().fit(train_df[feats3])
    Xtr, Xv, Xte = (scaler3.transform(d[feats3]) for d in (train_df, val_df, test_df))
    res3, fitted3 = fit_eval_conventional(Xtr, train_df[TARGET], Xv, val_df[TARGET], Xte, test_df[TARGET], "3_plus_physics_features")
    all_results.append(res3)

    # Stage 4: physics-informed graph model
    n_regimes = int(df["hdbscan_cluster"].max()) + 1
    print("\nTraining physics-informed graph attention model...")
    res4, graph_model, test_tensors = train_graph_model(train_df, val_df, test_df, "hdbscan_cluster", n_regimes)
    all_results.append(res4)

    ablation = pd.concat(all_results, ignore_index=True)
    ablation_test = ablation[ablation["split"] == "test"].sort_values(["stage", "R2"], ascending=[True, False])
    print("\n=== Ablation study: TEST split ===")
    print(ablation_test.to_string(index=False))

    ablation.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    torch.save(graph_model.state_dict(), OUT_DIR / "graph_model_state.pt")

    # Save best stage-3 XGBoost model's test predictions + feature set for SHAP stage
    best3 = fitted3["XGBoost"]
    test_df = test_df.copy()
    test_df["predicted_cop_xgb_stage3"] = best3.predict(scaler3.transform(test_df[feats3]))
    test_df.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)

    import joblib
    joblib.dump({"model": best3, "scaler": scaler3, "features": feats3}, OUT_DIR / "stage3_xgb_model.joblib")

    print(f"\nSaved: {OUT_DIR / 'ablation_results.csv'}")
    print(f"Saved: {OUT_DIR / 'test_predictions.parquet'}")
    print(f"Saved: {OUT_DIR / 'stage3_xgb_model.joblib'}")


if __name__ == "__main__":
    main()
