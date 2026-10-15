"""
CCHP Digital Twin - UI prototype on the v3 simulated dataset (R-454B + humidity).

Run from the project folder:
    streamlit run Dashboard/app.py

Reads the pipeline outputs in Outputs_v3/ (no retraining). Everything shown is
SIMULATED data; swap DATA_DIR / the loaders when lab or field data arrives, as
long as the column names are kept.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent.parent / "Outputs_v3"
STEP = timedelta(minutes=5)

# ---------------------------------------------------------------- labels/colors
SENSORS = {  # column -> (label, unit, group)
    "outdoor_temp_c": ("Outdoor temperature", "°C", "Weather"),
    "outdoor_rh_pct": ("Outdoor relative humidity", "%", "Weather"),
    "outdoor_dewpoint_c": ("Outdoor dew point", "°C", "Weather"),
    "humidity_ratio_g_kg": ("Humidity ratio", "g/kg", "Weather"),
    "return_air_temp_c": ("Return air temperature", "°C", "Air side"),
    "supply_air_temp_c": ("Supply air temperature", "°C", "Air side"),
    "airflow_cfm": ("Airflow", "CFM", "Air side"),
    "high_side_pressure_psig": ("High-side pressure", "psig", "Refrigerant"),
    "suction_pressure_psig": ("Suction pressure", "psig", "Refrigerant"),
    "liquid_line_temp_c": ("Liquid line temperature", "°C", "Refrigerant"),
    "suction_line_temp_c": ("Suction line temperature", "°C", "Refrigerant"),
    "discharge_line_temp_c": ("Discharge line temperature", "°C", "Refrigerant"),
    "top_of_outdoor_unit_temp_c": ("Outdoor coil (top of unit) temp.", "°C", "Refrigerant"),
    "superheat_k": ("Superheat", "K", "Derived (physics)"),
    "subcooling_k": ("Subcooling", "K", "Derived (physics)"),
    "pressure_ratio": ("Pressure ratio", "–", "Derived (physics)"),
    "temp_lift_k": ("Temperature lift", "K", "Derived (physics)"),
    "dewpoint_minus_coil_k": ("Dew point − coil temp.", "K", "Derived (physics)"),
    "minutes_since_defrost": ("Minutes since last defrost", "min", "Derived (physics)"),
    "carnot_cop": ("Carnot COP limit", "–", "Derived (physics)"),
    "compressor_frequency_hz": ("Compressor frequency", "Hz", "Equipment"),
    "eev_position_pct": ("EEV position", "%", "Equipment"),
    "electrical_power_w": ("Heat pump electrical power", "W", "Energy"),
    "supplemental_heat_power_w": ("Strip (backup) heat power", "W", "Energy"),
    "total_electrical_power_w": ("Total electrical power", "W", "Energy"),
    "delivered_heat_w": ("Delivered heat", "W", "Energy"),
    "measured_cop": ("Measured COP", "–", "Energy"),
}


def lab(col):
    name, unit, _ = SENSORS.get(col, (col, "", ""))
    return f"{name} ({unit})" if unit and unit != "–" else name


# Reference categorical palette (light mode), fixed order - colour follows the
# entity, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SITE_COLORS = {"Kamloops": SERIES[0], "Williams Lake": SERIES[1], "Prince George": SERIES[2]}
COARSE_COLORS = {"heating": SERIES[0], "backup_heat": SERIES[1], "defrost": SERIES[2],
                 "transient": SERIES[3], "off": "#9a9893"}
REGIME_COLORS = {"steady_heating": SERIES[0], "low_load_cycling": SERIES[1], "max_capacity": SERIES[2],
                 "supplemental_heat": SERIES[3], "startup_transient": SERIES[4], "defrost": SERIES[5],
                 "defrost_recovery": SERIES[6], "off_standby": "#9a9893"}
MODEL_COLORS = {"XGBoost": SERIES[0], "Graph attention": SERIES[1]}
MEASURED_COLOR = "#0b0b0b"
GRID = "#e4e3df"

PRESET_WINDOWS = {"1 day": timedelta(days=1), "3 days": timedelta(days=3), "1 week": timedelta(weeks=1),
                  "1 month": timedelta(days=30), "Season": None}

st.set_page_config(page_title="CCHP Digital Twin – Prototype", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading simulated dataset…")
def load_data():
    full = pd.read_parquet(DATA_DIR / "with_regimes.parquet")
    full["defrost_int"] = full["defrost_flag"].astype(int)
    pred = pd.read_parquet(DATA_DIR / "test_predictions.parquet",
                           columns=["timestamp", "site", "predicted_cop_xgb", "predicted_cop_graph"])
    conf = pd.read_parquet(DATA_DIR / "conformal_predictions.parquet",
                           columns=["timestamp", "site", "cop_lower_global", "cop_upper_global",
                                    "cop_lower_per_hdbscan_regime", "cop_upper_per_hdbscan_regime"])
    full = full.merge(pred, on=["timestamp", "site"], how="left").merge(conf, on=["timestamp", "site"], how="left")
    full["in_test"] = full["predicted_cop_xgb"].notna()

    # Re-index every site to the full 5-minute grid so missing intervals show
    # as gaps in the lines instead of being bridged.
    parts = []
    for site, g in full.groupby("site"):
        grid = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="5min")
        g = g.set_index("timestamp").reindex(grid)
        g["site"] = site
        g["missing"] = g["outdoor_temp_c"].isna()
        parts.append(g.rename_axis("timestamp").reset_index())
    full = pd.concat(parts, ignore_index=True)
    full["in_test"] = full["predicted_cop_xgb"].notna()

    tables = {
        "loso": pd.read_csv(DATA_DIR / "loso_results.csv"),
        "shap": pd.read_csv(DATA_DIR / "shap_global_importance.csv"),
    }
    return full, tables


df, tables = load_data()
T0, T1 = df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime()
TEST_START = df.loc[df["in_test"], "timestamp"].min().to_pydatetime()


# ---------------------------------------------------------------- time controls
def clamp(start, end):
    span = end - start
    if start < T0:
        start, end = T0, min(T0 + span, T1)
    if end > T1:
        start, end = max(T1 - span, T0), T1
    return start, end


if "win" not in st.session_state:
    st.session_state.win = (T1 - timedelta(weeks=1), T1)


def set_preset(name):
    length = PRESET_WINDOWS[name]
    if length is None:
        st.session_state.win = (T0, T1)
    else:
        end = st.session_state.win[1]
        st.session_state.win = clamp(end - length, end)


def shift(direction):
    s, e = st.session_state.win
    span = e - s
    st.session_state.win = clamp(s + direction * span, e + direction * span)


def jump_to_test():
    st.session_state.win = (TEST_START, T1)


with st.sidebar:
    st.header("Filters")
    sites = st.multiselect("Site", list(SITE_COLORS), default=["Prince George"],
                           help="Three simulated BC sites; same heat pump, different climate.")
    if not sites:
        st.warning("Pick at least one site.")
        st.stop()

    st.subheader("Time period")
    cols = st.columns(3)
    for i, name in enumerate(PRESET_WINDOWS):
        cols[i % 3].button(name, on_click=set_preset, args=(name,), width="stretch")
    c1, c2 = st.columns(2)
    c1.button("◀ Earlier", on_click=shift, args=(-1,), width="stretch")
    c2.button("Later ▶", on_click=shift, args=(1,), width="stretch")
    st.button("Jump to model test period", on_click=jump_to_test, width="stretch",
              help="Predictions exist only for the last ~15 % of each site's season (held-out test data).")
    st.slider("Custom range (drag either end)", min_value=T0, max_value=T1, step=timedelta(hours=1),
              format="MMM D, HH:mm", key="win")

    st.subheader("Display")
    resolution = st.radio("Time resolution", ["Auto", "5-minute (raw)", "Hourly mean", "Daily mean"],
                          horizontal=False,
                          help="Auto = raw 5-min data for windows up to 10 days, hourly means beyond.")
    st.caption("All data are **simulated** (v3, R-454B + humidity). Not for final performance claims.")

start, end = st.session_state.win
win = df[(df["timestamp"] >= start) & (df["timestamp"] <= end) & df["site"].isin(sites)]
span = end - start
rule = {"5-minute (raw)": None, "Hourly mean": "1h", "Daily mean": "1D"}.get(
    resolution, None if span <= timedelta(days=10) else "1h")


def resample(d, cols):
    if rule is None:
        return d[["timestamp", "site"] + cols]
    return (d.set_index("timestamp").groupby("site")[cols].resample(rule).mean().reset_index())


def style(fig, height=None):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), font=dict(size=13))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def defrost_spans(d):
    """Contiguous defrost intervals of one site -> list of (start, end)."""
    flag = d["defrost_int"].fillna(0).to_numpy() == 1
    if not flag.any():
        return []
    t = d["timestamp"].to_numpy()
    edges = np.diff(np.r_[0, flag.astype(int), 0])
    return [(t[a], t[b - 1] + np.timedelta64(5, "m")) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


# ---------------------------------------------------------------- header + KPIs
st.title("Cold-Climate Heat Pump – Digital Twin Prototype")
st.markdown(f"**{', '.join(sites)}** · {start:%b %d %Y %H:%M} → {end:%b %d %Y %H:%M} "
            f"({span.days} d {span.seconds // 3600} h) · resolution: {rule or '5 min'}")

obs = win[~win["missing"]]
k = st.columns(6)
energy_in = obs["total_electrical_power_w"].sum() * 5 / 60 / 1000  # kWh
heat_out = obs["delivered_heat_w"].sum() * 5 / 60 / 1000
k[0].metric("Mean outdoor temp.", f"{obs['outdoor_temp_c'].mean():.1f} °C",
            help=f"Min {obs['outdoor_temp_c'].min():.1f} °C")
k[1].metric("Mean outdoor RH", f"{obs['outdoor_rh_pct'].mean():.0f} %")
k[2].metric("System COP (window)", f"{heat_out / energy_in:.2f}" if energy_in else "–",
            help="Delivered heat ÷ total electrical energy (incl. strip heat) over the window.")
k[3].metric("Electricity used", f"{energy_in:,.0f} kWh")
n_def = sum(len(defrost_spans(g)) for _, g in win.groupby("site"))
k[4].metric("Defrost cycles", f"{n_def}")
k[5].metric("Data coverage", f"{100 * (1 - win['missing'].mean()):.1f} %" if len(win) else "–",
            help="Share of 5-min intervals with a reading (missing intervals are flagged, not filled).")

tab_ts, tab_rel, tab_model, tab_about = st.tabs(
    ["📈 Sensor readings", "🔗 Sensor relationships", "🎯 Model accuracy", "ℹ️ About the data"])

# ---------------------------------------------------------------- tab 1: time series
with tab_ts:
    groups = {}
    for c, (_, _, g) in SENSORS.items():
        groups.setdefault(g, []).append(c)
    q1, q2 = st.columns([3, 1])
    chosen = q1.multiselect(
        "Sensors to plot (one panel each, shared time axis)", list(SENSORS), format_func=lab,
        default=["outdoor_temp_c", "outdoor_rh_pct", "suction_pressure_psig", "compressor_frequency_hz",
                 "electrical_power_w", "measured_cop"])
    group_pick = q2.selectbox("…or add a whole group", ["—"] + list(groups))
    if group_pick != "—":
        chosen = list(dict.fromkeys(chosen + groups[group_pick]))
    shade = st.toggle("Shade defrost periods", value=span <= timedelta(days=14),
                      help="Off by default for long windows (hundreds of cycles).")

    if chosen:
        d = resample(win, chosen + ["defrost_int"])
        fig = make_subplots(rows=len(chosen), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                            subplot_titles=[lab(c) for c in chosen])
        for i, c in enumerate(chosen, 1):
            for s in sites:
                ds = d[d["site"] == s]
                fig.add_trace(go.Scattergl(x=ds["timestamp"], y=ds[c], mode="lines", name=s,
                                           line=dict(color=SITE_COLORS[s], width=1.5),
                                           legendgroup=s, showlegend=(i == 1),
                                           hovertemplate=f"{s}: %{{y:.2f}}<extra></extra>"), row=i, col=1)
        if shade:
            for s in sites:
                for a, b in defrost_spans(win[win["site"] == s]):
                    fig.add_vrect(x0=a, x1=b, fillcolor=SITE_COLORS[s], opacity=0.12, line_width=0)
        style(fig, height=max(300, 170 * len(chosen)))
        fig.update_annotations(font_size=13, x=0, xanchor="left")
        st.plotly_chart(fig, width="stretch")
        if shade:
            st.caption("Shaded bands = defrost cycles (unit reverses, COP drops ≤ 0).")

        with st.expander("Summary statistics for the selected window"):
            summ = win.groupby("site")[chosen].agg(["mean", "min", "max", "std"]).T.round(2)
            summ.index = [f"{lab(a)} · {b}" for a, b in summ.index]
            st.dataframe(summ, width="stretch")
        st.download_button("Download this window as CSV",
                           win[["timestamp", "site"] + chosen].to_csv(index=False).encode(),
                           file_name=f"cchp_window_{start:%Y%m%d}_{end:%Y%m%d}.csv")

    st.subheader("Operating regime timeline")
    st.caption("Simulator ground-truth regime (top of each site) vs. regime discovered by HDBSCAN from sensors only.")
    reg = win.dropna(subset=["_truth_regime"])
    if len(reg):
        rows = []
        for s in sites:
            r = reg[reg["site"] == s]
            rows.append(r.assign(track=f"{s} · true regime", label=r["_truth_regime"]))
            rows.append(r.assign(track=f"{s} · HDBSCAN cluster",
                                 label="cluster " + r["hdbscan_cluster"].astype(int).astype(str)))
        rr = pd.concat(rows)
        cluster_colors = {"cluster -1": "#9a9893", "cluster 0": SERIES[7], "cluster 1": SERIES[6],
                          "cluster 2": SERIES[4], "cluster 3": SERIES[0]}
        figr = px.scatter(rr, x="timestamp", y="track", color="label", render_mode="webgl",
                          color_discrete_map={**REGIME_COLORS, **cluster_colors},
                          category_orders={"label": list(REGIME_COLORS) + list(cluster_colors)})
        figr.update_traces(marker=dict(symbol="line-ns-open", size=14, line=dict(width=2)))
        style(figr, height=110 + 70 * len(sites) * 2).update_layout(hovermode="closest", legend_title_text="")
        figr.update_yaxes(title=None)
        st.plotly_chart(figr, width="stretch")

# ---------------------------------------------------------------- tab 2: relationships
PRESETS = {
    "COP vs outdoor temperature": ("outdoor_temp_c", "measured_cop"),
    "Suction pressure vs outdoor temperature": ("outdoor_temp_c", "suction_pressure_psig"),
    "Power vs compressor frequency": ("compressor_frequency_hz", "electrical_power_w"),
    "Frosting: dew point − coil vs time since defrost": ("minutes_since_defrost", "dewpoint_minus_coil_k"),
    "Pressure ratio vs temperature lift": ("temp_lift_k", "pressure_ratio"),
    "Discharge temp vs pressure ratio": ("pressure_ratio", "discharge_line_temp_c"),
}


def set_xy(name):
    st.session_state.x_col, st.session_state.y_col = PRESETS[name]


with tab_rel:
    st.markdown("**Quick views** – physics relationships between sensors")
    pc = st.columns(len(PRESETS))
    for i, name in enumerate(PRESETS):
        pc[i].button(name, on_click=set_xy, args=(name,), width="stretch")
    st.session_state.setdefault("x_col", "outdoor_temp_c")
    st.session_state.setdefault("y_col", "measured_cop")
    a, b, c = st.columns(3)
    xc = a.selectbox("X axis", list(SENSORS), format_func=lab, key="x_col")
    yc = b.selectbox("Y axis", list(SENSORS), format_func=lab, key="y_col")
    color_opts = {"Operating regime (simulator truth)": "_truth_regime_coarse", "Site": "site",
                  "HDBSCAN cluster": "hdbscan_label", "Outdoor RH (%)": "outdoor_rh_pct",
                  "Outdoor temperature (°C)": "outdoor_temp_c"}
    color_by = c.selectbox("Colour points by", list(color_opts))
    hide_off = st.checkbox("Hide 'off / standby' intervals", value=True)

    d = win[~win["missing"]].copy()
    if hide_off:
        d = d[d["_truth_regime_coarse"] != "off"]
    d["hdbscan_label"] = "cluster " + d["hdbscan_cluster"].astype(int).astype(str)
    d = d.dropna(subset=[xc, yc])
    MAXPTS = 25_000
    sampled = len(d) > MAXPTS
    dp = d.sample(MAXPTS, random_state=0) if sampled else d

    left, right = st.columns([3, 2])
    with left:
        ccol = color_opts[color_by]
        kw = dict(color=ccol, render_mode="webgl", opacity=0.55,
                  hover_data={"timestamp": True, "site": True, "_truth_regime": True})
        if ccol in ("outdoor_rh_pct", "outdoor_temp_c"):
            kw["color_continuous_scale"] = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
        else:
            kw["color_discrete_map"] = {**COARSE_COLORS, **SITE_COLORS,
                                        "cluster -1": "#9a9893", "cluster 0": SERIES[7], "cluster 1": SERIES[6],
                                        "cluster 2": SERIES[4], "cluster 3": SERIES[0]}
        figs = px.scatter(dp, x=xc, y=yc, labels={xc: lab(xc), yc: lab(yc), **{v: k for k, v in color_opts.items()}},
                          **kw)
        figs.update_traces(marker=dict(size=5))
        style(figs, height=520).update_layout(hovermode="closest")
        st.plotly_chart(figs, width="stretch")
        r = d[[xc, yc]].corr().iloc[0, 1] if len(d) > 2 else np.nan
        st.caption(f"Pearson r = **{r:.2f}** over {len(d):,} intervals"
                   + (f" (plot shows a random {MAXPTS:,})" if sampled else "")
                   + ". Click legend items to hide/show groups.")

    with right:
        default_corr = ["outdoor_temp_c", "outdoor_rh_pct", "suction_pressure_psig", "high_side_pressure_psig",
                        "discharge_line_temp_c", "compressor_frequency_hz", "electrical_power_w",
                        "superheat_k", "dewpoint_minus_coil_k", "measured_cop"]
        corr_cols = st.multiselect("Correlation matrix sensors", list(SENSORS), default=default_corr,
                                   format_func=lambda c: SENSORS[c][0])
        if len(corr_cols) >= 2:
            cm = d[corr_cols].corr()
            short = [SENSORS[c][0] for c in corr_cols]
            figc = go.Figure(go.Heatmap(
                z=cm.values, x=short, y=short, zmin=-1, zmax=1,
                colorscale=[[0, "#c0392b"], [0.5, "#f0efec"], [1, "#184f95"]],  # diverging, grey midpoint
                text=np.round(cm.values, 2), texttemplate="%{text}", textfont_size=10,
                hovertemplate="%{y} × %{x}: r = %{z:.2f}<extra></extra>"))
            style(figc, height=520).update_layout(hovermode="closest")
            figc.update_yaxes(autorange="reversed")
            st.plotly_chart(figc, width="stretch")

# ---------------------------------------------------------------- tab 3: model accuracy
MODELS = {"XGBoost": "predicted_cop_xgb", "Graph attention": "predicted_cop_graph"}


def metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan, bias=np.nan, n=len(y))
    e = p - y
    m = np.abs(y) >= 1
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
                MAPE=float(100 * np.mean(np.abs(e[m] / y[m]))) if m.any() else np.nan,
                bias=float(np.mean(e)), n=int(len(y)))


with tab_model:
    t = win[win["in_test"]]
    if t.empty:
        st.info(f"No model predictions in this window. Predictions exist only for the held-out **test period** "
                f"(from {TEST_START:%b %d %Y} to {T1:%b %d %Y}; last ~15 % of each site's season, never seen in training).")
        st.button("Jump to model test period ", on_click=jump_to_test)
    else:
        m1, m2, m3 = st.columns(3)
        model = m1.radio("Model", list(MODELS), horizontal=True)
        target_name = m2.radio("Score against", ["Measured COP", "True COP (simulator)"], horizontal=True,
                               help="Measured COP is noisy (sensor error). True COP is only available because the data are simulated.")
        calib = m3.radio("90 % interval calibration", ["Global", "Per HDBSCAN regime"], horizontal=True)
        target = "measured_cop" if target_name == "Measured COP" else "_truth_cop_hp"
        pcol = MODELS[model]
        lo, hi = (("cop_lower_global", "cop_upper_global") if calib == "Global"
                  else ("cop_lower_per_hdbscan_regime", "cop_upper_per_hdbscan_regime"))
        if len(t) < len(win[~win["missing"]]):
            st.caption(f"Scoring {len(t):,} test-period intervals inside the selected window.")

        mt = metrics(t[target].to_numpy(float), t[pcol].to_numpy(float))
        tt = t.dropna(subset=[target, lo, hi])
        cover = float(((tt[target] >= tt[lo]) & (tt[target] <= tt[hi])).mean()) if len(tt) else np.nan
        width = float((tt[hi] - tt[lo]).mean()) if len(tt) else np.nan
        kk = st.columns(6)
        kk[0].metric("RMSE", f"{mt['RMSE']:.3f}")
        kk[1].metric("MAE", f"{mt['MAE']:.3f}")
        kk[2].metric("R²", f"{mt['R2']:.3f}")
        kk[3].metric("MAPE", f"{mt['MAPE']:.1f} %", help="Only where |COP| ≥ 1 (excludes defrost near zero).")
        kk[4].metric("90 % interval coverage", f"{100 * cover:.1f} %",
                     delta=f"{100 * cover - 90:+.1f} pts vs target", delta_color="off",
                     help="Intervals come from the XGBoost conformal stage.")
        kk[5].metric("Mean interval width", f"{width:.2f} COP")

        # time series: measured vs predicted with band
        tp = t.sort_values("timestamp")
        figm = make_subplots(rows=len(sites), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                             subplot_titles=sites)
        for i, s in enumerate(sites, 1):
            g = tp[tp["site"] == s]
            # break lines at gaps
            g = g.set_index("timestamp").reindex(pd.date_range(g["timestamp"].min(), g["timestamp"].max(),
                                                               freq="5min")).rename_axis("timestamp").reset_index() if len(g) else g
            figm.add_trace(go.Scatter(x=g["timestamp"], y=g[hi], line=dict(width=0), showlegend=False,
                                      hoverinfo="skip"), row=i, col=1)
            figm.add_trace(go.Scatter(x=g["timestamp"], y=g[lo], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                      line=dict(width=0), name="90 % prediction interval", showlegend=(i == 1),
                                      hoverinfo="skip"), row=i, col=1)
            figm.add_trace(go.Scatter(x=g["timestamp"], y=g[target], name=target_name, showlegend=(i == 1),
                                      line=dict(color=MEASURED_COLOR, width=1.5),
                                      hovertemplate=f"{target_name}: %{{y:.2f}}<extra></extra>"), row=i, col=1)
            figm.add_trace(go.Scatter(x=g["timestamp"], y=g[pcol], name=f"{model} prediction", showlegend=(i == 1),
                                      line=dict(color=MODEL_COLORS[model], width=2, dash="dot"),
                                      hovertemplate=f"{model}: %{{y:.2f}}<extra></extra>"), row=i, col=1)
            figm.update_yaxes(title_text="COP", row=i, col=1)
        style(figm, height=260 * len(sites) + 60)
        figm.update_annotations(x=0, xanchor="left")
        st.subheader("Measured vs predicted COP")
        st.plotly_chart(figm, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Parity plot")
            tq = t.dropna(subset=[target, pcol])
            figp = px.scatter(tq, x=target, y=pcol, color="_truth_regime_coarse", render_mode="webgl", opacity=0.5,
                              color_discrete_map=COARSE_COLORS,
                              labels={target: target_name, pcol: f"{model} predicted COP",
                                      "_truth_regime_coarse": "Regime"})
            lim = [min(tq[target].min(), tq[pcol].min()), max(tq[target].max(), tq[pcol].max())]
            figp.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#8a8984", dash="dash", width=1),
                                      name="perfect prediction"))
            figp.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
            style(figp, height=450).update_layout(hovermode="closest")
            st.plotly_chart(figp, width="stretch")
        with c2:
            st.subheader("Error by operating regime")
            rows = []
            for mname, mc in MODELS.items():
                for rg, g in t.groupby("_truth_regime"):
                    m = metrics(g[target].to_numpy(float), g[mc].to_numpy(float))
                    rows.append(dict(model=mname, regime=rg, **m))
            er = pd.DataFrame(rows)
            er = er[er["n"] >= 3]
            figb = px.bar(er, x="regime", y="RMSE", color="model", barmode="group",
                          color_discrete_map=MODEL_COLORS, hover_data={"n": True, "MAE": ":.3f", "bias": ":.3f"},
                          category_orders={"regime": er.groupby("regime")["n"].sum().sort_values(ascending=False).index.tolist()})
            figb.update_traces(marker_line_width=0)
            style(figb, height=450).update_layout(hovermode="closest", bargap=0.3, bargroupgap=0.08,
                                                  xaxis_title=None, yaxis_title="RMSE (COP)")
            st.plotly_chart(figb, width="stretch")
            st.caption("Defrost / transient regimes are rare and fast-changing, so errors are largest there.")

        st.subheader("Where does the model miss? Residual vs a sensor")
        rx = st.selectbox("Residual against", ["outdoor_temp_c", "outdoor_rh_pct", "minutes_since_defrost",
                                               "compressor_frequency_hz", "dewpoint_minus_coil_k"], format_func=lab)
        tr = t.assign(residual=t[pcol] - t[target]).dropna(subset=["residual", rx])
        figr = px.scatter(tr, x=rx, y="residual", color="site", color_discrete_map=SITE_COLORS, opacity=0.4,
                          render_mode="webgl", labels={rx: lab(rx), "residual": "Prediction − actual (COP)"})
        figr.add_hline(y=0, line_color="#8a8984", line_width=1)
        figr.update_traces(marker=dict(size=4))
        style(figr, height=380).update_layout(hovermode="closest")
        st.plotly_chart(figr, width="stretch")

        with st.expander("Whole-season context: leave-one-site-out and feature importance"):
            e1, e2 = st.columns(2)
            e1.markdown("**Leave-one-site-out results** (train on two sites, test on the third)")
            e1.dataframe(tables["loso"], width="stretch", height=300)
            sh = tables["shap"].copy()
            e2.markdown("**Global SHAP importance (XGBoost)**")
            num = sh.select_dtypes("number").columns[0]
            namecol = sh.columns[0]
            sh = sh.sort_values(num).tail(15)
            fsh = px.bar(sh, x=num, y=namecol, orientation="h", color_discrete_sequence=[SERIES[0]])
            style(fsh, height=420).update_layout(hovermode="closest", yaxis_title=None, xaxis_title="mean |SHAP|")
            e2.plotly_chart(fsh, width="stretch")

# ---------------------------------------------------------------- tab 4: about
with tab_about:
    st.markdown(f"""
**What this is.** An interactive prototype of the digital-twin interface, running on the **v3 simulated dataset**
(R-454B variable-speed cold-climate heat pump, outdoor humidity included) while the physical sensor set-up is prepared.

* **Sites:** Kamloops, Williams Lake, Prince George · winter {T0:%b %d %Y} – {T1:%b %d %Y} · 5-minute readings.
* **Sensor readings** – any combination of the 20+ sensor and physics-derived channels on a shared time axis, with defrost cycles shaded.
* **Sensor relationships** – scatter of any two channels plus a correlation matrix for the selected window.
* **Model accuracy** – XGBoost and graph-attention COP models with conformal 90 % intervals, scored on the held-out
  test period ({TEST_START:%b %d} → {T1:%b %d}); predictions were produced by the pipeline in `Code/` (stages 3–5).
* Missing 5-minute intervals (~1.4–2.6 % per site) are shown as gaps, never filled.
* Regime colours in scatter plots use the 5 coarse simulator groups; click legend entries to isolate one.

**Caveat:** synthetic data – per proposal §3.2, it must not support final performance claims.
When lab/field data arrive, regenerate `Outputs_v3/`-style files with the same columns and point `DATA_DIR` at them.
""")
