from pathlib import Path
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar Plant Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"

HOURLY_FILE = DATA_DIR / "plant1_hourly.csv"
OPENMETEO_FILE = DATA_DIR / "open_meteo_weather.csv"
MERGED_WEATHER_FILE = DATA_DIR / "plant1_hourly_openmeteo.csv"

SET_B_WEIGHTS = WEIGHTS_DIR / "set_B_weights.npz"
SET_A_WEIGHTS = WEIGHTS_DIR / "set_A_weights.npz"
SUMMARY_FILE = RESULTS_DIR / "train_eval_summary.json"
TABLE2_FILE = RESULTS_DIR / "table2_test_rmse.csv"
TABLE3_FILE = RESULTS_DIR / "table3_theta_set_a.csv"
TRAIN_SCRIPT = ROOT_DIR / "src" / "train_eval.py"


# ============================================================
# CSS
# ============================================================

CSS_FILE = APP_DIR / "style.css"
if CSS_FILE.exists():
    st.markdown(
        f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value, decimals=1):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "—"


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


@st.cache_data
def load_hourly():
    df = safe_read_csv(HOURLY_FILE)
    if df is None:
        return None
    df = normalize_columns(df)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data
def load_weather():
    df = safe_read_csv(MERGED_WEATHER_FILE)
    if df is None:
        df = safe_read_csv(OPENMETEO_FILE)
    if df is None:
        return None
    df = normalize_columns(df)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data
def load_summary():
    return load_json(SUMMARY_FILE)


@st.cache_data
def load_rmse():
    return safe_read_csv(TABLE2_FILE)


@st.cache_data
def load_theta():
    return safe_read_csv(TABLE3_FILE)


# ============================================================
# MODEL LOADING / PREDICTION
# ============================================================

@st.cache_resource
def load_model():
    if not SET_B_WEIGHTS.exists():
        return None
    try:
        with np.load(SET_B_WEIGHTS, allow_pickle=True) as w:
            return {
                "theta_normal": np.asarray(w["theta_normal"], dtype=float),
                "theta_batch_gd": np.asarray(w["theta_batch_gd"], dtype=float),
                "theta_sgd": np.asarray(w["theta_sgd"], dtype=float),
                "feature_names": w["feature_names"].astype(str).tolist(),
                "scaler_mean": np.asarray(w["scaler_mean"], dtype=float),
                "scaler_std": np.asarray(w["scaler_std"], dtype=float),
                "cont_cols": w["cont_cols"].astype(str).tolist(),
            }
    except Exception as exc:
        st.error(f"Could not load Set B weights: {exc}")
        return None


def build_input_vector(model, hour, sw, temp, cloud):
    raw = np.asarray([sw, temp, cloud], dtype=float)
    means = model["scaler_mean"]
    stds = model["scaler_std"]
    if len(means) != 3 or len(stds) != 3:
        raise ValueError("Saved Set B scaler must contain exactly three continuous features.")
    if np.any(stds == 0):
        raise ValueError("Saved Set B scaler contains a zero standard deviation.")

    z = (raw - means) / stds
    sin_h = np.sin(2 * np.pi * hour / 24.0)
    cos_h = np.cos(2 * np.pi * hour / 24.0)
    return np.array([1.0, z[0], z[1], z[2], sin_h, cos_h], dtype=float), raw, z, sin_h, cos_h


def predict_set_b(model, hour, sw, temp, cloud, solver="normal"):
    if model is None:
        return None

    # A solar plant cannot generate AC power during the zero-output
    # portion of the 24-hour profile. Apply this rule in the prediction
    # function itself so every scenario/button behaves consistently.
    if int(hour) < 6 or int(hour) >= 18 or float(sw) <= 0:
        return 0.0

    x, *_ = build_input_vector(model, hour, sw, temp, cloud)
    theta_key = {
        "normal": "theta_normal",
        "batch": "theta_batch_gd",
        "sgd": "theta_sgd",
    }[solver]
    prediction = float(x @ model[theta_key])
    return max(0.0, prediction)


def scenario_values(name):
    return {
        "Solar Noon": {"hour": 12, "sw": 800.0, "temp": 32.0, "cloud": 10},
        "Overcast": {"hour": 12, "sw": 250.0, "temp": 28.0, "cloud": 90},
        "Morning": {"hour": 9, "sw": 450.0, "temp": 28.0, "cloud": 20},
        "Night": {"hour": 23, "sw": 0.0, "temp": 24.0, "cloud": 5},
    }[name]


def operating_period(hour):
    """Return the solar operating period for a selected 24-hour clock hour."""
    hour = int(hour)
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 20:
        return "Evening"
    return "Night"


def is_solar_daylight(hour):
    """Plant output is zero outside the daylight window used by the profile."""
    hour = int(hour)
    return 6 <= hour < 18


def render_period_markers(hour):
    """Render a clean four-part period guide without raw HTML/code blocks."""
    current = operating_period(hour)
    periods = [
        ("Morning", "06:00–11:59"),
        ("Afternoon", "12:00–16:59"),
        ("Evening", "17:00–19:59"),
        ("Night", "20:00–05:59"),
    ]

    cols = st.columns(4, gap="small")
    for col, (name, time_range) in zip(cols, periods):
        with col:
            if name == current:
                st.markdown("**●**", unsafe_allow_html=False)
            else:
                st.markdown("○", unsafe_allow_html=False)
            st.markdown(f"**{name}**")
            st.caption(time_range)


hourly = load_hourly()
weather = load_weather()
summary = load_summary()
rmse_table = load_rmse()
theta_table = load_theta()
model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">☀ SOLAR INTELLIGENCE</div>
            <div class="sidebar-brand-subtitle">Applied Machine Learning · AI4003</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Prediction Dashboard",
            "Power Analytics",
            "Weather Intelligence",
            "ML Performance",
            "Model Inspector",
            "Training Graphs",
            "Model Training",
            "Dataset",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="info-box">
            <b>PLANT 01</b><br>
            Solar Power Generation Dataset<br>
            India · 2020<br><br>
            <b>Prediction Model</b><br>
            Linear Regression · Set B<br>
            Public Weather + Time Features
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        st.markdown('<span class="model-badge">MODEL LOADED</span>', unsafe_allow_html=True)
    else:
        st.warning("Set B model weights are not available. Train the model first.")


# ============================================================
# PREDICTION DASHBOARD — REFERENCE UI
# ============================================================

if page == "Prediction Dashboard":
    st.markdown(
        """
        <div class="hero compact-hero">
            <div class="status-pill"><span class="status-dot"></span> MODEL PREDICTION ONLINE</div>
            <div class="hero-title">Solar Plant Prediction Dashboard</div>
            <div class="hero-subtitle">
                Enter public-weather conditions and time to obtain an AC-power prediction
                directly from the trained Set B linear-regression model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenarios = ["Solar Noon", "Overcast", "Morning", "Night"]
    selected_scenario = st.radio(
        "Reference Scenarios",
        scenarios,
        horizontal=True,
        key="scenario_selector",
    )

    # Presets only populate UI inputs. They are NOT used to train the model.
    preset = scenario_values(selected_scenario)
    if st.session_state.get("last_scenario") != selected_scenario:
        st.session_state["prediction_hour"] = preset["hour"]
        st.session_state["prediction_sw"] = preset["sw"]
        st.session_state["prediction_temp"] = preset["temp"]
        st.session_state["prediction_cloud"] = preset["cloud"]
        st.session_state["last_scenario"] = selected_scenario

    left, right = st.columns([1.05, 1.65], gap="large")

    with left:
        st.markdown('<div class="reference-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">MODEL INPUT PARAMETERS</div>', unsafe_allow_html=True)
        st.caption("Set B · Public Weather")

        hour = st.slider(
            "Hour of Day (24-Hour Cycle)", 0, 23,
            value=int(st.session_state.get("prediction_hour", preset["hour"])),
            step=1, key="prediction_hour",
        )
        st.markdown(f'<div class="input-readout">{hour:02d}:00</div>', unsafe_allow_html=True)
        render_period_markers(hour)
        st.caption("Period guide: Morning 06:00–11:59 · Afternoon 12:00–16:59 · Evening 17:00–19:59 · Night 20:00–05:59. Output is fixed at 0 kW outside 06:00–17:59.")

        sw = st.number_input(
            "Radiation (W/m²)",
            min_value=0.0, max_value=1500.0,
            value=float(st.session_state.get("prediction_sw", preset["sw"])),
            step=10.0, key="prediction_sw",
        )

        temp = st.number_input(
            "Temperature 2m (°C)",
            min_value=-20.0, max_value=60.0,
            value=float(st.session_state.get("prediction_temp", preset["temp"])),
            step=0.5, key="prediction_temp",
        )

        cloud = st.slider(
            "Cloud Cover (%)", 0, 100,
            value=int(st.session_state.get("prediction_cloud", preset["cloud"])),
            step=1, key="prediction_cloud",
        )

        st.markdown(
            '<div class="input-note">The three continuous inputs are standardized with the <b>training-set mean and standard deviation</b> saved with the model. Hour is represented by sin(hour) and cos(hour).</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    prediction = predict_set_b(model, hour, sw, temp, cloud) if model is not None else None

    with right:
        if prediction is None:
            st.error("Set B model weights are missing. Open Model Training and train the model first.")
        else:
            peak_kw = 0.0
            if hourly is not None and "ac_power" in hourly.columns:
                peak_kw = float(pd.to_numeric(hourly["ac_power"], errors="coerce").max())

            prediction_kw = float(prediction)
            utilization = prediction_kw / peak_kw * 100.0 if peak_kw > 0 else 0.0
            active_mw = prediction_kw / 1000.0

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="reference-metric prediction-metric">
                        <div class="metric-label">PREDICTED AC POWER</div>
                        <div class="reference-value">{prediction_kw:,.0f}<span> kW</span></div>
                        <div class="metric-caption">{active_mw:,.2f} MW Active Generation</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                bar_width = min(max(utilization, 0.0), 100.0)
                st.markdown(
                    f"""
                    <div class="reference-metric">
                        <div class="metric-label">CAPACITY UTILIZATION</div>
                        <div class="reference-value">{utilization:,.1f}<span>%</span></div>
                        <div class="metric-caption">Ref. Peak {peak_kw:,.1f} kW</div>
                        <div class="util-track"><div class="util-fill" style="width:{bar_width:.1f}%"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="reference-metric model-config">
                    <div class="metric-label">MODEL CONFIGURATION</div>
                    <div class="config-main">Set B (Public)</div>
                    <div class="metric-caption">Z-Score Standardized · Normal Equation · 6-parameter θ</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 24-hour profile. The current weather values are kept fixed so that
            # the visible curve isolates the learned time contribution.
            profile_hours = np.arange(24)
            profile_kw = np.array([
                predict_set_b(model, int(h), sw, temp, cloud) for h in profile_hours
            ], dtype=float)

            # predict_set_b applies the same daylight/radiation rule to every
            # hour, so the plotted profile and the displayed prediction agree.

            fig, ax = plt.subplots(figsize=(10, 3.4))
            ax.plot(profile_hours, profile_kw, linewidth=2.6)
            ax.scatter([hour], [prediction_kw], s=65, zorder=5)
            ax.axvline(hour, linestyle=":", linewidth=1.6, alpha=0.7)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_xticklabels(["00h", "06h", "12h", "18h", "23h"])
            ax.set_ylabel("AC Power (kW)")
            ax.set_xlabel("Hour of Day")
            ax.set_title("Estimated Diurnal Generation Profile", loc="left", fontweight="bold")
            ax.grid(alpha=0.18)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Exact transformation shown to make the frontend auditable.
            x, raw, z, sin_h, cos_h = build_input_vector(model, hour, sw, temp, cloud)
            theta = model["theta_normal"]
            contribution = theta * x
            names = model["feature_names"]

            st.markdown(
                '<div class="feature-breakdown-title">Feature Transformation & Weight Vector Breakdown</div>',
                unsafe_allow_html=True,
            )
            st.caption("x₀ = 1 is prepended; continuous inputs use training statistics; final prediction is max(0, xᵀθ).")

            rows = []
            for i, name in enumerate(names):
                if i == 0:
                    raw_value, mean, std, z_value = 1.0, "—", "—", 1.0
                elif i <= 3:
                    j = i - 1
                    raw_value = raw[j]
                    mean = model["scaler_mean"][j]
                    std = model["scaler_std"][j]
                    z_value = z[j]
                elif i == 4:
                    raw_value, mean, std, z_value = sin_h, "—", "—", sin_h
                else:
                    raw_value, mean, std, z_value = cos_h, "—", "—", cos_h
                rows.append({
                    "Feature": name,
                    "Raw Value": raw_value,
                    "Mean": mean,
                    "Std": std,
                    "Z-Score": z_value,
                    "Weight (θ)": theta[i],
                    "Contribution": contribution[i],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# POWER ANALYTICS
# ============================================================

elif page == "Power Analytics":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> HISTORICAL PLANT DATA</div><div class="hero-title">Power Analytics</div><div class="hero-subtitle">Explore plant-level AC/DC generation and hourly production patterns.</div></div>', unsafe_allow_html=True)

    if hourly is None:
        st.error("data/plant1_hourly.csv was not found. Run prepare.py first.")
    else:
        data = hourly.dropna(subset=["datetime"]).copy()
        min_date, max_date = data["datetime"].min().date(), data["datetime"].max().date()
        selected = st.date_input("Analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(selected, tuple) and len(selected) == 2:
            filtered = data[(data["datetime"].dt.date >= selected[0]) & (data["datetime"].dt.date <= selected[1])].copy()
        else:
            filtered = data.copy()

        peak = float(filtered["ac_power"].max())
        average = float(filtered["ac_power"].mean())
        energy_kwh = float(filtered["ac_power"].sum())
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">PEAK AC POWER</div><div class="metric-value">{peak:,.1f}<span class="metric-unit">kW</span></div><div class="metric-caption">Maximum hourly plant output</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">AVERAGE AC POWER</div><div class="metric-value">{average:,.1f}<span class="metric-unit">kW</span></div><div class="metric-caption">Mean across selected period</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">HOURLY POWER SUM</div><div class="metric-value">{energy_kwh/1000:,.1f}<span class="metric-unit">MWh*</span></div><div class="metric-caption">*Sum of hourly kW readings</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Plant Production Profile</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.plot(filtered["datetime"], filtered["ac_power"], linewidth=2.3)
        ax.set_xlabel("Datetime")
        ax.set_ylabel("AC Power (kW)")
        ax.set_title("Plant AC Power")
        ax.grid(alpha=0.18)
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(filtered["irradiation"], filtered["ac_power"], alpha=0.55, s=18)
            ax.set_xlabel("Irradiation (kW/m²)")
            ax.set_ylabel("AC Power (kW)")
            ax.set_title("AC Power vs Irradiation")
            ax.grid(alpha=0.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            temp = filtered.assign(hour=filtered["datetime"].dt.hour)
            avg_hour = temp.groupby("hour")["ac_power"].mean()
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(avg_hour.index, avg_hour.values, marker="o", linewidth=2)
            ax.set_xticks(range(24))
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Average AC Power (kW)")
            ax.set_title("Average AC Power by Hour")
            ax.grid(alpha=0.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ============================================================
# WEATHER
# ============================================================

elif page == "Weather Intelligence":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> PUBLIC WEATHER PIPELINE</div><div class="hero-title">Weather Intelligence</div><div class="hero-subtitle">Inspect the Open-Meteo features used by Set B.</div></div>', unsafe_allow_html=True)

    if weather is None:
        st.warning("Open-Meteo weather data was not found. Run fetch_weather.py first.")
    else:
        w = weather.copy()
        st.write("Available weather columns:", ", ".join(map(str, w.columns)))
        required = ["sw_radiation", "temp_2m", "cloud_cover"]
        missing = [c for c in required if c not in w.columns]
        if missing:
            st.error(f"Missing required Set B columns: {missing}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">SHORTWAVE RADIATION</div><div class="metric-value">{w["sw_radiation"].mean():,.1f}<span class="metric-unit">W/m²</span></div><div class="metric-caption">Dataset mean</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-label">2M TEMPERATURE</div><div class="metric-value">{w["temp_2m"].mean():,.1f}<span class="metric-unit">°C</span></div><div class="metric-caption">Dataset mean</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-label">CLOUD COVER</div><div class="metric-value">{w["cloud_cover"].mean():,.1f}<span class="metric-unit">%</span></div><div class="metric-caption">Dataset mean</div></div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(13, 4.8))
            if "datetime" in w.columns:
                ax.plot(w["datetime"], w["sw_radiation"], linewidth=1.8)
                ax.set_xlabel("Datetime")
            else:
                ax.plot(w["sw_radiation"], linewidth=1.8)
                ax.set_xlabel("Observation")
            ax.set_ylabel("Shortwave Radiation (W/m²)")
            ax.set_title("Public Weather Radiation")
            ax.grid(alpha=0.18)
            fig.autofmt_xdate()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(w.head(200), use_container_width=True, hide_index=True)


# ============================================================
# ML PERFORMANCE
# ============================================================

elif page == "ML Performance":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> MODEL EVALUATION</div><div class="hero-title">ML Performance</div><div class="hero-subtitle">Compare Normal Equation, Batch GD and SGD on the chronological test period.</div></div>', unsafe_allow_html=True)

    if rmse_table is not None:
        st.dataframe(rmse_table, use_container_width=True, hide_index=True)
    else:
        st.info("Run src/train_eval.py to generate the RMSE table.")

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Rows", summary.get("train_rows", "—"))
        c2.metric("Testing Rows", summary.get("test_rows", "—"))
        c3.metric("Set B Available", "YES" if summary.get("set_b_available") else "NO")

        st.markdown('<div class="section-title">Correctness Checks</div>', unsafe_allow_html=True)
        checks = {
            "Batch GD cost decreases": summary.get("set_a_batch_gd_monotone_decrease", "—"),
            "Set A |θ_BGD − θ_NE|": summary.get("set_a_max_abs_diff_bgd_vs_normal", "—"),
            "Set B |θ_BGD − θ_NE|": summary.get("set_b_max_abs_diff_bgd_vs_normal", "—"),
            "Set A daytime RMSE (kW)": summary.get("set_a_daytime_rmse_kw", "—"),
            "Set B daytime RMSE (kW)": summary.get("set_b_daytime_rmse_kw", "—"),
        }
        st.dataframe(pd.DataFrame(list(checks.items()), columns=["Check", "Result"]), use_container_width=True, hide_index=True)


# ============================================================
# MODEL INSPECTOR
# ============================================================

elif page == "Model Inspector":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> MODEL TRANSPARENCY</div><div class="hero-title">Model Inspector</div><div class="hero-subtitle">Inspect the exact saved Set B parameters and training scaler used by the prediction dashboard.</div></div>', unsafe_allow_html=True)

    if model is None:
        st.error("Set B weights are unavailable.")
    else:
        st.markdown('<div class="section-title">Learned Parameters</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": model["feature_names"], "Normal Equation θ": model["theta_normal"], "Batch GD θ": model["theta_batch_gd"], "SGD θ": model["theta_sgd"]}), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Training Feature Scaling</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": model["cont_cols"], "Training Mean": model["scaler_mean"], "Training Std": model["scaler_std"]}), use_container_width=True, hide_index=True)

        st.markdown('<div class="info-box">The frontend does not retrain or invent weights. It loads the saved NPZ parameters and reproduces the same z-score transformation and feature order used during training.</div>', unsafe_allow_html=True)

        if theta_table is not None:
            st.markdown('<div class="section-title">Set A θ Table</div>', unsafe_allow_html=True)
            st.dataframe(theta_table, use_container_width=True, hide_index=True)

        theta_b_file = RESULTS_DIR / "table4_theta_set_b.csv"
        theta_b = safe_read_csv(theta_b_file)
        if theta_b is not None:
            st.markdown('<div class="section-title">Set B θ Table</div>', unsafe_allow_html=True)
            st.dataframe(theta_b, use_container_width=True, hide_index=True)


# ============================================================
# TRAINING GRAPHS
# ============================================================

elif page == "Training Graphs":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> TRAINING ARTIFACTS</div><div class="hero-title">Training Graphs</div><div class="hero-subtitle">Plots generated by the supplied training and evaluation pipeline.</div></div>', unsafe_allow_html=True)

    graph_dir = RESULTS_DIR / "figures"
    graphs = [
        ("Batch GD Learning-Rate Experiments", "4_7_batch_gd_learning_rates.png"),
        ("SGD Learning-Rate Experiments", "4_7_sgd_learning_rates.png"),
        ("Actual vs Predicted — Test Week", "5_actual_vs_predicted_test_week.png"),
        ("Residuals vs Hour", "6_residuals_vs_hour.png"),
    ]
    for title, filename in graphs:
        path = graph_dir / filename
        if path.exists():
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Missing graph: {filename}")

# ============================================================
# MODEL TRAINING
# ============================================================

elif page == "Model Training":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> TRAIN / RETRAIN</div><div class="hero-title">Model Training Center</div><div class="hero-subtitle">Train the assignment model from the prepared datasets, then save the weights used by the dashboard.</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="training-grid">
            <div class="training-card"><b>Set A</b><br>On-site irradiation, module temperature, ambient temperature + time features.</div>
            <div class="training-card"><b>Set B</b><br>Public shortwave radiation, 2m temperature, cloud cover + time features.</div>
            <div class="training-card"><b>Target</b><br>Plant-level AC power in kW.</div>
            <div class="training-card"><b>Solvers</b><br>Normal Equation, Batch GD and SGD.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Training Status</div>', unsafe_allow_html=True)
    status_rows = [
        {"Artifact": "Prepared hourly data", "Status": "READY" if HOURLY_FILE.exists() else "MISSING", "Path": str(HOURLY_FILE)},
        {"Artifact": "Open-Meteo Set B data", "Status": "READY" if (OPENMETEO_FILE.exists() or MERGED_WEATHER_FILE.exists()) else "MISSING", "Path": str(OPENMETEO_FILE)},
        {"Artifact": "Set B trained weights", "Status": "READY" if SET_B_WEIGHTS.exists() else "MISSING", "Path": str(SET_B_WEIGHTS)},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Pipeline Controls</div>', unsafe_allow_html=True)
    prep_script = ROOT_DIR / "src" / "prepare.py"
    weather_script = ROOT_DIR / "src" / "fetch_weather.py"

    p1, p2, p3 = st.columns(3)

    def run_script(script, label):
        if not script.exists():
            st.error(f"{label} script not found: {script}")
            return False
        with st.spinner(f"Running {label}..."):
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ROOT_DIR),
                text=True,
                capture_output=True,
            )
        if result.returncode == 0:
            st.success(f"{label} completed.")
            with st.expander(f"{label} console output"):
                st.code((result.stdout or "")[-12000:])
            if result.stderr:
                with st.expander(f"{label} warnings / stderr"):
                    st.code(result.stderr[-8000:])
            return True
        st.error(f"{label} failed.")
        with st.expander(f"{label} error", expanded=True):
            st.code((result.stdout or "") + "\n" + (result.stderr or ""))
        return False

    with p1:
        if st.button("1 · PREPARE HOURLY DATA", use_container_width=True):
            if run_script(prep_script, "prepare.py"):
                st.cache_data.clear()
                st.rerun()
    with p2:
        if st.button("2 · FETCH PUBLIC WEATHER", use_container_width=True):
            if run_script(weather_script, "fetch_weather.py"):
                st.cache_data.clear()
                st.rerun()
    with p3:
        if st.button("3 · TRAIN / RETRAIN MODEL", use_container_width=True):
            if run_script(TRAIN_SCRIPT, "train_eval.py"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    st.markdown(
        '<div class="info-box"><b>Important:</b> the scenario buttons on the Prediction Dashboard are only test/preset inputs. They are not appended to the training data and do not change the learned θ. Retraining always uses the dataset files and the training pipeline.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":
    st.markdown('<div class="hero"><div class="status-pill"><span class="status-dot"></span> DATA EXPLORER</div><div class="hero-title">Plant Dataset</div><div class="hero-subtitle">Inspect the prepared hourly dataset used by the regression pipeline.</div></div>', unsafe_allow_html=True)

    if hourly is None:
        st.error("data/plant1_hourly.csv was not found. Run prepare.py first.")
    else:
        h = hourly.copy()
        rows = len(h)
        start = h["datetime"].min() if "datetime" in h.columns else None
        end = h["datetime"].max() if "datetime" in h.columns else None
        st.markdown(f'<div class="info-box"><b>Observations:</b> {rows:,}<br><b>First:</b> {start}<br><b>Last:</b> {end}</div>', unsafe_allow_html=True)
        query = st.text_input("Filter by datetime", placeholder="Example: 2020-06-10")
        if query and "datetime" in h.columns:
            h = h[h["datetime"].astype(str).str.contains(query, case=False, na=False)]
        st.dataframe(h, use_container_width=True, hide_index=True, height=560)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Solar Plant Intelligence Dashboard · AI4003 Applied Machine Learning · Linear Regression implemented manually with NumPy · Plant 01</div>',
    unsafe_allow_html=True,
)
