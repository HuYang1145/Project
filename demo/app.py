# streamlit run app.py
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loader
import loader_simulation
import predictor

warnings.filterwarnings("ignore")


def build_alert_cache_key(mode_name, data_context):
    """Cache 48-hour alerts by data sample rather than by the visible model selector."""
    hist_df = data_context.get("history")
    if hist_df is None or hist_df.empty:
        return None

    if mode_name == "Historical Simulation Demo":
        sample_key = data_context.get("sample_key")
        if sample_key:
            return f"simulation:{sample_key}"

    last_time = pd.to_datetime(hist_df["ds"].iloc[-1]).isoformat()
    return f"{mode_name}:{last_time}:{len(hist_df)}"

# ==========================================
# 1. Basic page configuration
# ==========================================
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. Sidebar controls
# ==========================================
with st.sidebar:
    st.header("Control Panel")

    st.subheader("1. Data Source Mode")
    mode = st.radio(
        "Select Operation Mode",
        ["Real-time API Mode", "Historical Simulation Demo"],
        index=0,
        help="Real-time mode consumes API calls; simulation mode backtests on historical data to verify model accuracy.",
    )

    if mode == "Historical Simulation Demo":
        st.info("Currently in backtesting mode: extracting pure test-set segments for rigorous evaluation.")
        if st.button("Switch Random Sample", use_container_width=True):
            loader_simulation.change_random_sample()
            st.session_state.cached_common_data = None
            st.session_state.cached_prophet_data = None
            st.rerun()
    else:
        if st.button("Refresh API Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    st.subheader("2. Prediction Model")
    model_config = predictor.MODELS_CONFIG

    # Hide MIXED_48H because it is only used by the 48-hour alert pipeline.
    model_map = {v["name"]: k for k, v in model_config.items() if k != "MIXED_48H"}
    display_names = list(model_map.keys())

    # Initialize the locked selection explicitly.
    if "locked_model_name" not in st.session_state:
        st.session_state.locked_model_name = display_names[0]

    # Copy the selected radio value into the locked state.
    def lock_model_choice():
        st.session_state.locked_model_name = st.session_state.temp_radio_selector

    # Resolve the current selection index.
    current_index = display_names.index(st.session_state.locked_model_name)

    selected_name = st.radio(
        "Select AI Model",
        display_names,
        index=current_index,
        key="temp_radio_selector",
        on_change=lock_model_choice,
    )

    selected_key = model_map[st.session_state.locked_model_name]

    st.subheader("3. Forecasting Horizon")
    if selected_key == "PROPHET":
        max_limit = 24
        help_msg = "Prophet relies on future weather data, so the API path only supports 24 hours."
    elif selected_key == "DIFFUSION":
        max_limit = 12
        help_msg = "DiffSTG generates 12-hour future probability distributions."
    else:
        max_limit = 12
        help_msg = "Deep learning models predict 12 hours ahead."

    if "n_steps" not in st.session_state:
        st.session_state.n_steps = 12

    if st.session_state.n_steps > max_limit:
        st.session_state.n_steps = max_limit

    n_steps = st.slider(
        "Forecast Duration (Hours)",
        min_value=1,
        max_value=max_limit,
        value=st.session_state.n_steps,
        help=help_msg,
    )
    st.session_state.n_steps = n_steps

    st.divider()

    st.subheader("4. Advanced Features")
    if "enable_48h_alert" not in st.session_state:
        st.session_state.enable_48h_alert = True

    enable_alert = st.checkbox(
        "Enable 48H Pollution Alert",
        value=st.session_state.enable_48h_alert,
        help="Disable this option to reduce memory usage when running heavy models.",
    )
    st.session_state.enable_48h_alert = enable_alert

# ==========================================
# 3. Data loading
# ==========================================
st.title("Beijing Air Quality Smart Monitoring System")

if "cached_common_data" not in st.session_state:
    st.session_state.cached_common_data = None
if "cached_prophet_data" not in st.session_state:
    st.session_state.cached_prophet_data = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None
if "alert_cache" not in st.session_state:
    st.session_state.alert_cache = {}

df_history = None
df_weather = None
ground_truth = None
data_context = {}

if st.session_state.last_mode != mode:
    st.session_state.cached_common_data = None
    st.session_state.cached_prophet_data = None
    st.session_state.last_mode = mode

with st.spinner(f"Loading data ({mode})..."):
    if mode == "Real-time API Mode":
        df_history = loader.fetch_owm_history()
        df_weather = loader.fetch_qweather_forecast()
        data_context = {"type": "api", "history": df_history, "weather": df_weather}
    else:
        if selected_key == "PROPHET":
            if st.session_state.cached_prophet_data is None:
                df_history, ground_truth, sim_context = loader_simulation.get_neuralprophet_simulation_data()
                st.session_state.cached_prophet_data = (df_history, ground_truth, sim_context)
            else:
                df_history, ground_truth, sim_context = st.session_state.cached_prophet_data
            df_weather = sim_context.get("weather")
            data_context = sim_context
        else:
            if st.session_state.cached_common_data is None:
                df_history, ground_truth, sim_context = loader_simulation.get_simulation_data(target_station="Dongsi")
                st.session_state.cached_common_data = (df_history, ground_truth, sim_context)
            else:
                df_history, ground_truth, sim_context = st.session_state.cached_common_data
            df_weather = sim_context.get("weather")
            data_context = {"type": "simulation", "history": df_history, "weather": df_weather, **sim_context}

# ==========================================
# 4. Top metrics
# ==========================================
cols = st.columns(4)

curr_time = datetime.now()
if df_history is not None and not df_history.empty:
    curr_pm25 = df_history["y"].iloc[-1]
    curr_time = df_history["ds"].iloc[-1]
    aqi_txt, aqi_cls = loader.get_aqi_info(curr_pm25)

    cols[0].metric("Current PM2.5", f"{curr_pm25:.1f}", "Measured")
    cols[1].metric("Air Quality AQI", aqi_txt, delta_color="inverse")
else:
    cols[0].metric("Current PM2.5", "--")
    cols[1].metric("Air Quality", "Unknown")

if mode == "Real-time API Mode" and df_weather is not None:
    avg_temp = df_weather["TEMP"].mean()
    cols[2].metric("Tomorrow's Avg Temp", f"{avg_temp:.1f} °C")
else:
    cols[2].metric("Target Area", "Beijing")

if mode == "Real-time API Mode":
    cols[3].metric("System Time", datetime.now().strftime("%H:%M"))
else:
    sim_time_str = curr_time.strftime("%Y-%m-%d %H:00") if df_history is not None else "--"
    cols[3].metric("Simulated Cutoff Time", sim_time_str)

st.markdown("---")

# ==========================================
# 4.5. 48-hour pollution alert
# ==========================================
alert_paused_for_memory = selected_key == "MIXED"

should_run_alert = (
    df_history is not None
    and len(df_history) >= 48
    and selected_key not in {"DIFFUSION", "MIXED"}
    and st.session_state.get("enable_48h_alert", True)
)

if alert_paused_for_memory:
    predictor.clear_hybrid_resources()
    if st.session_state.get("enable_48h_alert", True):
        st.info("48-hour alert is paused while BiLSTM-Hybrid is selected so only one LSTM model stays in memory.")

if should_run_alert:
    alert_cache_key = build_alert_cache_key(mode, data_context)
    if alert_cache_key not in st.session_state.alert_cache:
        predictor.clear_hybrid_resources()
        with st.spinner("Detecting 48-hour pollution risk..."):
            st.session_state.alert_cache[alert_cache_key] = predictor.predict_48h_pollution_alert(data_context)

    alert_info = st.session_state.alert_cache.get(alert_cache_key)

    if alert_info:
        st.markdown(
            f"""
            <div style="background-color: {alert_info['level_color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">48-Hour Pollution Alert</h2>
                <p style="color: white; font-size: 18px; margin: 10px 0;">
                    Expected <b>{alert_info['level_name']}</b> in <b>{alert_info['start_hour']}-{alert_info['end_hour']} hours</b>
                </p>
                <p style="color: white; margin: 5px 0;">Duration: ~{alert_info['duration']} hours</p>
                <p style="color: white; margin: 5px 0;">Advice: {alert_info['health_advice']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("Good air quality is expected for the next 48 hours.")

if selected_key != "MIXED":
    predictor.clear_hybrid_resources()

# ==========================================
# 5. Main chart
# ==========================================
st.subheader(f"Predictive Analysis: {model_config[selected_key]['name']}")

fig = go.Figure()

if df_history is not None:
    # Limit the historical display to the latest 24 hours to keep the chart compact.
    plot_history = df_history.tail(24)
    fig.add_trace(
        go.Scatter(
            x=plot_history["ds"],
            y=plot_history["y"],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color="black", width=2),
            marker=dict(size=5, color="black"),
        )
    )

if mode == "Historical Simulation Demo" and ground_truth is not None:
    gt_plot = ground_truth.head(n_steps)
    fig.add_trace(
        go.Scatter(
            x=gt_plot["ds"],
            y=gt_plot["y"],
            mode="lines+markers",
            name="Ground Truth",
            line=dict(color="#e74c3c", width=2, dash="dot"),
            marker=dict(size=6, color="#e74c3c"),
        )
    )

if df_history is not None:
    with st.spinner(f"Running {selected_name} model inference..."):
        pred_df = predictor.load_and_predict(selected_key, data_context, n_steps)

    if pred_df is not None:
        pred_df["yhat"] = pred_df["yhat"].clip(lower=0)

        last_hist_point = pd.DataFrame({"ds": [df_history["ds"].iloc[-1]], "yhat": [df_history["y"].iloc[-1]]})
        pred_with_connection = pd.concat([last_hist_point, pred_df[["ds", "yhat"]]], ignore_index=True)

        if "y_lower" in pred_df.columns and "y_upper" in pred_df.columns:
            pred_df["y_lower"] = pred_df["y_lower"].clip(lower=0)
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([pred_df["ds"], pred_df["ds"][::-1]]),
                    y=pd.concat([pred_df["y_upper"], pred_df["y_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(46, 204, 113, 0.3)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="90% Confidence Interval",
                    showlegend=True,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=pred_with_connection["ds"],
                y=pred_with_connection["yhat"],
                mode="lines+markers",
                name="Model Prediction (Mean)" if "y_lower" in pred_df.columns else "Model Prediction",
                line=dict(color="#3498db", width=3, dash="dash"),
                marker=dict(symbol="circle-open", size=8),
            )
        )

        if ground_truth is not None:
            try:
                merged = pd.merge(pred_df, ground_truth, on="ds", suffixes=("_pred", "_true"))
                if not merged.empty:
                    rmse = np.sqrt(((merged["yhat"] - merged["y"]) ** 2).mean())
                    mae = (merged["yhat"] - merged["y"]).abs().mean()
                    st.toast(f"Simulation MAE: {mae:.2f} | RMSE: {rmse:.2f}", icon=None)
            except Exception:
                pass
    else:
        if selected_key == "PROPHET" and df_weather is None:
            st.warning("Prophet requires weather forecast data.")
        elif selected_key == "DIFFUSION" and mode == "Real-time API Mode":
            st.warning("DiffSTG requires full-network spatial data. Switch to Historical Simulation Demo.")

fig.update_layout(
    height=550,
    xaxis_title="Time",
    yaxis_title="PM2.5 Concentration (µg/m³)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    shapes=[
        dict(
            type="line",
            x0=curr_time,
            y0=0,
            x1=curr_time,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dashdot"),
        )
    ],
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. Additional notes
# ==========================================
if selected_key == "DIFFUSION":
    st.info(
        """
        **About DiffSTG Generative Forecasting**

        Traditional models predict a single rigid path. DiffSTG uses a denoising diffusion process
        to generate eight possible future trajectories from the spatio-temporal graph.
        The blue line is the mean prediction, while the green band shows the 90% confidence interval.
        """
    )
