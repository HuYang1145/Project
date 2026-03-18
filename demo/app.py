# streamlit run app.py
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import pandas as pd
import numpy as np

# Import custom modules
import loader
import loader_simulation  
import predictor

warnings.filterwarnings('ignore')

# ==========================================
# 1. Basic Page Configuration
# ==========================================
st.set_page_config(
    page_title="Air Quality Prediction System", 
    page_icon="🌪️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Control Area
# ==========================================
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # --- A. Mode Switch ---
    st.subheader("1. Data Source Mode")
    mode = st.radio(
        "Select Operation Mode", 
        ["☁️ Real-time API Mode", "🧪 Historical Simulation Demo"], 
        index=0,
        help="Real-time mode consumes API calls; Simulation mode backtests on historical data to verify model accuracy."
    )
    
    if mode == "🧪 Historical Simulation Demo":
        st.info("Currently in backtesting mode: Extracting pure test-set segments for rigorous demonstration.")
        if st.button("🎲 Switch Random Sample", use_container_width=True):
            loader_simulation.change_random_sample()
            st.rerun()
    else:
        if st.button("🔄 Refresh API Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()
    
    # --- B. Model Selection (强力状态锁定版) ---
    st.subheader("2. Prediction Model")
    model_config = predictor.MODELS_CONFIG
    model_map = {v['name']: k for k, v in model_config.items()}
    display_names = list(model_map.keys())
    
    # 🔒 1. 显式初始化“硬皮笔记本”
    if 'locked_model_name' not in st.session_state:
        st.session_state.locked_model_name = display_names[0] # 默认 ARIMA
        
    # 🔒 2. 定义回调函数：只要用户手点了单选框，立刻把新值抄写到笔记本里
    def lock_model_choice():
        st.session_state.locked_model_name = st.session_state.temp_radio_selector

    # 🔒 3. 计算当前应该停留在哪个序号上
    current_index = display_names.index(st.session_state.locked_model_name)

    # 4. 渲染单选框，强制绑定 index 和 callback
    selected_name = st.radio(
        "Select AI Model", 
        display_names, 
        index=current_index,
        key="temp_radio_selector", 
        on_change=lock_model_choice # 一旦改变，立刻触发抄写
    )
    
    # 提取选中的模型标识
    selected_key = model_map[st.session_state.locked_model_name]

    # --- C. Prediction Settings ---
    st.subheader("3. Forecasting Horizon")
    
    # 动态限制预测时长
    if selected_key == "PROPHET":
        max_limit = 24
        help_msg = "Prophet relies on future weather, API only supports 24h."
    elif selected_key == "DIFFUSION":
        max_limit = 12
        help_msg = "DiffSTG generates 12h future probability distributions."
    else:
        max_limit = 12 # LSTM 已经改为直出12步
        help_msg = "Deep learning models predict 12 hours ahead."

    if 'n_steps' not in st.session_state:
        st.session_state.n_steps = 12
        
    if st.session_state.n_steps > max_limit:
        st.session_state.n_steps = max_limit

    n_steps = st.slider(
        "Forecast Duration (Hours)",
        min_value=1,
        max_value=max_limit,
        value=st.session_state.n_steps,
        help=help_msg
    )
    st.session_state.n_steps = n_steps

# ==========================================
# 3. Data Loading Logic
# ==========================================
st.title(f"🌤️ Beijing Air Quality Smart Monitoring System")

df_history = None
df_weather = None
ground_truth = None
data_context = {}

with st.spinner(f"Loading data ({mode})..."):
    if mode == "☁️ Real-time API Mode":
        df_history = loader.fetch_owm_history()
        df_weather = loader.fetch_qweather_forecast()
        data_context = {'type': 'api', 'history': df_history, 'weather': df_weather}
    else:
        # 🌟 针对 NeuralProphet 的专属数据通道
        if selected_key == "PROPHET":
            df_history, ground_truth, sim_context = loader_simulation.get_neuralprophet_simulation_data()
            df_weather = sim_context.get('weather')
            data_context = sim_context
        # 🌟 其他模型 (ARIMA, LSTM, Diffusion) 的常规数据通道
        else:
            df_history, ground_truth, sim_context = loader_simulation.get_simulation_data(target_station='Dongsi')
            df_weather = sim_context.get('weather')
            data_context = {'type': 'simulation', 'history': df_history, 'weather': df_weather, **sim_context}
# ==========================================
# 4. Top Metrics Cards
# ==========================================
cols = st.columns(4)

if df_history is not None and not df_history.empty:
    curr_pm25 = df_history['y'].iloc[-1]
    curr_time = df_history['ds'].iloc[-1]
    aqi_txt, aqi_cls = loader.get_aqi_info(curr_pm25)
    
    cols[0].metric("Current PM2.5", f"{curr_pm25:.1f}", "Measured")
    cols[1].metric("Air Quality AQI", aqi_txt, delta_color="inverse")
else:
    cols[0].metric("Current PM2.5", "--")
    cols[1].metric("Air Quality", "Unknown")

if mode == "☁️ Real-time API Mode" and df_weather is not None:
    avg_temp = df_weather['TEMP'].mean()
    cols[2].metric("Tomorrow's Avg Temp", f"{avg_temp:.1f} °C")
else:
    cols[2].metric("Target Area", "Beijing (Dongsi)")

if mode == "☁️ Real-time API Mode":
    cols[3].metric("System Time", datetime.now().strftime("%H:%M"))
else:
    sim_time_str = curr_time.strftime("%Y-%m-%d %H:00") if df_history is not None else "--"
    cols[3].metric("Simulated Cutoff Time", sim_time_str)

st.markdown("---")

# ==========================================
# 4.5. 48-Hour Pollution Alert (Auto-run)
# ==========================================
if df_history is not None:
    with st.spinner("🔍 Detecting 48-hour pollution risk..."):
        alert_info = predictor.predict_48h_pollution_alert(data_context)

    if alert_info:
        st.markdown(f"""
        <div style="background-color: {alert_info['level_color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">⚠️ 48-Hour Pollution Alert</h2>
            <p style="color: white; font-size: 18px; margin: 10px 0;">
                Expected <b>{alert_info['level_name']}</b> in <b>{alert_info['start_hour']}-{alert_info['end_hour']} hours</b>
            </p>
            <p style="color: white; margin: 5px 0;">Duration: ~{alert_info['duration']} hours</p>
            <p style="color: white; margin: 5px 0;">💡 {alert_info['health_advice']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Good air quality expected for the next 48 hours")

# ==========================================
# 5. Core Chart Plotting
# ==========================================
st.subheader(f"📈 Predictive Analysis: {model_config[selected_key]['name']}")

fig = go.Figure()

# A. Plot Historical Data (缩减展示到过去 24 小时)
if df_history is not None:
    plot_history = df_history.tail(24) # 👈 修改点：只展示过去24h，图表更紧凑清晰
    fig.add_trace(go.Scatter(
        x=plot_history['ds'], 
        y=plot_history['y'],
        mode='lines+markers', 
        name='Historical Data',
        line=dict(color='black', width=2),
        marker=dict(size=5, color='black')
    ))

# B. Plot Future Ground Truth
if mode == "🧪 Historical Simulation Demo" and ground_truth is not None:
    gt_plot = ground_truth.head(n_steps)
    fig.add_trace(go.Scatter(
        x=gt_plot['ds'], 
        y=gt_plot['y'],
        mode='lines+markers', 
        name='Ground Truth',
        line=dict(color='#e74c3c', width=2, dash='dot'), # 红色点划线
        marker=dict(size=6, color='#e74c3c')
    ))

# C. Call model for prediction
if df_history is not None:
    # 实时API模式：使用预加载结果
    # Load model on demand
    with st.spinner(f"Running {selected_name} model inference..."):
        pred_df = predictor.load_and_predict(selected_key, data_context, n_steps)

    # D. Plot Prediction Results
    if pred_df is not None:
        # 基础预测线 (均值/确定性预测)
        pred_df['yhat'] = pred_df['yhat'].clip(lower=0)

        # 添加连接点：历史最后一点 + 预测数据
        last_hist_point = pd.DataFrame({
            'ds': [df_history['ds'].iloc[-1]],
            'yhat': [df_history['y'].iloc[-1]]
        })
        pred_with_connection = pd.concat([last_hist_point, pred_df[['ds', 'yhat']]], ignore_index=True)

        # 🟢 如果模型返回了置信区间 (针对 DiffSTG/Prophet)
        if 'y_lower' in pred_df.columns and 'y_upper' in pred_df.columns:
            pred_df['y_lower'] = pred_df['y_lower'].clip(lower=0)
            
            # 画置信区间阴影 (先画上线，再填下线)
            fig.add_trace(go.Scatter(
                x=pd.concat([pred_df['ds'], pred_df['ds'][::-1]]),
                y=pd.concat([pred_df['y_upper'], pred_df['y_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.3)', # 绿色透明阴影
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='90% Confidence Interval',
                showlegend=True
            ))

        # 画出预测主线（包含连接点）
        fig.add_trace(go.Scatter(
            x=pred_with_connection['ds'],
            y=pred_with_connection['yhat'],
            mode='lines+markers',
            name='Model Prediction (Mean)' if 'y_lower' in pred_df.columns else 'Model Prediction',
            line=dict(color='#3498db', width=3, dash='dash'), # 蓝色虚线
            marker=dict(symbol='circle-open', size=8)
        ))
        
        # 误差计算与展示
        if ground_truth is not None:
            try:
                merged = pd.merge(pred_df, ground_truth, on='ds', suffixes=('_pred', '_true'))
                if not merged.empty:
                    rmse = np.sqrt(((merged['yhat'] - merged['y']) ** 2).mean())
                    mae = (merged['yhat'] - merged['y']).abs().mean()
                    st.toast(f"✅ Simulation MAE: {mae:.2f} | RMSE: {rmse:.2f}", icon="📊")
            except:
                pass
    else:
        if selected_key == "PROPHET" and df_weather is None:
            st.warning("⚠️ Prophet model requires weather forecast data.")
        elif selected_key == "DIFFUSION" and mode == "☁️ Real-time API Mode":
            st.warning("⚠️ DiffSTG requires full network spatial data. Please switch to Historical Simulation mode.")

# 统一布局设置
fig.update_layout(
    height=550,
    xaxis_title="Time",
    yaxis_title="PM2.5 Concentration (µg/m³)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    shapes=[ # 在当前时间点画一条红色的垂直分割线
        dict(
            type="line",
            x0=curr_time, y0=0, x1=curr_time, y1=1,
            yref="paper", line=dict(color="red", width=2, dash="dashdot")
        )
    ]
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. Additional Features
# ==========================================
if selected_key == "DIFFUSION":
    st.info("""
    **🤖 About DiffSTG Generative Forecasting**
    
    Traditional models predict a single rigid path. DiffSTG leverages a **Denoising Diffusion Process** to generate 8 possible future trajectories based on spatio-temporal graphs.
    The blue line represents the **Mean Prediction**, while the green shaded area reveals the **90% Confidence Interval**, providing crucial insights into future uncertainties.
    """)