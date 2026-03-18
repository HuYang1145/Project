# loader_simulation.py
# 专门用于毕设演示：从本地测试集数据中读取片段进行严格模拟

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta
import streamlit as st

# 设定基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向父目录
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'diffusion', 'AIR_BJ')
FLOW_PATH = os.path.join(DATA_DIR, 'flow.npy')

STATION_LIST = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 
                'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 
                'Wanliu', 'Wanshouxigong', 'Daxing', 'Fangshan', 'Yizhuang', 
                'Miyun', 'Yanqing', 'Yungang', 'Pinggu']

@st.cache_data
def load_full_data():
    """一次性加载全量数据并缓存"""
    if not os.path.exists(FLOW_PATH):
        st.error(f"❌ 找不到本地数据文件: {FLOW_PATH}")
        return None
    return np.load(FLOW_PATH)

def get_simulation_data(target_station='Dongsi'):
    """
    【学术严谨版】严格从 Test Set 中随机切取数据，并无缝对齐时间轴
    """
    # 1. 🚨 被你不小心删掉的：加载数据和切片逻辑 🚨
    data = load_full_data() # (Total_Time, 19, 1)
    if data is None: return None, None, None
    
    total_len = data.shape[0]
    
    # 核心防作弊机制：严格定位到测试集 (最后20%)
    test_start_idx = int(total_len * 0.8)
    
    # 切分点 t 必须在测试集范围内，且保证往前有 168h，往后有 12h
    min_idx = test_start_idx + 168
    max_idx = total_len - 12
    
    # Session State 保持页面刷新时不跳动
    if 'sim_t' not in st.session_state or st.session_state.sim_t >= max_idx or st.session_state.sim_t < min_idx:
        st.session_state.sim_t = random.randint(min_idx, max_idx)
    
    t = st.session_state.sim_t
    
    # 找到目标站点的索引
    if target_station in STATION_LIST:
        s_idx = STATION_LIST.index(target_station)
    else:
        s_idx = 0 
        
    # 2. 🌟 时间轴无缝咬合逻辑 🌟
    # 设定一个完美对齐的基准时间（当前整点），这就是“历史”的最后一刻
    anchor_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # --- A. 构造 History DataFrame (给 Prophet/LSTM/ARIMA 用) ---
    hist_vals = data[t-168:t, s_idx, 0]
    # i=167时，167-167=0，完美停在 anchor_time
    hist_dates = [anchor_time - timedelta(hours=167-i) for i in range(168)]
    history_df = pd.DataFrame({'ds': hist_dates, 'y': hist_vals})
    
    # --- B. 构造 Ground Truth (未来真值) ---
    gt_vals = data[t:t+12, s_idx, 0]
    # 从 anchor_time 的下一个小时开始，严密咬合
    gt_dates = [anchor_time + timedelta(hours=i+1) for i in range(12)]
    ground_truth_df = pd.DataFrame({'ds': gt_dates, 'y': gt_vals})

    # --- C. 构造伪造气象数据 (专给 Prophet 模拟用) ---
    weather_dates = hist_dates + gt_dates
    weather_df = pd.DataFrame({
        'ds': weather_dates,
        'TEMP': [15.0 + random.uniform(-5, 5) for _ in range(180)],
        'PRES': [1010.0 + random.uniform(-10, 10) for _ in range(180)],
        'DEWP': [5.0 + random.uniform(-2, 2) for _ in range(180)],
        'WSPM': [2.0 + random.uniform(-1, 3) for _ in range(180)],
        'RAIN': [0.0 for _ in range(180)]
    })
    
    # --- D. 构造 Diffusion Context (全量数据) ---
    context = {
        'type': 'simulation',
        'full_data': data, 
        'current_index': t, 
        'station_index': s_idx,
        'weather': weather_df  # 把伪造的天气打包塞进去给 Prophet
    }
    
    return history_df, ground_truth_df, context

def change_random_sample():
    """切换随机样本（强制重新随机，但保持模型选择）"""
    if 'sim_t' in st.session_state:
        del st.session_state.sim_t
    # 注意：不删除 locked_model_name，保持用户的模型选择状态

# ==========================================
# [新增] 专供 NeuralProphet 的数据加载逻辑
# ==========================================
NP_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'neuralprophet', 'Aotizhongxin_neuralprophet.csv')

def get_neuralprophet_simulation_data():
    """从专属的 NeuralProphet 清洗数据中截取片段"""
    if not os.path.exists(NP_DATA_PATH):
        st.error(f"❌ 找不到 NeuralProphet 专属数据: {NP_DATA_PATH}")
        return None, None, None

    df = pd.read_csv(NP_DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    
    total_len = len(df)
    test_start_idx = int(total_len * 0.8)
    
    min_idx = test_start_idx + 168
    max_idx = total_len - 24  # NeuralProphet 预测 24 步
    
    if 'sim_t' not in st.session_state or st.session_state.sim_t >= max_idx or st.session_state.sim_t < min_idx:
        st.session_state.sim_t = random.randint(min_idx, max_idx)
        
    t = st.session_state.sim_t
    
    # 锚点时间 (历史的最后一刻)
    anchor_time = df['ds'].iloc[t-1]
    
    # 1. 历史数据 (过去 168 小时，包含 y 和 天气特征)
    history_df = df.iloc[t-168 : t].copy()
    
    # 2. 未来真值 (未来 24 小时)
    ground_truth_df = df.iloc[t : t+24][['ds', 'y']].copy()
    
    # 3. 未来天气特征 (NeuralProphet 必须知道未来的天气才能预测)
    weather_df = df.iloc[t : t+24][['ds', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].copy()
    
    context = {
        'type': 'simulation',
        'history': history_df,
        'weather': weather_df
    }
    
    return history_df[['ds', 'y']], ground_truth_df, context