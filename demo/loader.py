import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import random

# ==========================================
# 1. API 配置区
# ==========================================
API_CONFIG = {
    "OWM": {
        "key": "46368cf05a7d13f95bd414fac0674b9c",
        "lat": 39.98,
        "lon": 116.39,
        "enabled": True
    },
    "QWEATHER": {
        "key": "c7d726eaa5bd4a8b8006dc47a9f8c237",
        "location_id": "101010100",  # 北京
        "host": "https://jh3p4233y7.re.qweatherapi.com",
        "enabled": True
    }
}

# ==========================================
# 2. 工具函数 (数学计算)
# ==========================================
def calculate_dew_point(temp_c, humidity_percent):
    """计算露点 (物理公式)"""
    if humidity_percent == 0: return temp_c
    a = 17.27
    b = 237.7
    humidity_percent = max(humidity_percent, 0.1)
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity_percent / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def get_aqi_info(pm25):
    """AQI 等级计算 (中国标准)"""
    if pm25 is None: return "--", "off"
    if pm25 <= 35: return "Excellent", "success"
    elif pm25 <= 75: return "Good", "success"
    elif pm25 <= 115: return "Lightly Polluted", "warning"
    elif pm25 <= 150: return "Moderately Polluted", "warning"
    elif pm25 <= 250: return "Heavily Polluted", "error"
    else: return "Severely Polluted", "error" # 600 会落在这里

# ==========================================
# 3. 数据获取与本地库同步机制
# ==========================================
import db_utils

# 确保启动时初始化本地数据库
db_utils.init_db()

@st.cache_data(ttl=1800)
def _sync_owm_to_db():
    """[隐藏核心] 定时从 API 拉取近期数据，并追加到本地 SQLite 黑洞中"""
    cfg = API_CONFIG["OWM"]
    if not cfg["enabled"]: return False
    
    # 每次仅拉取过去 10 天的数据用于更新补充
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=45)).timestamp()) 
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={cfg['lat']}&lon={cfg['lon']}&start={start}&end={end}&appid={cfg['key']}"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data.get('list', []):
                dt = datetime.fromtimestamp(item['dt'])
                pm25 = item['components']['pm2_5']
                # 无脑写入本地库 (db_utils 有 UNIQUE 锁，不会重复插入)
                db_utils.save_realtime_data(station='Dongsi', timestamp=dt, pm25=pm25)
            return True
    except Exception as e:
        print(f"API Sync Error: {e}")
    return False

def fetch_owm_history():
    """[暴露给外部的接口] 直接从本地数据库读取超长历史线，喂给模型"""
    
    # 1. 尝试去 API 进一次货（因为有 ttl=1800 缓存保护，半小时内只会真正执行一次）
    _sync_owm_to_db()
    
    # 2. 从本地粮仓 (SQLite) 提取多达 1000 条数据！彻底满足 CEEMDAN 胃口
    df = db_utils.get_recent_data(station='Dongsi', limit=1000)
    
    if df.empty:
        st.error("数据库为空且 API 拉取失败，请检查网络或 OWM Key。")
        return None
        
    # 3. 如果是刚建库的第一天，数据不够 168 条，启用数学平滑延拓（替代之前的随机捏造）
    if len(df) < 168:
        st.toast(f"⚠️ 数据库积累中(当前 {len(df)} 条)，已自动使用数学镜像延拓稳定末端。", icon="🧠")
        # 将现有真实数据镜像翻转补在前面，比 random.uniform 平滑得多
        mirrored_df = df.copy().iloc[::-1]
        
        # 修正镜像部分的时间戳，使其连续向历史推演
        time_diffs = df['ds'].diff().mean()
        if pd.isna(time_diffs): time_diffs = timedelta(hours=1)
        
        first_time = df['ds'].iloc[0]
        mirrored_df['ds'] = [first_time - (i+1)*time_diffs for i in range(len(mirrored_df))]
        
        df = pd.concat([mirrored_df, df]).sort_values('ds').reset_index(drop=True)
        
    # 返回前去掉不要的列，对齐预测器期望的数据格式
    return df[['ds', 'y']]

# 下面的 fetch_qweather_forecast 保持你原来的代码不变即可...

@st.cache_data(ttl=1800)
def fetch_qweather_forecast():
    """[API 2] 和风天气 - 获取未来气象特征"""
    cfg = API_CONFIG["QWEATHER"]
    if not cfg["enabled"]: return None
    
    url = f"{cfg['host']}/v7/weather/24h?location={cfg['location_id']}&key={cfg['key']}"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        data = r.json()
        if data['code'] != '200': return None
        
        records = []
        for item in data['hourly']:
            temp = float(item['temp'])
            humi = float(item['humidity'])
            
            # 🔥🔥 核心修复：单位换算 🔥🔥
            # API 返回 km/h，模型需要 m/s
            wspm_kmh = float(item['windSpeed'])
            wspm_ms = wspm_kmh / 3.6  
            
            records.append({
                # 移除时区信息，防止 Prophet 报错
                'ds': pd.to_datetime(item['fxTime']).replace(tzinfo=None),
                'TEMP': temp,
                'PRES': float(item['pressure']),
                'DEWP': calculate_dew_point(temp, humi),
                'WSPM': wspm_ms,  # 使用换算后的 m/s
                'RAIN': float(item['precip'])
            })
        
        # 转换为 DataFrame
        weather_df = pd.DataFrame(records)
        
        # 再次确保列名符合 Prophet 要求
        expected_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in expected_cols:
            if col not in weather_df.columns:
                st.error(f"❌ API 数据缺少列: {col}")
                return None
                
        return weather_df
        
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None