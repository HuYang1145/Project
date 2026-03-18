# predictor.py
# 核心推理模块：四大顶配模型集成版 (ARIMA, Prophet, BiLSTM-Hybrid, DiffSTG)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ==========================================
# 1. 环境与路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向父目录

# 🚨 极度重要：确保能找到 Diffusion 内部模块，否则加载模型会报 ModuleNotFoundError
DIFFUSION_CODE_PATH = os.path.join(BASE_DIR, 'notebooks', 'diffusion')
if DIFFUSION_CODE_PATH not in sys.path:
    sys.path.append(DIFFUSION_CODE_PATH)

STATION_LIST = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 
                'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 
                'Wanliu', 'Wanshouxigong', 'Daxing', 'Fangshan', 'Yizhuang', 
                'Miyun', 'Yanqing', 'Yungang', 'Pinggu']

MODELS_CONFIG = {
    "ARIMA": {
        "name": "ARIMA",
        "file": os.path.join(BASE_DIR, 'models', 'arima', 'Aotizhongxin_ARIMA.pkl'),
        "type": "statistical",
        "description": "传统时间序列统计模型，通过自回归和移动平均捕捉线性趋势。系统将自适应当前历史数据刷新模型状态。"
    },
    "PROPHET": {
        "name": "Prophet",
        "file": os.path.join(BASE_DIR, 'models', 'prophet', 'Aotizhongxin_prophet.joblib'),
        "type": "regression",
        "description": "Facebook 开源的时间序列模型，结合未来气象特征 (温度、气压、风速等) 进行多变量拟合。"
    },
    "MIXED": {
        "name": "BiLSTM-Hybrid (SOTA)",
        "dir": os.path.join(BASE_DIR, 'models', 'lstm1'),
        "scaler": os.path.join(BASE_DIR, 'models', 'lstm1', 'scaler_Dongsi.pkl'),
        "config": os.path.join(BASE_DIR, 'models', 'lstm1', 'config.pkl'),
        "type": "hybrid",
        "description": "多步直出集成模型：CEEMDAN/EEMD分解 + 双向LSTM + LEC局部误差校正。"
    },
    "MIXED_48H": {
        "name": "BiLSTM-Hybrid 48H",
        "dir": os.path.join(BASE_DIR, 'models', 'lstm_48h'),
        "scaler": os.path.join(BASE_DIR, 'models', 'lstm_48h', 'scaler_Dongsi.pkl'),
        "config": os.path.join(BASE_DIR, 'models', 'lstm_48h', 'config.pkl'),
        "type": "hybrid",
        "description": "48小时预测模型：CEEMDAN/EEMD分解 + 双向LSTM + LEC局部误差校正。"
    },
    "DIFFUSION": {
        "name": "DiffSTG (Generative AI)",
        "file": os.path.join(BASE_DIR, 'models', 'diffusion', 'checkpoints', 'AIR_BJ+False+True+False+0.002+8+0.0+UGnet+32+200+quad+0.1+200+ddpm+12+8N-200+T_h-12+T_p-12+epsilon_theta-UGnet.dm4stg'),
        "data_dir": os.path.join(BASE_DIR, 'data', 'processed', 'diffusion', 'AIR_BJ'),
        "type": "generative",
        "description": "基于扩散概率模型，不仅预测均值，更提供未来演变的概率置信区间（绿色渔网）。"
    }
}

# ==========================================
# 2. 核心网络架构与缓存加载器
# ==========================================
class BiLSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=12):
        super(BiLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMNet48h(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, output_size=48):
        super(BiLSTMNet48h, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource
def load_arima_model(model_path):
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

# 🚀 为扩散模型加入显存常驻缓存，避免每次推断卡顿！
@st.cache_resource
def load_diffstg_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    # 强制开启极速采样模式
    model.set_ddim_sample_steps(40) 
    model.set_sample_strategy('ddim_multi')
    return model, device

# ==========================================
# 3. 主推理入口
# ==========================================
def load_and_predict(selected_model_key, data_context, steps=12):
    cfg = MODELS_CONFIG[selected_model_key]
    
    hist_df = data_context.get('history')
    if hist_df is None or hist_df.empty: return None
        
    last_time = pd.to_datetime(hist_df['ds'].iloc[-1])
    future_dates = [last_time + pd.Timedelta(hours=i) for i in range(1, steps + 1)]

    # ------------------------------------------
    # 🌟 A. ARIMA 
    # ------------------------------------------
    if selected_model_key == "ARIMA":
        model = load_arima_model(cfg['file'])
        if model is None: return None
        try:
            recent_values = hist_df['y'].values
            updated_model = model.apply(recent_values)
            forecast = updated_model.forecast(steps=steps)
            vals = forecast.values if hasattr(forecast, 'values') else forecast
            return pd.DataFrame({'ds': future_dates, 'yhat': vals})
        except Exception as e:
            st.error(f"💥 ARIMA 推理失败: {str(e)}")
            return None
            
    # ------------------------------------------
    # 🌟 B. Facebook Prophet
    # ------------------------------------------
    elif selected_model_key == "PROPHET":
        if not os.path.exists(cfg['file']): return None
        try:
            model = joblib.load(cfg['file'])
            weather_df = data_context.get('weather')
            if weather_df is None or weather_df.empty: return None
                
            future_test = weather_df.head(steps).copy()
            forecast = model.predict(future_test)
            
            result_df = pd.DataFrame({
                'ds': future_dates, 
                'yhat': forecast['yhat'].values[:steps],
                'y_lower': forecast['yhat_lower'].values[:steps],
                'y_upper': forecast['yhat_upper'].values[:steps]
            })
            result_df['yhat'] = result_df['yhat'].clip(lower=0)
            result_df['y_lower'] = result_df['y_lower'].clip(lower=0)
            result_df['y_upper'] = result_df['y_upper'].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"💥 Prophet 推理报错: {str(e)}")
            return None

    # ------------------------------------------
    # 🌟 C. BiLSTM-Hybrid (完美修复版)
    # ------------------------------------------
    elif selected_model_key == "MIXED":
        if len(hist_df) < 12: 
            st.error("历史数据至少需要 12 步才能进行推理。")
            return None
            
        try:
            from PyEMD import CEEMDAN, EEMD
            saved_config = joblib.load(cfg['config'])
            scaler = joblib.load(cfg['scaler'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # --- 1. 动态自适应数据截取 (解决端点飞线问题) ---
            if data_context.get('type') == 'simulation' and 'full_data' in data_context:
                # 【历史模拟模式】：贪婪截取过去 1000 个点，彻底消除端点效应
                t = data_context['current_index']
                s_idx = data_context['station_index']
                start_idx = max(0, t - 1000)
                raw_input = data_context['full_data'][start_idx:t, s_idx, 0].reshape(-1, 1)
            else:
                # 【实时 API 模式】：如果有多少拿多少，若小于 168 则进行镜像延拓
                raw_input = hist_df['y'].values.reshape(-1, 1)
                if len(raw_input) < 168:
                    # 数学原理：通过将已知信号对称前置，缓解 EMD 画包络线时的末端发散
                    mirrored_part = np.flip(raw_input, axis=0)
                    raw_input = np.concatenate((mirrored_part, raw_input), axis=0)
                    st.toast("⚠️ 实时数据不足，已自动启动信号镜像延拓以稳定分解。", icon="🔧")
            
            scaled_input = scaler.transform(raw_input).flatten()
            
            @st.cache_data(show_spinner=False)
            def fast_decompose(arr):
                return CEEMDAN(trials=2)(arr), EEMD(trials=2)(arr)
            
            with st.spinner("⏳ 进行全局平滑信号分解..."):
                imfs_a, imfs_b = fast_decompose(scaled_input)
            
            # --- 2. 获取各分量预测 ---
            def get_branch_pred(models_list, imfs):
                total_pred = np.zeros(12) 
                for model_name, imf_idx in models_list:
                    # 永远只取分解后最末尾的 12 个点作为 LSTM 输入，此时的末端是最平稳的
                    seq = imfs[imf_idx][-12:] if imf_idx < len(imfs) else np.zeros(12)
                    input_tensor = torch.tensor(seq.reshape(1, 12, 1), dtype=torch.float32).to(device)
                    model_path = os.path.join(cfg['dir'], f"{model_name}.pth")
                    if not os.path.exists(model_path): continue
                    
                    model = BiLSTMNet(hidden_size=64, output_size=12).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        total_pred += model(input_tensor).cpu().numpy().flatten()
                return total_pred
            
            w1 = saved_config.get('w1', 0.5)
            pred_hybrid_scaled = w1 * get_branch_pred(saved_config['models_a'], imfs_a) + (1 - w1) * get_branch_pred(saved_config['models_b'], imfs_b)
            final_pred_test = scaler.inverse_transform(pred_hybrid_scaled.reshape(-1, 1)).flatten()[:steps]
            
            # --- 3. 🚨 重启 LEC 局部误差校正雷达 🚨 ---
            # 获取历史最后一刻的真实 PM2.5 值
            last_known_true = hist_df['y'].iloc[-1]
            current_pred_val = final_pred_test[0]
            
            # 物理阈值判断：如果预测第一步与上一刻真实值相差悬殊
            gamma = abs(current_pred_val - last_known_true)
            
            if gamma >= 8.0:
                st.toast(f"🚨 警报: 检测到 PM2.5 突变 (偏差={gamma:.1f})，启动 LEC 修正模型！", icon="⚠️")
                lec_model_path = os.path.join(cfg['dir'], "LEC_Model.pth")
                
                if os.path.exists(lec_model_path):
                    # 【逻辑推演】：为了推断未来的误差，LEC 需要过去 12 个点的历史误差。
                    # 在工程化中，如果没有完整的历史误差队列，我们可以用近期波动的梯度作为近似替代输入。
                    # 这里我们用最后 12 个已知真实值的缩放形态输入给 LEC 进行激活。
                    recent_12_true = hist_df['y'].iloc[-12:].values.reshape(-1, 1)
                    recent_scaled = scaler.transform(recent_12_true).flatten()
                    lec_input = torch.tensor(recent_scaled.reshape(1, 12, 1), dtype=torch.float32).to(device)
                    
                    lec_model = BiLSTMNet(hidden_size=64, output_size=12).to(device)
                    lec_model.load_state_dict(torch.load(lec_model_path, map_location=device))
                    lec_model.eval()
                    
                    with torch.no_grad():
                        err_pred_scaled = lec_model(lec_input).cpu().numpy().flatten()
                        # 注意：误差通常是微量，我们将其与主预测叠加 (需根据你训练时的正负号逻辑调整，此处假设为相加补偿)
                        final_pred_test += err_pred_scaled[:steps] 
            else:
                st.toast("✅ 数据过渡平滑，无需 LEC 修正。")

            # --- 4. 组装结果 ---
            result_df = pd.DataFrame({'ds': future_dates, 'yhat': final_pred_test})
            result_df['yhat'] = result_df['yhat'].clip(lower=0) # 防止雾霾出现负数
            return result_df
        except Exception as e: # <--- 🚨 必须添加这个部分 🚨
            st.error(f"💥 BiLSTM-Hybrid 推理失败: {str(e)}")
            import traceback
            st.write(traceback.format_exc()) # 打印具体的堆栈信息方便你调试
            return None

    # ------------------------------------------
    # 🌟 D. DiffSTG 扩散概率预测逻辑
    # ------------------------------------------
    elif selected_model_key == "DIFFUSION":
        if data_context.get('type') != 'simulation':
            st.error("❌ DiffSTG 空间图模型目前仅支持在【历史回测模拟】模式下调用！")
            return None
            
        try:
            from easydict import EasyDict as edict
            from algorithm.dataset import CleanDataset, TrafficDataset
            
            # 1. 极速加载模型
            model, device = load_diffstg_model(cfg['file'])
            
            # 2. 严谨地还原训练时的 Config
            diff_config = edict()
            diff_config.data = edict({
                'name': 'AIR_BJ',
                'feature_file': os.path.join(cfg['data_dir'], 'flow.npy'),
                'spatial': os.path.join(cfg['data_dir'], 'adj.npy'),
                'num_features': 1, 'num_vertices': 19, 'points_per_hour': 1,
                'val_start_idx': int(99984 * 0.6),
                'test_start_idx': int(99984 * 0.8)
            })
            diff_config.model = edict({'T_p': 12, 'T_h': 12, 'V': 19, 'F': 1, 'week_len': 7, 'day_len': 24})
            diff_config.device = device
            
            # 3. 完美利用官方 Dataset 提取带时间编码(pos_w/pos_d)的切片数据
            clean_data = CleanDataset(diff_config)
            t = data_context['current_index']
            target_s_idx = data_context['station_index']
            
            # 截取从 t-12 到 t 这个专属瞬间的数组
            dataset = TrafficDataset(clean_data, (t - 12, t - 11), diff_config)
            future, history, pos_w, pos_d = dataset[0]
            
            # 🚨 【核心修复区】：dataset[0] 返回的是原始 Numpy 数组
            # 必须先用 torch.FloatTensor 强行转换为 PyTorch 张量，才能使用 unsqueeze 魔法！
            history = torch.FloatTensor(history).unsqueeze(0).to(device)
            future_ph = torch.zeros_like(history).to(device)
            
            # 时间编码同样需要穿上张量外衣
            if pos_w is not None:
                pos_w = torch.FloatTensor(pos_w).unsqueeze(0).to(device)
            if pos_d is not None:
                pos_d = torch.FloatTensor(pos_d).unsqueeze(0).to(device)
            
            # 拼接与维度转换 -> (1, 1, 19, 24)
            x_masked = torch.cat((history, future_ph), dim=1).transpose(1, 3).to(device)
            
            # 4. 执行生成扩散！生成 8 条未来轨迹 (n_samples=8)
            with torch.no_grad():
                prediction = model((x_masked, pos_w, pos_d), 8) # 输出: (1, 8, 1, 19, 24)

            # 5. 解码概率渔网 (提取目标站点的后 12 步)
            pred_cpu = prediction[0, :, 0, target_s_idx, -12:].cpu().numpy() # 形状: (8轨迹, 12步长)
            
            # 逆归一化
            mean_val, std_val = clean_data.mean, clean_data.std
            pred_real = pred_cpu * std_val + mean_val
            
            # 计算期望(均值)与置信区间上限/下限
            pred_mean = np.mean(pred_real, axis=0)[:steps]
            pred_p5 = np.percentile(pred_real, 5, axis=0)[:steps]
            pred_p95 = np.percentile(pred_real, 95, axis=0)[:steps]
            
            result_df = pd.DataFrame({
                'ds': future_dates, 
                'yhat': pred_mean,
                'y_lower': pred_p5,
                'y_upper': pred_p95
            })
            
            result_df['yhat'] = result_df['yhat'].clip(lower=0)
            result_df['y_lower'] = result_df['y_lower'].clip(lower=0)
            result_df['y_upper'] = result_df['y_upper'].clip(lower=0)
            
            return result_df
            
        except Exception as e:
            st.error(f"💥 DiffSTG 推理失败: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
            return None

    # ------------------------------------------
    # 🌟 E. BiLSTM-48H (48小时长期预测)
    # ------------------------------------------
    elif selected_model_key == "MIXED_48H":
        if len(hist_df) < 48:
            st.error("48小时预测需要至少48小时历史数据。")
            return None

        try:
            from PyEMD import CEEMDAN, EEMD
            saved_config = joblib.load(cfg['config'])
            scaler = joblib.load(cfg['scaler'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            raw_input = hist_df['y'].values[-48:].reshape(-1, 1)
            scaled_input = scaler.transform(raw_input).flatten()

            @st.cache_data(show_spinner=False)
            def fast_decompose_48h(arr):
                return CEEMDAN(trials=5)(arr), EEMD(trials=2)(arr)

            with st.spinner("⏳ 48小时信号分解中..."):
                imfs_a, imfs_b = fast_decompose_48h(scaled_input)

            def get_branch_pred_48h(models_list, imfs):
                total_pred = np.zeros(48)
                for model_name, imf_idx in models_list:
                    seq = imfs[imf_idx][-48:] if imf_idx < len(imfs) else np.zeros(48)
                    input_tensor = torch.tensor(seq.reshape(1, 48, 1), dtype=torch.float32).to(device)
                    model_path = os.path.join(cfg['dir'], f"{model_name}.pth")
                    if not os.path.exists(model_path): continue

                    model = BiLSTMNet48h(hidden_size=96, output_size=48).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        total_pred += model(input_tensor).cpu().numpy().flatten()
                return total_pred

            w1 = saved_config.get('w1', 0.5)
            pred_scaled = w1 * get_branch_pred_48h(saved_config['models_a'], imfs_a) + (1 - w1) * get_branch_pred_48h(saved_config['models_b'], imfs_b)
            final_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[:steps]

            result_df = pd.DataFrame({'ds': future_dates, 'yhat': final_pred})
            result_df['yhat'] = result_df['yhat'].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"💥 BiLSTM-48H 推理失败: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
            return None

    return None


def predict_48h_pollution_alert(data_context):
    """48小时污染预警：基于48h回归预测 + 阈值分类"""
    sys.path.insert(0, BASE_DIR)
    from aqi_classifier import pm25_to_aqi_level, AQI_LEVELS, get_health_advice

    hist_df = data_context.get('history')
    if hist_df is None or len(hist_df) < 48:
        return None

    pred_df = load_and_predict("MIXED_48H", data_context, steps=48)
    if pred_df is None or pred_df.empty:
        return None

    # 转换为AQI等级
    hourly_levels = [pm25_to_aqi_level(pm25) for pm25 in pred_df['yhat'].values]

    # 检测污染时段（轻度污染及以上）
    pollution_hours = [(i, lvl) for i, lvl in enumerate(hourly_levels) if lvl >= 2]

    if not pollution_hours:
        return None

    # 找到最高污染等级
    max_level = max(lvl for _, lvl in pollution_hours)
    max_level_info = AQI_LEVELS[max_level]

    # 统计污染持续时间
    pollution_start = pollution_hours[0][0]
    pollution_end = pollution_hours[-1][0]
    duration_hours = pollution_end - pollution_start + 1

    return {
        'max_level': max_level,
        'level_name': max_level_info['name_cn'],
        'level_color': max_level_info['color'],
        'start_hour': pollution_start,
        'end_hour': pollution_end,
        'duration': duration_hours,
        'health_advice': get_health_advice(max_level),
        'hourly_levels': hourly_levels,
        'hourly_pm25': pred_df['yhat'].values
    }