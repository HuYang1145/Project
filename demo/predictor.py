# predictor.py
# Core inference module integrating ARIMA, Prophet, BiLSTM-Hybrid, and DiffSTG

import os
import sys
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ==========================================
# 1. Environment and paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent directory

# Ensure Diffusion internal modules can be imported, otherwise model loading fails.
DIFFUSION_CODE_PATH = os.path.join(BASE_DIR, "notebooks", "diffusion")
if DIFFUSION_CODE_PATH not in sys.path:
    sys.path.append(DIFFUSION_CODE_PATH)

STATION_LIST = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
    "Daxing",
    "Fangshan",
    "Yizhuang",
    "Miyun",
    "Yanqing",
    "Yungang",
    "Pinggu",
]

MODELS_CONFIG = {
    "ARIMA": {
        "name": "ARIMA",
        "file": os.path.join(BASE_DIR, "models", "arima", "Aotizhongxin_ARIMA.pkl"),
        "type": "statistical",
        "description": "A classical time-series model that captures linear trends with autoregression and moving averages.",
    },
    "PROPHET": {
        "name": "Prophet",
        "file": os.path.join(BASE_DIR, "models", "prophet", "Aotizhongxin_prophet.joblib"),
        "type": "regression",
        "description": "A multivariate time-series model that uses future weather features such as temperature, pressure, and wind speed.",
    },
    "MIXED": {
        "name": "BiLSTM-Hybrid (SOTA)",
        "dir": os.path.join(BASE_DIR, "models", "lstm1"),
        "scaler": os.path.join(BASE_DIR, "models", "lstm1", "scaler_Dongsi.pkl"),
        "config": os.path.join(BASE_DIR, "models", "lstm1", "config.pkl"),
        "type": "hybrid",
        "description": "A direct multi-step ensemble with CEEMDAN/EEMD decomposition, bidirectional LSTMs, and local error correction.",
    },
    "MIXED_48H": {
        "name": "BiLSTM-Hybrid 48H",
        "dir": os.path.join(BASE_DIR, "models", "lstm_48h"),
        "scaler": os.path.join(BASE_DIR, "models", "lstm_48h", "scaler_Dongsi.pkl"),
        "config": os.path.join(BASE_DIR, "models", "lstm_48h", "config.pkl"),
        "type": "hybrid",
        "description": "A 48-hour forecast model with CEEMDAN/EEMD decomposition, bidirectional LSTMs, and local error correction.",
    },
    "DIFFUSION": {
        "name": "DiffSTG (Generative AI)",
        "file": os.path.join(
            BASE_DIR,
            "models",
            "diffusion",
            "checkpoints",
            "AIR_BJ+False+True+False+0.002+8+0.0+UGnet+32+200+quad+0.1+200+ddpm+12+8N-200+T_h-12+T_p-12+epsilon_theta-UGnet.dm4stg",
        ),
        "data_dir": os.path.join(BASE_DIR, "data", "processed", "diffusion", "AIR_BJ"),
        "type": "generative",
        "description": "A diffusion probabilistic model that returns both mean forecasts and confidence intervals for future evolution.",
    },
}


# ==========================================
# 2. Core network definitions and cached loaders
# ==========================================
class BiLSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=12):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class BiLSTMNet48h(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, output_size=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


@st.cache_resource
def load_joblib_resource(file_path):
    if not os.path.exists(file_path):
        return None
    return joblib.load(file_path)


@st.cache_resource
def load_arima_model(model_path):
    return load_joblib_resource(model_path)


@st.cache_resource
def load_prophet_model(model_path):
    return load_joblib_resource(model_path)


@st.cache_resource
def load_diffstg_model(model_path):
    """Keep the DiffSTG model cached to avoid repeated cold starts."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.set_ddim_sample_steps(40)
    model.set_sample_strategy("ddim_multi")
    return model, device


@st.cache_resource
def load_hybrid_bundle(model_dir, config_path, scaler_path, hidden_size, output_size, include_lec=False):
    """Cache hybrid ensemble weights so Streamlit reruns do not reload them from disk."""
    saved_config = load_joblib_resource(config_path)
    scaler = load_joblib_resource(scaler_path)
    if saved_config is None or scaler is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_cls = BiLSTMNet48h if output_size == 48 else BiLSTMNet

    model_names = {name for name, _ in saved_config.get("models_a", []) + saved_config.get("models_b", [])}
    loaded_models = {}

    for model_name in model_names:
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            continue

        model = network_cls(hidden_size=hidden_size, output_size=output_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        loaded_models[model_name] = model

    lec_model = None
    if include_lec:
        lec_model_path = os.path.join(model_dir, "LEC_Model.pth")
        if os.path.exists(lec_model_path):
            lec_model = network_cls(hidden_size=hidden_size, output_size=output_size).to(device)
            lec_model.load_state_dict(torch.load(lec_model_path, map_location=device))
            lec_model.eval()

    return {
        "config": saved_config,
        "scaler": scaler,
        "device": device,
        "models": loaded_models,
        "lec_model": lec_model,
    }


def clear_hybrid_resources():
    """Release cached hybrid bundles so only one LSTM family stays resident."""
    load_hybrid_bundle.clear()


# ==========================================
# 3. Main inference entry point
# ==========================================
GLOBAL_ML_LOCK = threading.Lock()

@st.cache_data(show_spinner=False)
def fast_decompose(arr_tuple):
    """Use a tuple input so Streamlit cache keys remain stable."""
    from PyEMD import CEEMDAN, EEMD
    arr = np.array(arr_tuple)
    return CEEMDAN(trials=2)(arr), EEMD(trials=2)(arr)

@st.cache_data(show_spinner=False)
def fast_decompose_48h(arr_tuple):
    """Use a tuple input so Streamlit cache keys remain stable."""
    from PyEMD import CEEMDAN, EEMD
    arr = np.array(arr_tuple)
    return CEEMDAN(trials=5)(arr), EEMD(trials=2)(arr)

def _inner_load_and_predict(selected_model_key, data_context, steps=12):
    cfg = MODELS_CONFIG[selected_model_key]

    hist_df = data_context.get("history")
    if hist_df is None or hist_df.empty:
        return None

    last_time = pd.to_datetime(hist_df["ds"].iloc[-1])
    future_dates = [last_time + pd.Timedelta(hours=i) for i in range(1, steps + 1)]

    def get_lec_threshold(horizon_steps):
        """Use a horizon-conditioned LEC threshold to match the training branch."""
        return 8.0 if horizon_steps <= 12 else 12.0

    # ------------------------------------------
    # A. ARIMA
    # ------------------------------------------
    if selected_model_key == "ARIMA":
        model = load_arima_model(cfg["file"])
        if model is None:
            return None
        try:
            recent_values = hist_df["y"].values
            updated_model = model.apply(recent_values)
            forecast = updated_model.forecast(steps=steps)
            vals = forecast.values if hasattr(forecast, "values") else forecast
            return pd.DataFrame({"ds": future_dates, "yhat": vals})
        except Exception as e:
            st.error(f"ARIMA inference failed: {e}")
            return None

    # ------------------------------------------
    # B. Prophet
    # ------------------------------------------
    if selected_model_key == "PROPHET":
        try:
            model = load_prophet_model(cfg["file"])
            if model is None:
                return None
            weather_df = data_context.get("weather")
            if weather_df is None or weather_df.empty:
                return None

            future_test = weather_df.head(steps).copy()
            forecast = model.predict(future_test)

            result_df = pd.DataFrame(
                {
                    "ds": future_dates,
                    "yhat": forecast["yhat"].values[:steps],
                    "y_lower": forecast["yhat_lower"].values[:steps],
                    "y_upper": forecast["yhat_upper"].values[:steps],
                }
            )
            result_df["yhat"] = result_df["yhat"].clip(lower=0)
            result_df["y_lower"] = result_df["y_lower"].clip(lower=0)
            result_df["y_upper"] = result_df["y_upper"].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"Prophet inference failed: {e}")
            return None

    # ------------------------------------------
    # C. BiLSTM-Hybrid
    # ------------------------------------------
    if selected_model_key == "MIXED":
        if len(hist_df) < 12:
            st.error("At least 12 historical steps are required for inference.")
            return None

        try:
            from PyEMD import CEEMDAN, EEMD

            bundle = load_hybrid_bundle(cfg["dir"], cfg["config"], cfg["scaler"], hidden_size=64, output_size=12, include_lec=True)
            if bundle is None:
                return None

            saved_config = bundle["config"]
            scaler = bundle["scaler"]
            device = bundle["device"]
            loaded_models = bundle["models"]
            lec_model = bundle["lec_model"]

            # Dynamically adapt the input span to reduce endpoint artifacts.
            if data_context.get("type") == "simulation" and "full_data" in data_context:
                t = data_context["current_index"]
                s_idx = data_context["station_index"]
                start_idx = max(0, t - 1000)
                raw_input = data_context["full_data"][start_idx:t, s_idx, 0].reshape(-1, 1)
            else:
                raw_input = hist_df["y"].values.reshape(-1, 1)
                if len(raw_input) < 168:
                    mirrored_part = np.flip(raw_input, axis=0)
                    raw_input = np.concatenate((mirrored_part, raw_input), axis=0)
                    st.toast("Realtime history is short. Applied mirrored extension to stabilize decomposition.", icon=None)

            scaled_input = scaler.transform(raw_input).flatten()

            with st.spinner("Running signal decomposition..."):
                imfs_a, imfs_b = fast_decompose(tuple(scaled_input))

            def get_branch_pred(models_list, imfs):
                total_pred = np.zeros(12)
                for model_name, imf_idx in models_list:
                    seq = imfs[imf_idx][-12:] if imf_idx < len(imfs) else np.zeros(12)
                    input_tensor = torch.tensor(seq.reshape(1, 12, 1), dtype=torch.float32).to(device)
                    model = loaded_models.get(model_name)
                    if model is None:
                        continue
                    with torch.no_grad():
                        total_pred += model(input_tensor).cpu().numpy().flatten()
                return total_pred

            w1 = saved_config.get("w1", 0.5)
            pred_hybrid_scaled = w1 * get_branch_pred(saved_config["models_a"], imfs_a) + (1 - w1) * get_branch_pred(
                saved_config["models_b"], imfs_b
            )
            final_pred_test = scaler.inverse_transform(pred_hybrid_scaled.reshape(-1, 1)).flatten()[:steps]

            # Trigger local error correction when the first-step jump is large.
            last_known_true = hist_df["y"].iloc[-1]
            current_pred_val = final_pred_test[0]
            gamma = abs(current_pred_val - last_known_true)

            lec_threshold = get_lec_threshold(steps)

            if gamma >= lec_threshold:
                st.toast(
                    f"PM2.5 mutation detected (delta={gamma:.1f}, threshold={lec_threshold:.1f}). Triggering LEC correction.",
                    icon=None,
                )
                if lec_model is not None:
                    recent_12_true = hist_df["y"].iloc[-12:].values.reshape(-1, 1)
                    recent_scaled = scaler.transform(recent_12_true).flatten()
                    lec_input = torch.tensor(recent_scaled.reshape(1, 12, 1), dtype=torch.float32).to(device)

                    with torch.no_grad():
                        err_pred_scaled = lec_model(lec_input).cpu().numpy().flatten()
                        final_pred_test += err_pred_scaled[:steps]
            else:
                st.toast("The transition looks smooth. LEC correction was not required.", icon=None)

            result_df = pd.DataFrame({"ds": future_dates, "yhat": final_pred_test})
            result_df["yhat"] = result_df["yhat"].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"BiLSTM-Hybrid inference failed: {e}")
            import traceback

            st.write(traceback.format_exc())
            return None

    # ------------------------------------------
    # D. DiffSTG
    # ------------------------------------------
    if selected_model_key == "DIFFUSION":
        if data_context.get("type") != "simulation":
            return None

        try:
            from easydict import EasyDict as edict
            from algorithm.dataset import CleanDataset, TrafficDataset

            model, device = load_diffstg_model(cfg["file"])

            diff_config = edict()
            diff_config.data = edict(
                {
                    "name": "AIR_BJ",
                    "feature_file": os.path.join(cfg["data_dir"], "flow.npy"),
                    "spatial": os.path.join(cfg["data_dir"], "adj.npy"),
                    "num_features": 1,
                    "num_vertices": 19,
                    "points_per_hour": 1,
                    "val_start_idx": int(99984 * 0.6),
                    "test_start_idx": int(99984 * 0.8),
                }
            )
            diff_config.model = edict({"T_p": 12, "T_h": 12, "V": 19, "F": 1, "week_len": 7, "day_len": 24})
            diff_config.device = device

            clean_data = CleanDataset(diff_config)
            t = data_context["current_index"]
            target_s_idx = data_context["station_index"]

            dataset = TrafficDataset(clean_data, (t - 12, t - 11), diff_config)
            future, history, pos_w, pos_d = dataset[0]

            history = torch.FloatTensor(history).unsqueeze(0).to(device)
            future_ph = torch.zeros_like(history).to(device)

            if pos_w is not None:
                pos_w = torch.FloatTensor(pos_w).unsqueeze(0).to(device)
            if pos_d is not None:
                pos_d = torch.FloatTensor(pos_d).unsqueeze(0).to(device)

            x_masked = torch.cat((history, future_ph), dim=1).transpose(1, 3).to(device)

            with torch.no_grad():
                prediction = model((x_masked, pos_w, pos_d), 8)

            pred_cpu = prediction[0, :, 0, target_s_idx, -12:].cpu().numpy()

            mean_val, std_val = clean_data.mean, clean_data.std
            pred_real = pred_cpu * std_val + mean_val

            pred_mean = np.mean(pred_real, axis=0)[:steps]
            pred_p5 = np.percentile(pred_real, 5, axis=0)[:steps]
            pred_p95 = np.percentile(pred_real, 95, axis=0)[:steps]

            result_df = pd.DataFrame({"ds": future_dates, "yhat": pred_mean, "y_lower": pred_p5, "y_upper": pred_p95})
            result_df["yhat"] = result_df["yhat"].clip(lower=0)
            result_df["y_lower"] = result_df["y_lower"].clip(lower=0)
            result_df["y_upper"] = result_df["y_upper"].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"DiffSTG inference failed: {e}")
            import traceback

            st.write(traceback.format_exc())
            return None

    # ------------------------------------------
    # E. BiLSTM-48H
    # ------------------------------------------
    if selected_model_key == "MIXED_48H":
        if len(hist_df) < 48:
            st.error("The 48-hour model requires at least 48 hours of history.")
            return None

        try:
            from PyEMD import CEEMDAN, EEMD

            bundle = load_hybrid_bundle(cfg["dir"], cfg["config"], cfg["scaler"], hidden_size=96, output_size=48)
            if bundle is None:
                return None

            saved_config = bundle["config"]
            scaler = bundle["scaler"]
            device = bundle["device"]
            loaded_models = bundle["models"]

            raw_input = hist_df["y"].values[-48:].reshape(-1, 1)
            scaled_input = scaler.transform(raw_input).flatten()

            with st.spinner("Running 48-hour signal decomposition..."):
                imfs_a, imfs_b = fast_decompose_48h(tuple(scaled_input))

            def get_branch_pred_48h(models_list, imfs):
                total_pred = np.zeros(48)
                for model_name, imf_idx in models_list:
                    seq = imfs[imf_idx][-48:] if imf_idx < len(imfs) else np.zeros(48)
                    input_tensor = torch.tensor(seq.reshape(1, 48, 1), dtype=torch.float32).to(device)
                    model = loaded_models.get(model_name)
                    if model is None:
                        continue
                    with torch.no_grad():
                        total_pred += model(input_tensor).cpu().numpy().flatten()
                return total_pred

            w1 = saved_config.get("w1", 0.5)
            pred_scaled = w1 * get_branch_pred_48h(saved_config["models_a"], imfs_a) + (1 - w1) * get_branch_pred_48h(
                saved_config["models_b"], imfs_b
            )
            final_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[:steps]

            result_df = pd.DataFrame({"ds": future_dates, "yhat": final_pred})
            result_df["yhat"] = result_df["yhat"].clip(lower=0)
            return result_df
        except Exception as e:
            st.error(f"BiLSTM-48H inference failed: {e}")
            import traceback

            st.write(traceback.format_exc())
            return None

    return None


def load_and_predict(selected_model_key, data_context, steps=12):
    """
    Thread-safe wrapper to prevent concurrent execution of heavy models 
    due to Streamlit's rapid rerun behavior.
    """
    # Fast-fail for models trying to run in unsupported environments to avoid lock contention
    if selected_model_key == "DIFFUSION" and data_context.get("type") != "simulation":
        return None

    with GLOBAL_ML_LOCK:
        return _inner_load_and_predict(selected_model_key, data_context, steps)


def predict_48h_pollution_alert(data_context):
    """Generate a 48-hour pollution alert from regression forecasts plus threshold classification."""
    sys.path.insert(0, BASE_DIR)
    from aqi_classifier import AQI_LEVELS, get_health_advice, pm25_to_aqi_level

    hist_df = data_context.get("history")
    if hist_df is None or len(hist_df) < 48:
        return None

    pred_df = load_and_predict("MIXED_48H", data_context, steps=48)
    if pred_df is None or pred_df.empty:
        return None

    hourly_levels = [pm25_to_aqi_level(pm25) for pm25 in pred_df["yhat"].values]
    pollution_hours = [(i, lvl) for i, lvl in enumerate(hourly_levels) if lvl >= 2]

    if not pollution_hours:
        return None

    max_level = max(lvl for _, lvl in pollution_hours)
    max_level_info = AQI_LEVELS[max_level]

    pollution_start = pollution_hours[0][0]
    pollution_end = pollution_hours[-1][0]
    duration_hours = pollution_end - pollution_start + 1

    return {
        "max_level": max_level,
        "level_name": max_level_info["name_cn"],
        "level_color": max_level_info["color"],
        "start_hour": pollution_start,
        "end_hour": pollution_end,
        "duration": duration_hours,
        "health_advice": get_health_advice(max_level),
        "hourly_levels": hourly_levels,
        "hourly_pm25": pred_df["yhat"].values,
    }
