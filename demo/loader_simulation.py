# loader_simulation.py
# Strict local simulation using slices from the test set

import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent directory
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "diffusion", "AIR_BJ")
FLOW_PATH = os.path.join(DATA_DIR, "flow.npy")

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


@st.cache_data
def load_full_data():
    """Load and cache the full diffusion dataset."""
    if not os.path.exists(FLOW_PATH):
        st.error(f"Local data file not found: {FLOW_PATH}")
        return None
    return np.load(FLOW_PATH)


@st.cache_data
def load_neuralprophet_data():
    """Load and cache the NeuralProphet backtesting frame."""
    if not os.path.exists(NP_DATA_PATH):
        st.error(f"NeuralProphet data file not found: {NP_DATA_PATH}")
        return None

    df = pd.read_csv(NP_DATA_PATH)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _ensure_shared_simulation_time():
    """
    Keep one shared cutoff time for all simulation loaders until the user
    explicitly requests a new random sample.
    """
    df = load_neuralprophet_data()
    if df is None or df.empty:
        return None, None, None

    total_len = len(df)
    test_start_idx = int(total_len * 0.8)
    min_idx = test_start_idx + 168
    max_idx = total_len - 24

    max_offset = max_idx - min_idx
    if max_offset < 0:
        st.error("NeuralProphet simulation window is invalid.")
        return None, None, None

    if "sim_hour_offset" not in st.session_state or st.session_state.sim_hour_offset > max_offset:
        st.session_state.sim_hour_offset = random.randint(0, max_offset)

    offset = st.session_state.sim_hour_offset
    cutoff_time = df.iloc[min_idx + offset - 1]["ds"]
    st.session_state.sim_cutoff_time = cutoff_time
    return cutoff_time, offset, max_offset


def get_simulation_data(target_station="Dongsi"):
    """Sample a strict test-set slice with a shared simulation cutoff time."""
    data = load_full_data()  # (total_time, 19, 1)
    if data is None:
        return None, None, None

    cutoff_time, shared_offset, shared_max_offset = _ensure_shared_simulation_time()
    if cutoff_time is None:
        return None, None, None

    total_len = data.shape[0]

    # Restrict sampling to the final 20% test split.
    test_start_idx = int(total_len * 0.8)

    # The split point must leave 168 hours behind and 12 hours ahead.
    min_idx = test_start_idx + 168
    max_idx = total_len - 12

    max_common_offset = max_idx - min_idx
    if max_common_offset < 0:
        st.error("Diffusion simulation window is invalid.")
        return None, None, None

    ratio = shared_offset / max(shared_max_offset, 1)
    t = min_idx + int(round(ratio * max_common_offset))
    t = max(min_idx, min(t, max_idx))

    if target_station in STATION_LIST:
        s_idx = STATION_LIST.index(target_station)
    else:
        s_idx = 0

    # A. History frame for Prophet/LSTM/ARIMA
    hist_vals = data[t - 168 : t, s_idx, 0]
    hist_dates = [cutoff_time - timedelta(hours=167 - i) for i in range(168)]
    history_df = pd.DataFrame({"ds": hist_dates, "y": hist_vals})

    # B. Future ground truth
    gt_vals = data[t : t + 12, s_idx, 0]
    gt_dates = [cutoff_time + timedelta(hours=i + 1) for i in range(12)]
    ground_truth_df = pd.DataFrame({"ds": gt_dates, "y": gt_vals})

    # C. Synthetic weather data for Prophet simulation
    weather_dates = hist_dates + gt_dates
    weather_df = pd.DataFrame(
        {
            "ds": weather_dates,
            "TEMP": [15.0 + random.uniform(-5, 5) for _ in range(180)],
            "PRES": [1010.0 + random.uniform(-10, 10) for _ in range(180)],
            "DEWP": [5.0 + random.uniform(-2, 2) for _ in range(180)],
            "WSPM": [2.0 + random.uniform(-1, 3) for _ in range(180)],
            "RAIN": [0.0 for _ in range(180)],
        }
    )

    # D. Diffusion context using the full dataset
    context = {
        "type": "simulation",
        "full_data": data,
        "current_index": t,
        "station_index": s_idx,
        "weather": weather_df,
        "sample_key": f"common:{shared_offset}:{cutoff_time.isoformat()}",
        "simulated_cutoff_time": cutoff_time,
    }

    return history_df, ground_truth_df, context


def change_random_sample():
    """Force a new random sample while preserving the selected model."""
    for key in ("sim_t", "sim_hour_offset", "sim_cutoff_time"):
        if key in st.session_state:
            del st.session_state[key]
    # Keep locked_model_name so the user's model choice persists.


# Dedicated NeuralProphet simulation loader
NP_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "neuralprophet", "Aotizhongxin_neuralprophet.csv")


def get_neuralprophet_simulation_data():
    """Sample a slice from the cleaned NeuralProphet dataset."""
    df = load_neuralprophet_data()
    if df is None or df.empty:
        return None, None, None

    cutoff_time, shared_offset, _ = _ensure_shared_simulation_time()
    if cutoff_time is None:
        return None, None, None

    total_len = len(df)
    test_start_idx = int(total_len * 0.8)

    min_idx = test_start_idx + 168
    max_idx = total_len - 24

    t = min_idx + shared_offset
    t = max(min_idx, min(t, max_idx))

    history_df = df.iloc[t - 168 : t].copy()
    ground_truth_df = df.iloc[t : t + 24][["ds", "y"]].copy()
    weather_df = df.iloc[t : t + 24][["ds", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].copy()

    context = {
        "type": "simulation",
        "history": history_df[["ds", "y"]].copy(),
        "weather": weather_df,
        "sample_key": f"prophet:{shared_offset}:{cutoff_time.isoformat()}",
        "simulated_cutoff_time": cutoff_time,
    }

    return history_df[["ds", "y"]], ground_truth_df, context
