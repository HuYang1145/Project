# generate_thesis_plots.py
# Generate ARIMA forecast figures for the thesis

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data/processed/lstm/Dongsi_PM2.5.csv')
MODEL_PATH = os.path.join(BASE_DIR, '../../models/arima/Dongsi_ARIMA.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, '../../results/figures_summary')
THESIS_OUTPUT_DIR = os.path.join(BASE_DIR, '../../毕业论文/figures/ch4_results')

TRAIN_RATIO = 0.8
HORIZONS = [3, 6, 12]  # 3h, 6h, 12h

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THESIS_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Load data and model
# ==========================================
print(">>> [Step 1] Loading Data and Model...")
df = pd.read_csv(DATA_PATH)
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
elif 'ds' in df.columns:
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

target_col = 'PM2.5' if 'PM2.5' in df.columns else ('y' if 'y' in df.columns else df.columns[0])
pm25_data = df[target_col]

# Split the train and test sets
split_idx = int(len(pm25_data) * TRAIN_RATIO)
test_data = pm25_data.iloc[split_idx:]

# Load the model
with open(MODEL_PATH, 'rb') as f:
    model_fit = pickle.load(f)

print(f"   Data loaded: {len(test_data)} test samples")
print(f"   Model loaded: ARIMA model")

# ==========================================
# 2. Generate forecasts
# ==========================================
print("\n>>> [Step 2] Preparing Evaluation Window...")
max_h = max(HORIZONS)
max_start = len(test_data) - max_h
if max_start <= 0:
    raise RuntimeError("Test set is too short for the configured horizons.")

# Select a representative window with strong variation to avoid trivial flat segments.
test_vals = test_data.values
window_std = np.array([np.std(test_vals[i:i + max_h]) for i in range(max_start + 1)])
start_pos = int(np.argmax(window_std))
print(f"   Selected test window start offset: {start_pos} (std={window_std[start_pos]:.2f})")


def forecast_at_horizon(h):
    history_end = split_idx + start_pos
    history_values = pm25_data.iloc[:history_end].values
    y_true = test_data.iloc[start_pos:start_pos + h].values
    time_index = test_data.index[start_pos:start_pos + h]
    updated_model = model_fit.apply(history_values)
    pred = updated_model.forecast(steps=h)
    y_pred = pred.values if hasattr(pred, "values") else np.asarray(pred)
    return time_index, y_true, y_pred

# ==========================================
# 3. Generate figures
# ==========================================
print("\n>>> [Step 3] Generating Thesis Plots...")

plt.style.use('seaborn-v0_8-whitegrid')

for h in HORIZONS:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract the data
    time_index, y_true, y_pred = forecast_at_horizon(h)

    # Plot
    ax.plot(time_index, y_true, 'k.-', label='Ground Truth', linewidth=2, markersize=4)
    ax.plot(time_index, y_pred, 'r--', label='ARIMA Forecast', linewidth=2.5)

    # Title and labels
    ax.set_title(f'ARIMA Forecast vs Actuals ({h}-Hour Horizon)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, f'arima_forecast_{h}h.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved: {output_path}")
    plt.close()

# Create a single panel figure for the chapter to reduce figure count.
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 5.2), sharey=True)

for ax, h in zip(axes, HORIZONS):
    time_index, y_true, y_pred = forecast_at_horizon(h)

    ax.plot(time_index, y_true, 'k.-', label='Ground Truth', linewidth=1.8, markersize=3)
    ax.plot(time_index, y_pred, 'r--', label='ARIMA Forecast', linewidth=2.0)
    ax.set_title(f'{h}-Hour Horizon', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=10)
axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('ARIMA Forecast vs Actuals Across Multiple Horizons', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

panel_name = 'arima_forecast_panel.png'
panel_output = os.path.join(OUTPUT_DIR, panel_name)
panel_thesis_output = os.path.join(THESIS_OUTPUT_DIR, panel_name)
fig.savefig(panel_output, dpi=300, bbox_inches='tight')
fig.savefig(panel_thesis_output, dpi=300, bbox_inches='tight')
print(f"   [OK] Saved: {panel_output}")
print(f"   [OK] Saved: {panel_thesis_output}")
plt.close(fig)

print("\n>>> [Complete] All ARIMA thesis plots generated!")
