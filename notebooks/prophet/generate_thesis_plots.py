# generate_thesis_plots.py
# Generate Prophet forecast figures for the thesis.

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data/processed/prophet/Aotizhongxin_prophet.csv')
MODEL_PATH = os.path.join(BASE_DIR, '../../models/prophet/Aotizhongxin_prophet.joblib')
OUTPUT_DIR = os.path.join(BASE_DIR, '../../results/figures_summary')
THESIS_OUTPUT_DIR = os.path.join(BASE_DIR, '../../毕业论文/figures/ch4_results')

TRAIN_RATIO = 0.8
REGRESSORS = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
HORIZONS = [3, 6, 12]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THESIS_OUTPUT_DIR, exist_ok=True)


def save_to_both_dirs(fig, filename: str) -> None:
    output_path = os.path.join(OUTPUT_DIR, filename)
    thesis_output_path = os.path.join(THESIS_OUTPUT_DIR, filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(thesis_output_path, dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved: {output_path}")
    print(f"   [OK] Saved: {thesis_output_path}")


# ==========================================
# 1. Load data and model
# ==========================================
print(">>> [Step 1] Loading Data and Model...")
df = pd.read_csv(DATA_PATH)
if 'date' in df.columns:
    df.rename(columns={'date': 'ds'}, inplace=True)
elif 'datetime' in df.columns:
    df.rename(columns={'datetime': 'ds'}, inplace=True)
if 'PM2.5' in df.columns:
    df.rename(columns={'PM2.5': 'y'}, inplace=True)

df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)
df.ffill(inplace=True)

split_idx = int(len(df) * TRAIN_RATIO)
test_df = df.iloc[split_idx:].copy()

model = joblib.load(MODEL_PATH)
print(f"   Data loaded: {len(test_df)} test samples")
print("   Model loaded: Prophet with 5 regressors")

# ==========================================
# 2. Generate forecasts
# ==========================================
print("\n>>> [Step 2] Generating Forecasts...")
future_test = test_df[['ds'] + REGRESSORS].copy()
forecast = model.predict(future_test)

results = pd.DataFrame(
    {
        'ds': test_df['ds'].values,
        'y_true': test_df['y'].values,
        'y_pred': forecast['yhat'].values,
        'y_lower': forecast['yhat_lower'].values,
        'y_upper': forecast['yhat_upper'].values,
    }
)
results['y_pred'] = results['y_pred'].clip(lower=0)
results['y_lower'] = results['y_lower'].clip(lower=0)

# ==========================================
# 3. Generate per-horizon figures
# ==========================================
print("\n>>> [Step 3] Generating Per-Horizon Thesis Plots...")
plt.style.use('seaborn-v0_8-whitegrid')

for h in HORIZONS:
    fig, ax = plt.subplots(figsize=(14, 6))
    subset = results.iloc[:h]

    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    r2 = r2_score(subset['y_true'], subset['y_pred'])

    ax.plot(subset['ds'], subset['y_true'], 'k.-', label='Ground Truth', linewidth=2, markersize=4)
    ax.plot(subset['ds'], subset['y_pred'], 'b--', label='Prophet Forecast', linewidth=2.5)
    ax.fill_between(
        subset['ds'],
        subset['y_lower'],
        subset['y_upper'],
        color='blue',
        alpha=0.15,
        label='Uncertainty Interval',
    )
    ax.set_title(
        f'Prophet Forecast vs Actuals ({h}-Hour Horizon)\nMAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (ug/m3)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_to_both_dirs(fig, f'prophet_forecast_{h}h.png')
    plt.close(fig)

# ==========================================
# 4. Generate integrated triptych figure
# ==========================================
print("\n>>> [Step 4] Generating Integrated 3/6/12-Hour Triptych...")
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 5.4), sharey=True)

for ax, h in zip(axes, HORIZONS):
    subset = results.iloc[:h]
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
    r2 = r2_score(subset['y_true'], subset['y_pred'])

    ax.plot(subset['ds'], subset['y_true'], 'k.-', label='Ground Truth', linewidth=2.0, markersize=4)
    ax.plot(subset['ds'], subset['y_pred'], 'b--', label='Prophet Forecast', linewidth=2.2)
    ax.fill_between(
        subset['ds'],
        subset['y_lower'],
        subset['y_upper'],
        color='blue',
        alpha=0.15,
        label='Uncertainty Interval',
    )
    ax.set_title(f'{h}-Hour\nMAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].set_ylabel('PM2.5 Concentration (ug/m3)', fontsize=10)
axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('Prophet Forecast vs Ground Truth Across 3/6/12-Hour Horizons', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
save_to_both_dirs(fig, 'prophet_forecast_triptych_3_6_12h.png')
plt.close(fig)

print("\n>>> [Complete] All Prophet thesis plots generated!")
