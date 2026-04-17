# generate_thesis_plots.py
# Generate DiffSTG forecast figures for the thesis

import sys
import os
BASE_DIR = os.getcwd()
sys.path.insert(0, os.path.join(BASE_DIR, '../../notebooks/diffusion'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(BASE_DIR, '../../results/figures_summary')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(">>> [Step 1] Loading DiffSTG Results...")

# DiffSTG results are usually saved under models/diffusion/results/
RESULTS_DIR = os.path.join(BASE_DIR, '../../models/diffusion/results')

# Find the latest prediction result files
result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.npy')]
if not result_files:
    print("   [ERROR] No DiffSTG result files found!")
    print("   Please run the DiffSTG model first to generate predictions.")
    sys.exit(1)

# Load prediction results (assuming predictions.npy and ground_truth.npy)
try:
    predictions = np.load(os.path.join(RESULTS_DIR, 'predictions.npy'))
    ground_truth = np.load(os.path.join(RESULTS_DIR, 'ground_truth.npy'))
    print(f"   Loaded predictions: {predictions.shape}")
    print(f"   Loaded ground truth: {ground_truth.shape}")
except:
    print("   [WARNING] Standard result files not found. Using evaluation summary data.")
    # If saved predictions are missing, only the evaluation summary is available.
    print("   Skipping visualization generation.")
    sys.exit(0)

# Generate figures
print("\n>>> [Step 2] Generating Thesis Plots...")
plt.style.use('seaborn-v0_8-whitegrid')

horizons = [3, 6, 12]
for h in horizons:
    fig, ax = plt.subplots(figsize=(14, 6))

    plot_len = min(200, len(ground_truth))
    y_true_h = ground_truth[:plot_len, h-1]
    y_pred_h = predictions[:plot_len, h-1]

    time_index = np.arange(plot_len)

    mae = mean_absolute_error(y_true_h, y_pred_h)
    rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    r2 = r2_score(y_true_h, y_pred_h)

    ax.plot(time_index, y_true_h, 'k.-', label='Ground Truth', linewidth=2, markersize=3)
    ax.plot(time_index, y_pred_h, 'm--', label='DiffSTG Forecast', linewidth=2.5)

    ax.set_title(f'DiffSTG {h}-Hour Forecast vs Actuals\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (ug/m3)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'diffstg_forecast_{h}h.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   [OK] Saved: {output_path}")
    plt.close()

print("\n>>> [Complete] All DiffSTG thesis plots generated!")
