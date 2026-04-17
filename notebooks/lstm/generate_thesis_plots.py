# generate_thesis_plots.py
# Generate BiLSTM-Hybrid forecast figures for the thesis

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PyEMD import CEEMDAN, EEMD
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../../models/lstm1')
DATA_PATH = os.path.join(BASE_DIR, '../../data/processed/lstm/Dongsi_PM2.5.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '../../results/figures_summary')

SEQ_LEN = 12
TARGET_LEN = 12
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network definition
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

class SingleComponentDataset(Dataset):
    def __init__(self, data_seq, seq_len, target_len):
        self.data = torch.FloatTensor(data_seq).view(-1, 1)
        self.seq_len = seq_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.target_len]
        return x, y.squeeze(-1)

def predict_branch(models_list, imfs_data, model_dir, device):
    total_samples = len(imfs_data[0]) - SEQ_LEN - TARGET_LEN + 1
    total_pred = np.zeros((total_samples, TARGET_LEN))
    for model_name, imf_idx in models_list:
        if imf_idx >= len(imfs_data):
            print(f"   [WARNING] Skipping {model_name}: IMF index {imf_idx} out of range (max: {len(imfs_data)-1})")
            continue
        input_data = imfs_data[imf_idx]
        model = BiLSTMNet(hidden_size=64, output_size=TARGET_LEN).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pth")))
        model.eval()
        dataset = SingleComponentDataset(input_data, SEQ_LEN, TARGET_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for x_batch, _ in loader:
                output = model(x_batch.to(device))
                preds.append(output.cpu().numpy())
        total_pred += np.concatenate(preds, axis=0)
    return total_pred

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(">>> [Step 1] Loading Data and Models...")
    df = pd.read_csv(DATA_PATH)
    pm25_col = 'PM2.5' if 'PM2.5' in df.columns else [c for c in df.columns if c != 'time'][0]
    raw_values = df[pm25_col].values.reshape(-1, 1)

    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_Dongsi.pkl'))
    scaled_data = scaler.transform(raw_values).flatten()

    config = joblib.load(os.path.join(MODEL_DIR, 'config.pkl'))
    w1 = config['w1']
    models_a = config['models_a']
    models_b = config['models_b']

    print(f"   Loaded hybrid weights: w_A={w1:.2f}, w_B={1-w1:.2f}")
    print(f"   Total models: {len(models_a)} CEEMDAN + {len(models_b)} RLMD")

    print("\n>>> [Step 2] Signal Decomposition (this may take 10-15 minutes)...")
    ceemdan = CEEMDAN(trials=5, processes=1)
    imfs_a_all = ceemdan(scaled_data)
    print(f"   CEEMDAN complete: {len(imfs_a_all)} IMFs")

    rlmd = EEMD(trials=2, processes=1)
    imfs_b_all = rlmd(scaled_data)
    print(f"   EEMD complete: {len(imfs_b_all)} IMFs")

    total_len = len(scaled_data)
    test_start = int(total_len * 0.90)
    test_data = scaled_data[test_start:]
    imfs_test_a = [imf[test_start:] for imf in imfs_a_all]
    imfs_test_b = [imf[test_start:] for imf in imfs_b_all]

    print(f"   Test set size: {len(test_data)} samples")

    print("\n>>> [Step 3] Generating Predictions...")
    preds_a = predict_branch(models_a, imfs_test_a, MODEL_DIR, DEVICE)
    preds_b = predict_branch(models_b, imfs_test_b, MODEL_DIR, DEVICE)
    final_pred = w1 * preds_a + (1-w1) * preds_b

    temp_ds = SingleComponentDataset(test_data, SEQ_LEN, TARGET_LEN)
    y_true_list = []
    for _, y in DataLoader(temp_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
        y_true_list.append(y.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    print(f"   Predictions shape: {final_pred.shape}")

    print("\n>>> [Step 4] Generating Thesis Plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    horizons = [3, 6, 12]
    for h in horizons:
        fig, ax = plt.subplots(figsize=(14, 6))

        plot_len = min(200, len(y_true))
        y_true_h = scaler.inverse_transform(y_true[:plot_len, h-1].reshape(-1, 1)).flatten()
        y_pred_h = scaler.inverse_transform(final_pred[:plot_len, h-1].reshape(-1, 1)).flatten()

        time_index = np.arange(plot_len)

        mae = mean_absolute_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        r2 = r2_score(y_true_h, y_pred_h)

        ax.plot(time_index, y_true_h, 'k.-', label='Ground Truth', linewidth=2, markersize=3)
        ax.plot(time_index, y_pred_h, 'g--', label='BiLSTM-Hybrid Forecast', linewidth=2.5)

        ax.set_title(f'BiLSTM-Hybrid {h}-Hour Forecast vs Actuals\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('PM2.5 Concentration (ug/m3)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'bilstm_forecast_{h}h.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: {output_path}")
        plt.close()

    print("\n>>> [Complete] All BiLSTM-Hybrid thesis plots generated!")

if __name__ == '__main__':
    main()
