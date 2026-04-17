import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PyEMD import CEEMDAN, EEMD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


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


class SingleComponentDataset(Dataset):
    def __init__(self, data_seq, seq_len=12, target_len=12):
        self.data = torch.FloatTensor(data_seq).view(-1, 1)
        self.seq_len = seq_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.target_len]
        return x, y.squeeze(-1)


def predict_with_model(model, data_seq, seq_len=12, target_len=12, batch_size=64):
    ds = SingleComponentDataset(data_seq, seq_len=seq_len, target_len=target_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(next(model.parameters()).device)).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([])


def predict_branch(models_list, imfs_data, model_cache):
    total_samples = len(imfs_data[0]) - 12 - 12 + 1
    total_pred = np.zeros((total_samples, 12))
    for model_name, imf_idx in models_list:
        if imf_idx >= len(imfs_data):
            continue
        model = model_cache.get(model_name)
        if model is None:
            continue
        input_data = imfs_data[imf_idx]
        total_pred += predict_with_model(model, input_data)
    return total_pred


def contiguous_segments(mask):
    segments = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def compute_event_metrics(y_true_1d, y_pred_1d, event_threshold=75.0):
    event_mask = y_true_1d >= event_threshold
    segments = contiguous_segments(event_mask)
    if not segments:
        return np.nan, np.nan, np.nan, "[nan, nan]"

    recalls = []
    peak_lags = []
    leads = []
    for seg_start, seg_end in segments:
        detected = np.any(y_pred_1d[seg_start : seg_end + 1] >= event_threshold)
        recalls.append(1.0 if detected else 0.0)

        true_peak_idx = seg_start + int(np.argmax(y_true_1d[seg_start : seg_end + 1]))
        search_start = max(0, seg_start - 3)
        search_end = min(len(y_pred_1d) - 1, seg_end + 3)
        pred_peak_idx = search_start + int(np.argmax(y_pred_1d[search_start : search_end + 1]))

        lag = pred_peak_idx - true_peak_idx
        lead = -lag
        peak_lags.append(lag)
        leads.append(lead)

    mean_recall = float(np.mean(recalls))
    mean_abs_peak_lag = float(np.mean(np.abs(peak_lags)))
    mean_lead = float(np.mean(leads))
    q1, q3 = np.percentile(leads, [25, 75])
    lead_iqr = f"[{q1:.0f}h, {q3:.0f}h]"
    return mean_recall, mean_abs_peak_lag, mean_lead, lead_iqr


def permutation_entropy_norm(x, order=3, delay=1):
    x = np.asarray(x, dtype=float)
    n = len(x) - delay * (order - 1)
    if n <= 0:
        return np.nan

    counts = {}
    for i in range(n):
        window = x[i : i + order * delay : delay]
        ranks = tuple(np.argsort(window))
        counts[ranks] = counts.get(ranks, 0) + 1

    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    pe = -np.sum(probs * np.log2(probs))
    pe_max = math.log2(math.factorial(order))
    return float(pe / pe_max) if pe_max > 0 else np.nan


def metrics_row(y_true_raw, y_pred_raw, label):
    row = {"setting": label}
    for h in [3, 6, 12]:
        yt = y_true_raw[:, h - 1]
        yp = y_pred_raw[:, h - 1]
        row[f"MAE_{h}h"] = float(mean_absolute_error(yt, yp))
        row[f"RMSE_{h}h"] = float(np.sqrt(mean_squared_error(yt, yp)))
        row[f"R2_{h}h"] = float(r2_score(yt, yp))
    recall, peak_lag_abs, mean_lead, lead_iqr = compute_event_metrics(y_true_raw[:, 2], y_pred_raw[:, 2], event_threshold=75.0)
    row["Recall@3h(>=75)"] = recall
    row["Mean|PeakLag|@3h"] = peak_lag_abs
    row["MeanLead@3h"] = mean_lead
    row["LeadIQR@3h"] = lead_iqr
    return row


def apply_lec(pred_base, y_true_scaled, scaler, lec_model, trigger_threshold=8.0):
    valid_start = 12
    residual_1step = y_true_scaled[:, 0] - pred_base[:, 0]
    res_1d = np.zeros(len(y_true_scaled) + 12 + 12 - 1)
    res_1d[valid_start : valid_start + len(residual_1step)] = residual_1step
    err_pred = predict_with_model(lec_model, res_1d)

    min_len = min(len(y_true_scaled), len(err_pred), len(pred_base))
    corrected = pred_base[:min_len].copy()

    for i in range(1, min_len):
        current_pred_val = scaler.inverse_transform([[pred_base[i, 0]]])[0][0]
        prev_true_val = scaler.inverse_transform([[y_true_scaled[i - 1, 0]]])[0][0]
        gamma = abs(current_pred_val - prev_true_val)
        if gamma >= trigger_threshold:
            corrected[i, :] += err_pred[i, :]
    return corrected[:min_len], min_len


def train_lec_model(residual_series, lr, epochs, device):
    model = BiLSTMNet(hidden_size=64, output_size=12).to(device)
    ds = SingleComponentDataset(residual_series, seq_len=12, target_len=12)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "processed", "lstm", "Dongsi_PM2.5.csv")
    model_dir = os.path.join(base_dir, "models", "lstm1")
    out_dir = os.path.join(base_dir, "results", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    raw_values = df["PM2.5"].astype(float).values.reshape(-1, 1)
    scaler = joblib.load(os.path.join(model_dir, "scaler_Dongsi.pkl"))
    config = joblib.load(os.path.join(model_dir, "config.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaled_data = scaler.transform(raw_values).flatten()

    total_len = len(scaled_data)
    train_end = int(total_len * 0.70)
    lec_end = int(total_len * 0.90)
    lec_data = scaled_data[train_end:lec_end]
    test_data = scaled_data[lec_end:]

    print("Decomposing full series for PE/branch sensitivity...")
    ceemdan = CEEMDAN(trials=10, parallel=False)
    eemd = EEMD(trials=5, parallel=False)
    imfs_a_all = ceemdan(scaled_data)
    imfs_b_all = eemd(scaled_data)

    imfs_lec_a = [imf[train_end:lec_end] for imf in imfs_a_all]
    imfs_test_a = [imf[lec_end:] for imf in imfs_a_all]
    imfs_lec_b = [imf[train_end:lec_end] for imf in imfs_b_all]
    imfs_test_b = [imf[lec_end:] for imf in imfs_b_all]

    print("Loading trained component models...")
    model_cache = {}
    for model_name, _ in config["models_a"] + config["models_b"]:
        if model_name in model_cache:
            continue
        m = BiLSTMNet(hidden_size=64, output_size=12).to(device)
        m.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pth"), map_location=device))
        m.eval()
        model_cache[model_name] = m

    base_lec_ds = SingleComponentDataset(lec_data, seq_len=12, target_len=12)
    y_lec_list = []
    for _, yb in DataLoader(base_lec_ds, batch_size=64, shuffle=False, num_workers=0):
        y_lec_list.append(yb.numpy())
    y_lec_true = np.concatenate(y_lec_list, axis=0).squeeze()

    test_ds = SingleComponentDataset(test_data, seq_len=12, target_len=12)
    y_test_list = []
    for _, yb in DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0):
        y_test_list.append(yb.numpy())
    y_test_true = np.concatenate(y_test_list, axis=0).squeeze()

    # -------------------------------------------------
    # A) PE threshold sensitivity (with fixed LEC model)
    # -------------------------------------------------
    print("Running PE threshold sensitivity...")
    pe_a = {i: permutation_entropy_norm(imfs_a_all[i][:train_end], order=3, delay=1) for i in range(len(imfs_a_all))}
    pe_b = {i: permutation_entropy_norm(imfs_b_all[i][:train_end], order=3, delay=1) for i in range(len(imfs_b_all))}

    lec_model_fixed = BiLSTMNet(hidden_size=64, output_size=12).to(device)
    lec_model_fixed.load_state_dict(torch.load(os.path.join(model_dir, "LEC_Model.pth"), map_location=device))
    lec_model_fixed.eval()

    pe_rows = []
    pe_grid = [0.80, 0.85, 0.90, 0.95]
    for thr in pe_grid:
        models_a = [(n, idx) for (n, idx) in config["models_a"] if (idx in pe_a and pe_a[idx] <= thr)]
        models_b = [(n, idx) for (n, idx) in config["models_b"] if (idx in pe_b and pe_b[idx] <= thr)]
        if not models_a and not models_b:
            continue

        pred_lec_a = predict_branch(models_a, imfs_lec_a, model_cache) if models_a else np.zeros((len(y_lec_true), 12))
        pred_lec_b = predict_branch(models_b, imfs_lec_b, model_cache) if models_b else np.zeros((len(y_lec_true), 12))
        pred_test_a = predict_branch(models_a, imfs_test_a, model_cache) if models_a else np.zeros((len(y_test_true), 12))
        pred_test_b = predict_branch(models_b, imfs_test_b, model_cache) if models_b else np.zeros((len(y_test_true), 12))

        min_len_lec = min(len(y_lec_true), len(pred_lec_a), len(pred_lec_b))
        y_lec_cut = y_lec_true[:min_len_lec]
        pa = pred_lec_a[:min_len_lec]
        pb = pred_lec_b[:min_len_lec]
        best_w = 0.5
        best_mse = float("inf")
        for w in np.linspace(0, 1, 21):
            comb = w * pa + (1.0 - w) * pb
            mse = float(np.mean((y_lec_cut - comb) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_w = float(w)

        min_len_test = min(len(y_test_true), len(pred_test_a), len(pred_test_b))
        pred_base_test = best_w * pred_test_a[:min_len_test] + (1.0 - best_w) * pred_test_b[:min_len_test]
        y_test_cut = y_test_true[:min_len_test]

        pred_corr, min_len_corr = apply_lec(pred_base_test, y_test_cut, scaler, lec_model_fixed, trigger_threshold=8.0)
        y_true_raw = scaler.inverse_transform(y_test_cut[:min_len_corr].reshape(-1, 1)).reshape(min_len_corr, 12)
        y_pred_raw = scaler.inverse_transform(pred_corr.reshape(-1, 1)).reshape(min_len_corr, 12)
        y_true_raw = np.clip(y_true_raw, 0, np.inf)
        y_pred_raw = np.clip(y_pred_raw, 0, np.inf)

        row = metrics_row(y_true_raw, y_pred_raw, label=f"PE<={thr:.2f}")
        row["PE_threshold"] = thr
        row["w_A_recalibrated"] = best_w
        row["n_models_A"] = len(models_a)
        row["n_models_B"] = len(models_b)
        pe_rows.append(row)

    pe_df = pd.DataFrame(pe_rows)
    pe_csv = os.path.join(out_dir, "pe_threshold_sensitivity.csv")
    pe_df.to_csv(pe_csv, index=False, encoding="utf-8")

    # -------------------------------------------------
    # B) LEC lr/epochs sensitivity (retrain LEC only)
    # -------------------------------------------------
    print("Running LEC lr/epochs sensitivity...")
    base_pred_lec_a = predict_branch(config["models_a"], imfs_lec_a, model_cache)
    base_pred_lec_b = predict_branch(config["models_b"], imfs_lec_b, model_cache)
    base_pred_test_a = predict_branch(config["models_a"], imfs_test_a, model_cache)
    base_pred_test_b = predict_branch(config["models_b"], imfs_test_b, model_cache)
    w1 = float(config["w1"])
    base_pred_lec = w1 * base_pred_lec_a + (1.0 - w1) * base_pred_lec_b
    base_pred_test = w1 * base_pred_test_a + (1.0 - w1) * base_pred_test_b

    min_len_lec = min(len(y_lec_true), len(base_pred_lec))
    min_len_test = min(len(y_test_true), len(base_pred_test))
    y_lec_cut = y_lec_true[:min_len_lec]
    y_test_cut = y_test_true[:min_len_test]
    base_pred_lec = base_pred_lec[:min_len_lec]
    base_pred_test = base_pred_test[:min_len_test]

    residual_1step_lec = y_lec_cut[:, 0] - base_pred_lec[:, 0]
    res_series = np.zeros(len(lec_data))
    res_series[12 : 12 + len(residual_1step_lec)] = residual_1step_lec

    hp_configs = [
        (1e-4, 50),
        (5e-4, 20),
        (5e-4, 50),
        (5e-4, 80),
        (1e-3, 50),
    ]
    hp_rows = []
    for lr, epochs in hp_configs:
        print(f"  Training LEC model: lr={lr}, epochs={epochs}")
        lec_model = train_lec_model(res_series, lr=lr, epochs=epochs, device=device)
        pred_corr, min_len_corr = apply_lec(base_pred_test, y_test_cut, scaler, lec_model, trigger_threshold=8.0)

        y_true_raw = scaler.inverse_transform(y_test_cut[:min_len_corr].reshape(-1, 1)).reshape(min_len_corr, 12)
        y_pred_raw = scaler.inverse_transform(pred_corr.reshape(-1, 1)).reshape(min_len_corr, 12)
        y_true_raw = np.clip(y_true_raw, 0, np.inf)
        y_pred_raw = np.clip(y_pred_raw, 0, np.inf)

        row = metrics_row(y_true_raw, y_pred_raw, label=f"lr={lr},epochs={epochs}")
        row["learning_rate"] = lr
        row["epochs"] = epochs
        hp_rows.append(row)

    hp_df = pd.DataFrame(hp_rows)
    hp_csv = os.path.join(out_dir, "lec_lr_epoch_sensitivity.csv")
    hp_df.to_csv(hp_csv, index=False, encoding="utf-8")

    summary_txt = os.path.join(out_dir, "hyperparam_sensitivity_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Hyperparameter Sensitivity (PE / LEC lr&epochs)\n")
        f.write("=" * 66 + "\n\n")
        f.write("[PE Threshold]\n")
        f.write(pe_df.to_string(index=False))
        f.write("\n\n[LEC lr/epochs]\n")
        f.write(hp_df.to_string(index=False))
        f.write("\n")

    print(f"[OK] Saved: {pe_csv}")
    print(f"[OK] Saved: {hp_csv}")
    print(f"[OK] Saved: {summary_txt}")


if __name__ == "__main__":
    main()
