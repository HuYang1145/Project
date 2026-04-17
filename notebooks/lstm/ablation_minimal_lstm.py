import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        return self.fc(out[:, -1, :])


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
        for x_batch, _ in loader:
            preds.append(model(x_batch.to(next(model.parameters()).device)).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([])


def load_model_cache(model_names, model_dir, device):
    cache = {}
    for model_name in model_names:
        if model_name in cache:
            continue
        model = BiLSTMNet(hidden_size=64, output_size=12).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pth"), map_location=device))
        model.eval()
        cache[model_name] = model
    return cache


def predict_branch(models_list, imfs_data, model_cache):
    total_samples = len(imfs_data[0]) - 12 - 12 + 1
    total_pred = np.zeros((total_samples, 12))
    for model_name, imf_idx in models_list:
        if imf_idx >= len(imfs_data):
            continue
        model = model_cache.get(model_name)
        if model is None:
            continue
        total_pred += predict_with_model(model, imfs_data[imf_idx])
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


def apply_lec(pred_base, y_true_scaled, scaler, lec_model, trigger_threshold=8.0):
    valid_start = 12
    residual_1step = y_true_scaled[:, 0] - pred_base[:, 0]
    res_1d = np.zeros(len(y_true_scaled) + 12 + 12 - 1)
    res_1d[valid_start : valid_start + len(residual_1step)] = residual_1step
    err_pred = predict_with_model(lec_model, res_1d)

    min_len = min(len(y_true_scaled), len(err_pred), len(pred_base))
    corrected = pred_base[:min_len].copy()
    correction_count = 0
    for i in range(1, min_len):
        current_pred_val = scaler.inverse_transform([[pred_base[i, 0]]])[0][0]
        prev_true_val = scaler.inverse_transform([[y_true_scaled[i - 1, 0]]])[0][0]
        gamma = abs(current_pred_val - prev_true_val)
        if gamma >= trigger_threshold:
            corrected[i, :] += err_pred[i, :]
            correction_count += 1
    return corrected, min_len, correction_count


def evaluate_setting(name, pred_scaled, y_true_scaled, scaler, lec_model=None, lec_threshold=8.0):
    min_len = min(len(pred_scaled), len(y_true_scaled))
    pred_cut = pred_scaled[:min_len]
    y_cut = y_true_scaled[:min_len]

    correction_count = 0
    if lec_model is not None:
        pred_eval, min_len_corr, correction_count = apply_lec(
            pred_cut, y_cut, scaler, lec_model, trigger_threshold=lec_threshold
        )
        y_cut = y_cut[:min_len_corr]
        min_len = min_len_corr
    else:
        pred_eval = pred_cut

    y_true_raw = scaler.inverse_transform(y_cut.reshape(-1, 1)).reshape(min_len, 12)
    y_pred_raw = scaler.inverse_transform(pred_eval.reshape(-1, 1)).reshape(min_len, 12)
    y_true_raw = np.clip(y_true_raw, 0.0, np.inf)
    y_pred_raw = np.clip(y_pred_raw, 0.0, np.inf)

    row = {
        "setting": name,
        "samples": int(min_len),
        "lec_corrections": int(correction_count),
    }
    for h in [3, 6, 12]:
        yt = y_true_raw[:, h - 1]
        yp = y_pred_raw[:, h - 1]
        row[f"MAE_{h}h"] = float(mean_absolute_error(yt, yp))
        row[f"RMSE_{h}h"] = float(np.sqrt(mean_squared_error(yt, yp)))
        row[f"R2_{h}h"] = float(r2_score(yt, yp))

    recall, peak_lag_abs, mean_lead, lead_iqr = compute_event_metrics(
        y_true_raw[:, 2], y_pred_raw[:, 2], event_threshold=75.0
    )
    row["Recall@3h(>=75)"] = recall
    row["Mean|PeakLag|@3h"] = peak_lag_abs
    row["MeanLead@3h"] = mean_lead
    row["LeadIQR@3h"] = lead_iqr
    return row


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(os.path.join(model_dir, "scaler_Dongsi.pkl"))
    config = joblib.load(os.path.join(model_dir, "config.pkl"))
    w_a = float(config["w1"])
    models_a = config["models_a"]
    models_b = config["models_b"]

    print("Loading data...")
    df = pd.read_csv(data_path)
    raw_values = df["PM2.5"].astype(float).values.reshape(-1, 1)
    scaled_data = scaler.transform(raw_values).flatten()

    total_len = len(scaled_data)
    test_start = int(total_len * 0.90)
    test_data = scaled_data[test_start:]

    print("Decomposing test segment (CEEMDAN/EEMD)...")
    imfs_a = CEEMDAN(trials=10, parallel=False)(test_data)
    imfs_b = EEMD(trials=5, parallel=False)(test_data)

    print("Loading component models...")
    all_model_names = [n for n, _ in models_a] + [n for n, _ in models_b]
    model_cache = load_model_cache(all_model_names, model_dir, device)

    print("Predicting branch outputs...")
    pred_a = predict_branch(models_a, imfs_a, model_cache)
    pred_b = predict_branch(models_b, imfs_b, model_cache)

    ds_test = SingleComponentDataset(test_data, seq_len=12, target_len=12)
    y_test_list = []
    for _, y_batch in DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0):
        y_test_list.append(y_batch.numpy())
    y_true = np.concatenate(y_test_list, axis=0).squeeze()

    min_len = min(len(y_true), len(pred_a), len(pred_b))
    y_true = y_true[:min_len]
    pred_a = pred_a[:min_len]
    pred_b = pred_b[:min_len]
    pred_full = w_a * pred_a + (1.0 - w_a) * pred_b

    lec_model = BiLSTMNet(hidden_size=64, output_size=12).to(device)
    lec_model.load_state_dict(torch.load(os.path.join(model_dir, "LEC_Model.pth"), map_location=device))
    lec_model.eval()
    lec_threshold = 8.0

    scenarios = [
        ("Full(A+B)+LEC", pred_full, lec_model),
        ("Full(A+B),NoLEC", pred_full, None),
        ("A-only+LEC", pred_a, lec_model),
        ("A-only,NoLEC", pred_a, None),
        ("B-only+LEC", pred_b, lec_model),
        ("B-only,NoLEC", pred_b, None),
    ]

    print("Evaluating ablation settings...")
    rows = []
    for setting_name, pred_scaled, setting_lec_model in scenarios:
        row = evaluate_setting(
            setting_name,
            pred_scaled,
            y_true,
            scaler,
            lec_model=setting_lec_model,
            lec_threshold=lec_threshold,
        )
        if setting_name.startswith("Full"):
            row["w_A"] = w_a
        elif setting_name.startswith("A-only"):
            row["w_A"] = 1.0
        else:
            row["w_A"] = 0.0
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df[
        [
            "setting",
            "w_A",
            "samples",
            "lec_corrections",
            "MAE_3h",
            "RMSE_3h",
            "R2_3h",
            "MAE_6h",
            "RMSE_6h",
            "R2_6h",
            "MAE_12h",
            "RMSE_12h",
            "R2_12h",
            "Recall@3h(>=75)",
            "Mean|PeakLag|@3h",
            "MeanLead@3h",
            "LeadIQR@3h",
        ]
    ]

    out_csv = os.path.join(out_dir, "lstm_minimal_ablation.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    summary_lines = [
        "BiLSTM-Hybrid Minimal Ablation (12h, Dongsi)",
        "=" * 64,
        "Notes:",
        "- Uses existing trained components in models/lstm1 (no full retraining).",
        "- PE policy follows trained setup (fixed IMF1/IMF2 removal).",
        "- LEC threshold is fixed at 8.0 for this 12h branch.",
        "",
        out_df.to_string(index=False),
        "",
    ]
    out_txt = os.path.join(out_dir, "lstm_minimal_ablation_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"[OK] Saved CSV: {out_csv}")
    print(f"[OK] Saved summary: {out_txt}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
