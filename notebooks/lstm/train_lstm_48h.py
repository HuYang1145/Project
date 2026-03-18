# train_lstm_48h.py
# 48小时(2天)预测模型：基于 CEEMDAN+BiLSTM+LEC 架构
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
import warnings
from PyEMD import CEEMDAN, EEMD
import antropy as ant

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置 (48小时 = 2天)
# ==========================================
class Config:
    CURRENT_DIR = os.getcwd()
    TARGET_STATION = 'Dongsi'
    DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, f'../../data/processed/lstm/{TARGET_STATION}_PM2.5.csv'))
    MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../models/lstm_48h'))

    SEQ_LEN = 48       # 输入：过去48小时
    TARGET_LEN = 48    # 输出：未来48小时
    INPUT_SIZE = 1

    EPOCHS = 50
    EEMD_TRIALS = 5
    BATCH_SIZE = 48
    HIDDEN_SIZE = 96
    LEARNING_RATE = 0.0005

    LEC_THRESHOLD = 12.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(Config.MODEL_DIR):
    os.makedirs(Config.MODEL_DIR)

# ==========================================
# 2. 网络定义
# ==========================================
class BiLSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, output_size=48):
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

# ==========================================
# 3. 核心函数
# ==========================================
def train_component_model(data_series, model_name):
    dataset = SingleComponentDataset(data_series, Config.SEQ_LEN, Config.TARGET_LEN)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    model = BiLSTMNet(hidden_size=Config.HIDDEN_SIZE, output_size=Config.TARGET_LEN).to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    model.train()
    print(f"    Training {model_name:<15} ...", end="")
    for epoch in range(Config.EPOCHS):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    print(" Done")
    torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, f"{model_name}.pth"))
    return model

def predict_with_saved_model(model, data_series):
    dataset = SingleComponentDataset(data_series, Config.SEQ_LEN, Config.TARGET_LEN)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch, _ in loader:
            output = model(x_batch.to(Config.DEVICE))
            preds.append(output.cpu().numpy())
    return np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([])

def predict_branch(models_list, imfs_data):
    total_samples = len(imfs_data[0]) - Config.SEQ_LEN - Config.TARGET_LEN + 1
    total_pred = np.zeros((total_samples, Config.TARGET_LEN))
    for model_name, imf_idx in models_list:
        input_data = imfs_data[imf_idx] if imf_idx < len(imfs_data) else np.zeros(len(imfs_data[0]))
        model = BiLSTMNet(hidden_size=Config.HIDDEN_SIZE, output_size=Config.TARGET_LEN).to(Config.DEVICE)
        model.load_state_dict(torch.load(os.path.join(Config.MODEL_DIR, f"{model_name}.pth")))
        total_pred += predict_with_saved_model(model, input_data)
    return total_pred

def process_branch(branch_name, imfs_data_train, drop_idx_list):
    models_list = []
    total_samples = len(imfs_data_train[0]) - Config.SEQ_LEN - Config.TARGET_LEN + 1
    preds_sum = np.zeros((total_samples, Config.TARGET_LEN))
    for i in range(len(imfs_data_train)):
        if i in drop_idx_list:
            print(f"    [Drop] {branch_name} IMF {i+1} (Noise)")
            continue
        model_name = f"{branch_name}_IMF_{i+1}"
        model = train_component_model(imfs_data_train[i], model_name)
        models_list.append((model_name, i))
        preds_sum += predict_with_saved_model(model, imfs_data_train[i])
    return models_list, preds_sum

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(">>> [Step 1] Loading Data...")
    df = pd.read_csv(Config.DATA_PATH)
    pm25_col = 'PM2.5' if 'PM2.5' in df.columns else [c for c in df.columns if c != 'time'][0]
    raw_values = df[pm25_col].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_values).flatten()
    joblib.dump(scaler, os.path.join(Config.MODEL_DIR, f'scaler_{Config.TARGET_STATION}.pkl'))

    print("\n>>> [Step 2] Signal Decomposition...")
    ceemdan = CEEMDAN(trials=Config.EEMD_TRIALS)
    imfs_a_all = ceemdan(scaled_data)
    rlmd_substitute = EEMD(trials=max(1, int(Config.EEMD_TRIALS/2)))
    imfs_b_all = rlmd_substitute(scaled_data)

    print("\n    >>> Hardcoded drop IMF1 and IMF2 (high-frequency noise)...")
    drop_idx_a = [0, 1]  # 丢弃 CEEMDAN IMF1, IMF2
    drop_idx_b = [0, 1]  # 丢弃 RLMD IMF1, IMF2

    print("\n>>> [Step 3] Splitting 70/20/10...")
    total_len = len(scaled_data)
    train_end = int(total_len * 0.70)
    lec_end = int(total_len * 0.90)

    lec_train_data = scaled_data[train_end:lec_end]
    test_data = scaled_data[lec_end:]

    imfs_train_a = [imf[:train_end] for imf in imfs_a_all]
    imfs_lec_a = [imf[train_end:lec_end] for imf in imfs_a_all]
    imfs_test_a = [imf[lec_end:] for imf in imfs_a_all]

    imfs_train_b = [imf[:train_end] for imf in imfs_b_all]
    imfs_lec_b = [imf[train_end:lec_end] for imf in imfs_b_all]
    imfs_test_b = [imf[lec_end:] for imf in imfs_b_all]

    print("\n>>> [Step 4] Training Base Models...")
    models_a_list, _ = process_branch("CEEMDAN", imfs_train_a, drop_idx_a)
    models_b_list, _ = process_branch("RLMD_Sub", imfs_train_b, drop_idx_b)

    print("\n>>> [Step 5] Calculating Weights...")
    preds_lec_a = predict_branch(models_a_list, imfs_lec_a)
    preds_lec_b = predict_branch(models_b_list, imfs_lec_b)

    temp_ds_lec = SingleComponentDataset(lec_train_data, Config.SEQ_LEN, Config.TARGET_LEN)
    y_lec_true = []
    for _, y in DataLoader(temp_ds_lec, batch_size=Config.BATCH_SIZE, shuffle=False):
        y_lec_true.append(y.numpy())
    y_lec_true = np.concatenate(y_lec_true, axis=0).squeeze()

    best_w = 0.5
    min_mse = float('inf')
    for w in np.linspace(0, 1, 20):
        combined = w * preds_lec_a + (1-w) * preds_lec_b
        mse = np.mean((y_lec_true - combined)**2)
        if mse < min_mse:
            min_mse = mse
            best_w = w

    print(f"    Optimal Weight: w_A = {best_w:.2f}")
    config_dict = {'w1': best_w, 'models_a': models_a_list, 'models_b': models_b_list}
    joblib.dump(config_dict, os.path.join(Config.MODEL_DIR, 'config.pkl'))

    print("\n>>> [Step 6] Training LEC Model...")
    pred_hybrid_lec = best_w * preds_lec_a + (1-best_w) * preds_lec_b
    residuals_lec = y_lec_true - pred_hybrid_lec

    res_1d = np.zeros(len(lec_train_data))
    valid_start = Config.SEQ_LEN
    valid_len = len(residuals_lec)
    res_1d[valid_start : valid_start+valid_len] = residuals_lec[:, 0]

    lec_model = train_component_model(res_1d, "LEC_Model")

    print(f"\n>>> [Step 7] Testing...")
    preds_test_a = predict_branch(config_dict['models_a'], imfs_test_a)
    preds_test_b = predict_branch(config_dict['models_b'], imfs_test_b)
    final_pred_test = best_w * preds_test_a + (1-best_w) * preds_test_b

    temp_ds_test = SingleComponentDataset(test_data, Config.SEQ_LEN, Config.TARGET_LEN)
    y_test_list = []
    for _, y in DataLoader(temp_ds_test, batch_size=Config.BATCH_SIZE, shuffle=False):
        y_test_list.append(y.numpy())
    y_test_true = np.concatenate(y_test_list, axis=0).squeeze()

    residuals_test_1step = y_test_true[:, 0] - final_pred_test[:, 0]
    res_test_1d = np.zeros(len(test_data))
    res_test_1d[valid_start : valid_start + len(residuals_test_1step)] = residuals_test_1step
    err_pred_test = predict_with_saved_model(lec_model, res_test_1d)

    final_pred_corrected = final_pred_test.copy()
    min_len = min(len(y_test_true), len(err_pred_test))

    correction_count = 0
    for i in range(1, min_len):
        current_pred_val = scaler.inverse_transform([[final_pred_test[i, 0]]])[0][0]
        prev_known_true = scaler.inverse_transform([[y_test_true[i-1, 0]]])[0][0]
        gamma = abs(current_pred_val - prev_known_true)

        if gamma >= Config.LEC_THRESHOLD:
            final_pred_corrected[i, :] += err_pred_test[i, :]
            correction_count += 1

    print(f"    LEC corrections: {correction_count}")

    target_step = 47
    pred_raw = scaler.inverse_transform(final_pred_test[:min_len, target_step].reshape(-1, 1))
    pred_corrected = scaler.inverse_transform(final_pred_corrected[:min_len, target_step].reshape(-1, 1))
    true_vals = scaler.inverse_transform(y_test_true[:min_len, target_step].reshape(-1, 1))

    rmse_raw = np.sqrt(mean_squared_error(true_vals, pred_raw))
    rmse_corrected = np.sqrt(mean_squared_error(true_vals, pred_corrected))

    print(f"\nFinal Results (48h forecast):")
    print(f"Base Model RMSE: {rmse_raw:.4f}")
    print(f"With LEC RMSE: {rmse_corrected:.4f}")

if __name__ == "__main__":
    main()

