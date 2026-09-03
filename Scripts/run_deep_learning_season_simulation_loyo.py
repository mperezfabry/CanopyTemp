import os
import gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(min(8, os.cpu_count() or 4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(f"PYTORCH DEVICE: {device} | CUDA AVAILABLE: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
else:
    print("NOTICE: Running on CPU using multi-threaded PyTorch optimization.")
print("=" * 80)

class RawSequenceCNN1D(nn.Module):
    def __init__(self, num_channels=8, filters=32, kernel_size=5):
        super(RawSequenceCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filters, filters * 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters * 2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

class RawSequenceLSTM(nn.Module):
    def __init__(self, num_channels=8, hidden_dim=64, num_layers=2):
        super(RawSequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(num_channels, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.fc(out).squeeze(-1)

class RawSequenceMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super(RawSequenceMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x).squeeze(-1)

def interpolate_stages(df):
    if 'growth_stage' in df.columns and 'doy' in df.columns:
        df_stages = df.dropna(subset=['growth_stage'])[['year', 'doy', 'growth_stage']].copy()
        if not df_stages.empty:
            stage_order = {'V3':1, 'V4':2, 'V5':3, 'V6':4, 'V7':5, 'V8':6, 'V9':7, 'V10':8, 'V11':9, 'V12':10,
                           'V13':11, 'V14':12, 'V15':13, 'V16':14, 'V17':15, 'V18':16, 'V19':17, 'V20':18,
                           'VT':19, 'R1':20, 'R2':21, 'R3':22, 'R4':23, 'R5':24, 'R6':25}
            df_stages['stage_num'] = df_stages['growth_stage'].map(stage_order)
            stage_counts = df_stages.groupby(['year', 'doy', 'stage_num']).size().reset_index(name='count')
            modes = stage_counts.sort_values(['year', 'doy', 'count'], ascending=[True, True, False]).drop_duplicates(subset=['year', 'doy'])
            modes = modes.rename(columns={'stage_num': 'stage_num_interp'})[['year', 'doy', 'stage_num_interp']]
            return modes

    doys = df[['year', 'doy']].drop_duplicates().copy()
    def map_doy(d):
        if d <= 175: return 4
        elif d <= 190: return 8
        elif d <= 210: return 10
        elif d <= 230: return 20
        elif d <= 250: return 22
        else: return 25
    doys['stage_num_interp'] = doys['doy'].apply(map_doy)
    return doys

def prepare_3d_raw_sequence_dataset(parquet_path):
    print(f"Loading 3D Raw Sequence dataset from: {parquet_path}")
    if not os.path.exists(parquet_path):
        parquet_path = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.parquet'

    df = pd.read_parquet(parquet_path)
    if 'crop' in df.columns:
        df = df[df['crop'] != 'Sunflower'].copy()

    if 'doy' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['doy'] = df['timestamp'].dt.dayofyear
        df['hour'] = df['timestamp'].dt.hour
        df['year'] = df['timestamp'].dt.year

    df_stages_interp = interpolate_stages(df)
    df = pd.merge(df, df_stages_interp, on=['year', 'doy'], how='inner')
    df['stage_num'] = df['stage_num_interp']

    # Weather standardization
    if 'air_temp_c' in df.columns and 'weather_air_temp_c' not in df.columns:
        df['weather_air_temp_c'] = df['air_temp_c']
    if 'precip_mm' in df.columns and 'weather_precip_mm' not in df.columns:
        df['weather_precip_mm'] = df['precip_mm']

    if 'canopy_temp' in df.columns:
        df['canopy_temp'] = df['canopy_temp'].where((df['canopy_temp'] >= 5.0) & (df['canopy_temp'] <= 55.0), np.nan)
    elif 'Solar_noon_Avg' in df.columns:
        df['canopy_temp'] = df['Solar_noon_Avg']
    else:
        df['canopy_temp'] = 25.0

    if 'CWSI' not in df.columns:
        if 'canopy_temp' in df.columns and 'weather_air_temp_c' in df.columns:
            diff = df['canopy_temp'] - df['weather_air_temp_c']
            d_min, d_max = diff.min(), diff.max()
            if pd.notna(d_min) and pd.notna(d_max) and d_max > d_min:
                df['CWSI'] = ((diff - d_min) / (d_max - d_min)).clip(0.0, 1.0)
            else:
                df['CWSI'] = 0.0
        else:
            df['CWSI'] = 0.0

    swc_cols = [c for c in ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120'] if c in df.columns]
    for col in swc_cols:
        df[col] = df.groupby(['year', 'plot'])[col].ffill().bfill() if col in df.columns else 0.20
    
    if swc_cols:
        df['swc_rz'] = df[swc_cols].mean(axis=1)
    else:
        df['swc_rz'] = 0.20
        for c in ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120']:
            df[c] = 0.20
        swc_cols = ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120']

    for col, default_val in [('irrigation_depth_mm', 0.0), ('weather_precip_mm', 0.0), ('CWSI', 0.0), ('weather_air_temp_c', 20.0)]:
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    df['water_inflow'] = df['weather_precip_mm'].fillna(0) + df['irrigation_depth_mm'].fillna(0)

    df = df[df['grain_yield'].notna()].copy()
    max_y = df.groupby('year')['grain_yield'].transform('max')
    df['relative_yield_loss'] = (max_y - df['grain_yield']) / max_y

    channels = ['canopy_temp', 'CWSI'] + swc_cols + ['swc_rz', 'water_inflow']
    for c in channels:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    return df, channels

def run_fast_loyo_deep_learning(df, channels, target_col='grain_yield'):
    print(f"\n" + "=" * 80)
    print(f"STARTING FAST LOYO DEEP LEARNING SIMULATION FOR TARGET: [{target_col}]")
    print("=" * 80)

    summary_csv_path = f'/mnt/Data/LIRF/Scripts/deep_learning_simulation_results_{target_col}_loyo.csv'

    stages_to_sim = [8, 10, 20, 22, 25] # V10, V12, R1, R3, R6
    stage_names = {8: 'V10', 10: 'V12', 20: 'R1', 22: 'R3', 25: 'R6'}

    results = []

    for stg_num in stages_to_sim:
        stg_name = stage_names[stg_num]
        df_sub = df[df['stage_num'] <= stg_num].copy()
        
        # Daily aggregation per plot to prevent CUDA OOM on 30,000+ length sub-hourly sequences
        df_daily = df_sub.groupby(['year', 'plot', 'doy']).agg(
            {c: 'mean' if c != 'water_inflow' else 'sum' for c in channels}
        ).reset_index()
        
        target_map = df_sub.groupby(['year', 'plot'])[target_col].first().to_dict()
        
        plot_seqs = []
        plot_meta = []

        for (yr, p), grp in df_daily.groupby(['year', 'plot']):
            if (yr, p) in target_map and pd.notna(target_map[(yr, p)]):
                target_val = target_map[(yr, p)]
                seq_vals = grp[channels].values
                if len(seq_vals) > 0:
                    plot_seqs.append(seq_vals)
                    plot_meta.append({'year': yr, 'plot': p, 'target': target_val})

        if not plot_seqs:
            continue

        max_len = max(len(s) for s in plot_seqs)
        n_plots = len(plot_seqs)
        n_channels = len(channels)

        X_3d = np.zeros((n_plots, max_len, n_channels), dtype=np.float32)
        y_arr = np.array([m['target'] for m in plot_meta], dtype=np.float32)
        years_arr = np.array([m['year'] for m in plot_meta])

        for idx, seq in enumerate(plot_seqs):
            X_3d[idx, :len(seq), :] = seq

        logo = LeaveOneGroupOut()
        models_dict = {
            '1D-CNN': lambda: RawSequenceCNN1D(num_channels=n_channels),
            'LSTM': lambda: RawSequenceLSTM(num_channels=n_channels),
            'MLP': lambda: RawSequenceMLP(input_dim=n_channels)
        }

        for m_name, model_fn in models_dict.items():
            all_preds = []
            all_actuals = []

            for train_idx, test_idx in logo.split(X_3d, y_arr, groups=years_arr):
                X_tr, y_tr = X_3d[train_idx], y_arr[train_idx]
                X_te, y_te = X_3d[test_idx], y_arr[test_idx]

                scaler = StandardScaler()
                X_tr_flat = X_tr.reshape(-1, n_channels)
                X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(X_tr.shape)
                X_te_scaled = scaler.transform(X_te.reshape(-1, n_channels)).reshape(X_te.shape)

                model = model_fn().to(device)
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.MSELoss()

                ds_tr = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
                loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

                model.train()
                for epoch in range(15):
                    for bx, by in loader_tr:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    bx_te = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
                    p = model(bx_te).cpu().numpy()

                all_preds.extend(p)
                all_actuals.extend(y_te)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            r2 = r2_score(all_actuals, all_preds)
            rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))

            results.append({
                'stage': stg_name,
                'model': m_name,
                'R2': round(r2, 4),
                'RMSE': round(rmse, 2)
            })
            print(f"  [{stg_name}] {m_name:10s}: R² = {r2:.4f} | RMSE = {rmse:.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(summary_csv_path, index=False)
    print(f"Saved Deep Learning LOYO Results to: {summary_csv_path}")

def main():
    p_ml = '/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet'
    df, channels = prepare_3d_raw_sequence_dataset(p_ml)
    run_fast_loyo_deep_learning(df, channels, target_col='grain_yield')

if __name__ == '__main__':
    main()
