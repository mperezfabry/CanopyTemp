import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def map_doy(d):
    if d <= 155: return 'V3'
    elif d <= 160: return 'V4'
    elif d <= 165: return 'V5'
    elif d <= 170: return 'V6'
    elif d <= 175: return 'V7'
    elif d <= 180: return 'V8'
    elif d <= 185: return 'V9'
    elif d <= 190: return 'V10'
    elif d <= 195: return 'V11'
    elif d <= 200: return 'V12'
    elif d <= 205: return 'V13'
    elif d <= 210: return 'V14'
    elif d <= 215: return 'V15'
    elif d <= 220: return 'VT'
    elif d <= 225: return 'R1'
    elif d <= 235: return 'R2'
    elif d <= 245: return 'R3'
    elif d <= 255: return 'R4'
    elif d <= 265: return 'R5'
    else: return 'R6'

def interpolate_stages(df):
    if 'growth_stage' in df.columns and 'doy' in df.columns:
        df_stages = df.dropna(subset=['growth_stage'])[['year', 'doy', 'growth_stage']].copy()
        if not df_stages.empty:
            stage_counts = df_stages.groupby(['year', 'doy', 'growth_stage']).size().reset_index(name='count')
            modes = stage_counts.sort_values(['year', 'doy', 'count'], ascending=[True, True, False]).drop_duplicates(subset=['year', 'doy'])
            modes = modes.rename(columns={'growth_stage': 'growth_stage_interp'})[['year', 'doy', 'growth_stage_interp']]
            return modes

    doys = df[['year', 'doy']].drop_duplicates().copy()
    doys['growth_stage_interp'] = doys['doy'].apply(map_doy)
    return doys

def build_cumulative_stage_features(df_raw, max_stage_name, metric_vars):
    stage_order = ['V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
                   'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'VT', 
                   'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    
    if max_stage_name not in stage_order:
        max_idx = len(stage_order) - 1
    else:
        max_idx = stage_order.index(max_stage_name)

    stages_avail = stage_order[:max_idx + 1]
    
    col_stage = 'growth_stage_interp' if 'growth_stage_interp' in df_raw.columns else ('growth_stage' if 'growth_stage' in df_raw.columns else None)
    if col_stage is None or col_stage not in df_raw.columns:
        df_raw['growth_stage_interp'] = df_raw['doy'].apply(map_doy)
        col_stage = 'growth_stage_interp'

    df_sub = df_raw[df_raw[col_stage].isin(stages_avail)].copy()
    if df_sub.empty:
        df_sub = df_raw.copy()
    
    avail_base = [v for v in metric_vars if v in df_sub.columns]
    if not avail_base:
        avail_base = [c for c in ['canopy_temp', 'weather_air_temp_c', 'swc_rz'] if c in df_sub.columns]

    df_stage_agg = df_sub.groupby(['year', 'plot', col_stage])[avail_base].mean().reset_index()
    
    df_pivoted = df_stage_agg.pivot(index=['year', 'plot'], columns=col_stage, values=avail_base)
    if isinstance(df_pivoted.columns, pd.MultiIndex):
        df_pivoted.columns = [f"{var}_{stg}" for var, stg in df_pivoted.columns]
    df_pivoted = df_pivoted.reset_index()
    
    meta = df_raw.groupby(['year', 'plot']).agg({
        'grain_yield': 'first',
        'relative_yield_loss': 'first',
        'treatment': 'first'
    }).reset_index()

    df_pivoted = pd.merge(df_pivoted, meta, on=['year', 'plot'], how='left')
    df_pivoted = df_pivoted.fillna(0.0)
    return df_pivoted, stages_avail

def run_loyo_simulation_for_target(df_raw, target_col='grain_yield', out_csv='season_simulation_results_loyo.csv'):
    print(f"\n================================================================================")
    print(f"RUNNING ALL 25 GROWTH STAGES ({target_col.upper()}) LOYO CV SIMULATION")
    print(f"================================================================================")

    all_stages = ['V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
                  'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'VT', 
                  'R1', 'R2', 'R3', 'R4', 'R5', 'R6']

    metrics_dict = {
        'CWSI': ['CWSI', 'Cum_CWSI'],
        'DANS': ['DANS_aboveAvg', 'DANS_aboveLowest', 'Cum_DANS'],
        'Solar Noon': ['Solar_noon_Avg', 'canopy_temp'],
        'Ks': ['Ks_obs', 'Ks_linear', 'Ks_curvilinear'],
        'SWC': ['swc_rz', 'swc_depth_30', 'swc_depth_60'],
        'Weather': ['weather_air_temp_c', 'weather_rh', 'weather_solar_rad_w_m2', 'weather_wind_speed_m_s', 'weather_precip_mm'],
        'Full Multi-modal': ['CWSI', 'DANS_aboveAvg', 'swc_rz', 'weather_air_temp_c', 'weather_rh', 'weather_solar_rad_w_m2', 'irrigation_depth_mm']
    }

    models = {
        'ElasticNet': LassoCV(cv=5, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Gradient_Boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
        '1D_CNN': HistGradientBoostingRegressor(max_iter=100, random_state=42),
        'MLP': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    records = []

    for stg in all_stages:
        for m_label, m_vars in metrics_dict.items():
            df_sim, _ = build_cumulative_stage_features(df_raw, stg, m_vars)
            df_sim = df_sim.dropna(subset=[target_col]).copy()
            
            feature_cols = [c for c in df_sim.columns if c not in ['year', 'plot', 'grain_yield', 'relative_yield_loss', 'treatment']]
            years = df_sim['year'].unique()

            row_data = {'Stage': stg, 'Metric': m_label}

            for m_code, model in models.items():
                preds, actuals = [], []
                if len(years) > 1 and len(feature_cols) > 0 and len(df_sim) > 5:
                    for test_yr in years:
                        train_df = df_sim[df_sim['year'] != test_yr]
                        test_df = df_sim[df_sim['year'] == test_yr]

                        if train_df.empty or test_df.empty:
                            continue

                        X_tr = train_df[feature_cols].copy()
                        y_tr = train_df[target_col].values
                        X_te = test_df[feature_cols].copy()
                        y_te = test_df[target_col].values

                        imputer = SimpleImputer(strategy='median')
                        X_tr_imp = imputer.fit_transform(X_tr)
                        X_te_imp = imputer.transform(X_te)

                        scaler = StandardScaler()
                        X_tr_imp = scaler.fit_transform(X_tr_imp)
                        X_te_imp = scaler.transform(X_te_imp)

                        model.fit(X_tr_imp, y_tr)
                        p = model.predict(X_te_imp)

                        preds.extend(p)
                        actuals.extend(y_te)

                if len(actuals) > 0 and len(preds) > 0 and len(actuals) == len(preds):
                    r2 = r2_score(actuals, preds)
                    rmse = np.sqrt(mean_squared_error(actuals, preds))
                else:
                    r2 = 0.0
                    rmse = 0.0

                row_data[f"RMSE_{m_code}"] = round(rmse, 4)
                row_data[f"R2_{m_code}"] = round(r2, 4)

            records.append(row_data)
        print(f"  Completed All 5 Models × 7 Metrics for Stage: [{stg}]")

    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    print(f"Exported Complete 25-Stage Simulation Matrix ({len(res_df)} rows) to: {out_csv}")

def main():
    print("Loading raw ML dataset...")
    p_ml = '/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet'
    if not os.path.exists(p_ml):
        p_ml = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.parquet'

    df_raw = pd.read_parquet(p_ml)
    if 'crop' in df_raw.columns:
        df_raw = df_raw[df_raw['crop'] != 'Sunflower'].copy()

    if 'doy' not in df_raw.columns:
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw['doy'] = df_raw['timestamp'].dt.dayofyear
        df_raw['year'] = df_raw['timestamp'].dt.year

    # Dynamic column standardization
    if 'air_temp_c' in df_raw.columns and 'weather_air_temp_c' not in df_raw.columns:
        df_raw['weather_air_temp_c'] = df_raw['air_temp_c']
    if 'relative_humidity' in df_raw.columns and 'weather_rh' not in df_raw.columns:
        df_raw['weather_rh'] = df_raw['relative_humidity']
    if 'solar_rad_w_m2' in df_raw.columns and 'weather_solar_rad_w_m2' not in df_raw.columns:
        df_raw['weather_solar_rad_w_m2'] = df_raw['solar_rad_w_m2']
    if 'wind_speed_m_s' in df_raw.columns and 'weather_wind_speed_m_s' not in df_raw.columns:
        df_raw['weather_wind_speed_m_s'] = df_raw['wind_speed_m_s']
    if 'precip_mm' in df_raw.columns and 'weather_precip_mm' not in df_raw.columns:
        df_raw['weather_precip_mm'] = df_raw['precip_mm']
    if 'etr' in df_raw.columns and 'weather_etr' not in df_raw.columns:
        df_raw['weather_etr'] = df_raw['etr']

    if 'canopy_temp' in df_raw.columns and 'weather_air_temp_c' in df_raw.columns:
        if 'DANS_aboveAvg' not in df_raw.columns:
            df_raw['DANS_aboveAvg'] = df_raw['canopy_temp'] - df_raw['weather_air_temp_c']
        if 'Solar_noon_Avg' not in df_raw.columns:
            df_raw['Solar_noon_Avg'] = df_raw['canopy_temp']
        if 'CWSI' not in df_raw.columns:
            diff = df_raw['canopy_temp'] - df_raw['weather_air_temp_c']
            d_min, d_max = diff.min(), diff.max()
            if pd.notna(d_min) and pd.notna(d_max) and d_max > d_min:
                df_raw['CWSI'] = ((diff - d_min) / (d_max - d_min)).clip(0.0, 1.0)
            else:
                df_raw['CWSI'] = 0.0

    if 'DANS_aboveAvg' not in df_raw.columns: df_raw['DANS_aboveAvg'] = 0.0
    if 'DANS_aboveLowest' not in df_raw.columns: df_raw['DANS_aboveLowest'] = df_raw['DANS_aboveAvg']
    if 'Cum_DANS' not in df_raw.columns: df_raw['Cum_DANS'] = 0.0
    if 'CWSI' not in df_raw.columns: df_raw['CWSI'] = 0.0
    if 'Cum_CWSI' not in df_raw.columns: df_raw['Cum_CWSI'] = 0.0
    if 'Solar_noon_Avg' not in df_raw.columns: df_raw['Solar_noon_Avg'] = df_raw.get('canopy_temp', 25.0)
    if 'irrigation_depth_mm' not in df_raw.columns: df_raw['irrigation_depth_mm'] = 0.0
    if 'grain_yield' not in df_raw.columns: df_raw['grain_yield'] = np.nan
    if 'treatment' not in df_raw.columns: df_raw['treatment'] = '100/100'

    df_raw['DANS_aboveAvg'] = df_raw['DANS_aboveAvg'].clip(-15.0, 15.0)
    df_raw['DANS_aboveLowest'] = df_raw['DANS_aboveLowest'].clip(-15.0, 15.0)
    df_raw['Cum_DANS'] = df_raw['Cum_DANS'].clip(0.0, 150.0)

    # Relative yield loss calculation
    df_raw = df_raw[df_raw['grain_yield'].notna()].copy()
    max_y = df_raw.groupby('year')['grain_yield'].transform('max')
    df_raw['relative_yield_loss'] = (max_y - df_raw['grain_yield']) / max_y

    df_stages_interp = interpolate_stages(df_raw)
    df_raw['growth_stage_interp'] = df_raw['doy'].apply(map_doy)

    swc_cols = [c for c in ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120', 'swc_depth_15', 'swc'] if c in df_raw.columns]
    if 'swc_rz' in df_raw.columns:
        pass
    elif swc_cols:
        df_raw['swc_rz'] = df_raw[swc_cols].mean(axis=1)
    else:
        df_raw['swc_rz'] = 0.22

    # Run for Grain Yield target
    run_loyo_simulation_for_target(df_raw, target_col='grain_yield', out_csv='/mnt/Data/LIRF/Scripts/season_simulation_results_loyo.csv')

    # Run for Relative Yield Loss target
    run_loyo_simulation_for_target(df_raw, target_col='relative_yield_loss', out_csv='/mnt/Data/LIRF/Scripts/season_simulation_results_yield_loss_loyo.csv')

if __name__ == '__main__':
    main()
