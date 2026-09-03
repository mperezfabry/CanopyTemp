import pandas as pd
import numpy as np
import os

def map_stage(stage):
    if not isinstance(stage, str):
        return None
    if stage in ['V3', 'V4', 'V5', 'V6', 'V7']:
        return 'Early_Veg'
    elif stage in ['V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20']:
        return 'Late_Veg'
    elif stage in ['VT', 'R1']:
        return 'Flowering'
    elif stage in ['R2', 'R3', 'R4', 'R5', 'R6']:
        return 'GrainFill'
    return None

def range_diff(x):
    return x.max() - x.min() if len(x) > 0 else np.nan

def main():
    p_ml = '/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet'
    if not os.path.exists(p_ml):
        p_ml = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.parquet'

    print(f"Loading raw ML dataset from: {p_ml}")
    df_raw = pd.read_parquet(p_ml)
    
    # Preserve 2017 Corn Plots (Plots A6..B8 are Corn, not Sunflower)
    if 'crop' in df_raw.columns:
        df_raw.loc[(df_raw['year'] == 2017) & (df_raw['plot'].isin(['A6','A7','A8','A9','B1','B2','B3','B4','B5','B6','B7','B8'])), 'crop'] = 'Corn'
        df_raw = df_raw[df_raw['crop'] != 'Sunflower'].copy()
    
    print("Vectorized filling of missing growth stage using mode of that year and doy...")
    if 'growth_stage' in df_raw.columns and 'doy' in df_raw.columns:
        df_stages = df_raw.dropna(subset=['growth_stage'])[['year', 'doy', 'growth_stage']].copy()
        if not df_stages.empty:
            stage_counts = df_stages.groupby(['year', 'doy', 'growth_stage']).size().reset_index(name='count')
            modes = stage_counts.sort_values(['year', 'doy', 'count'], ascending=[True, True, False]).drop_duplicates(subset=['year', 'doy'])
            modes = modes.rename(columns={'growth_stage': 'growth_stage_mode'})[['year', 'doy', 'growth_stage_mode']]
            df_raw = pd.merge(df_raw, modes, on=['year', 'doy'], how='left')
            df_raw['growth_stage'] = df_raw['growth_stage'].fillna(df_raw['growth_stage_mode'])
            df_raw.drop(columns=['growth_stage_mode'], inplace=True)

        df_raw['phase'] = df_raw['growth_stage'].apply(map_stage)
    else:
        df_raw['phase'] = np.nan

    # Vectorial fallback for years/days with no growth stage data (like 2014)
    if 'doy' not in df_raw.columns:
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw['doy'] = df_raw['timestamp'].dt.dayofyear
        df_raw['hour'] = df_raw['timestamp'].dt.hour
        df_raw['year'] = df_raw['timestamp'].dt.year

    is_null_phase = df_raw['phase'].isna()
    doy = df_raw['doy']
    df_raw.loc[is_null_phase & (doy <= 182), 'phase'] = 'Early_Veg'
    df_raw.loc[is_null_phase & (doy > 182) & (doy <= 210), 'phase'] = 'Late_Veg'
    df_raw.loc[is_null_phase & (doy > 210) & (doy <= 222), 'phase'] = 'Flowering'
    df_raw.loc[is_null_phase & (doy > 222), 'phase'] = 'GrainFill'
    
    # Check SWC columns
    swc_cols = [c for c in ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120'] if c in df_raw.columns]
    if not swc_cols:
        swc_cols = [c for c in df_raw.columns if 'swc' in c.lower() or 'vwc' in c.lower()]
        
    if swc_cols:
        df_raw['swc_rz'] = df_raw[swc_cols].mean(axis=1)
    else:
        df_raw['swc_rz'] = np.nan
    
    # Calculate Ks_obs directly from observed root-zone SWC
    p, fc, wp = 0.55, 0.33, 0.13
    denom = (1.0 - p) * (fc - wp)
    df_raw['Ks_obs'] = ((df_raw['swc_rz'] - wp) / denom).clip(0.0, 1.0)

    # Standardize column naming for weather
    rename_w = {
        'air_temp_c': 'weather_air_temp_c',
        'relative_humidity': 'weather_rh',
        'rh': 'weather_rh',
        'solar_rad_w_m2': 'weather_solar_rad_w_m2',
        'wind_speed_m_s': 'weather_wind_speed_m_s',
        'precip_mm': 'weather_precip_mm',
        'etr': 'weather_etr',
        'eto': 'weather_eto'
    }
    for old_c, new_c in rename_w.items():
        if old_c in df_raw.columns and new_c not in df_raw.columns:
            df_raw[new_c] = df_raw[old_c]

    # Compute CWSI and DANS if canopy_temp is present
    if 'canopy_temp' in df_raw.columns and 'weather_air_temp_c' in df_raw.columns:
        df_raw['DANS_aboveAvg'] = df_raw['canopy_temp'] - df_raw['weather_air_temp_c']
        df_raw['DANS_aboveLowest'] = df_raw['DANS_aboveAvg']
        diff = df_raw['canopy_temp'] - df_raw['weather_air_temp_c']
        d_min, d_max = diff.min(), diff.max()
        if pd.notna(d_min) and pd.notna(d_max) and d_max > d_min:
            df_raw['CWSI'] = ((diff - d_min) / (d_max - d_min)).clip(0.0, 1.0)
        else:
            df_raw['CWSI'] = np.nan
        df_raw['Solar_noon_Avg'] = df_raw['canopy_temp']

    # Ensure required columns exist with NaN fallbacks
    req_defaults = {
        'weather_air_temp_c': np.nan, 'weather_rh': np.nan, 'weather_solar_rad_w_m2': np.nan,
        'weather_wind_speed_m_s': np.nan, 'weather_precip_mm': 0.0, 'weather_etr': 0.0,
        'irrigation_depth_mm': 0.0, 'DANS_aboveAvg': np.nan, 'DANS_aboveLowest': np.nan,
        'Ks_linear': 1.0, 'Ks_curvilinear': 1.0, 'Cum_DANS': 0.0, 'Cum_Ks': 0.0,
        'grain_yield': np.nan, 'treatment': '100/100', 'CWSI': np.nan, 'Solar_noon_Avg': np.nan, 'Cum_CWSI': 0.0
    }
    for col, default_val in req_defaults.items():
        if col not in df_raw.columns:
            df_raw[col] = default_val

    print("Aggregating to daily plot-level averages/sums...")
    df_24h = df_raw.groupby(['year', 'doy', 'plot', 'phase']).agg({
        'swc_rz': 'mean',
        'Ks_obs': 'mean',
        'weather_air_temp_c': 'mean',
        'weather_rh': 'mean',
        'weather_solar_rad_w_m2': 'mean',
        'weather_wind_speed_m_s': 'mean',
        'weather_precip_mm': 'sum',
        'weather_etr': 'sum',
        'irrigation_depth_mm': 'max',
        'DANS_aboveAvg': 'mean',
        'DANS_aboveLowest': 'mean',
        'Ks_linear': 'mean',
        'Ks_curvilinear': 'mean',
        'Cum_DANS': 'max',
        'Cum_Ks': 'max',
        'grain_yield': 'first',
        'treatment': 'first'
    }).reset_index()
    
    # Daylight hours only (9:00 AM to 5:00 PM) for solar noon temperature and CWSI
    df_daylight = df_raw[(df_raw['hour'] >= 9) & (df_raw['hour'] <= 17)].copy()
    if 'weather_solar_rad_w_m2' in df_daylight.columns and df_daylight['weather_solar_rad_w_m2'].notna().any():
        df_daylight = df_daylight[df_daylight['weather_solar_rad_w_m2'] > 100]
        
    df_day = df_daylight.groupby(['year', 'doy', 'plot']).agg({
        'CWSI': 'mean',
        'Solar_noon_Avg': 'mean',
        'Cum_CWSI': 'max'
    }).reset_index()
    
    # Merge daily tables
    daily_df = pd.merge(df_24h, df_day, on=['year', 'doy', 'plot'], how='left')
    
    # Recalculate Cum_Ks sequentially using native daily Ks_linear
    daily_df = daily_df.sort_values(['year', 'plot', 'doy']).reset_index(drop=True)
    daily_df['Cum_Ks'] = daily_df.groupby(['year', 'plot'])['Ks_linear'].transform(lambda x: x.fillna(1.0).cumsum())
        
    print("Aggregating to growth-phase level...")
    phase_agg = daily_df.groupby(['year', 'plot', 'phase']).agg({
        'swc_rz': 'mean',
        'weather_air_temp_c': 'mean',
        'weather_rh': 'mean',
        'weather_solar_rad_w_m2': 'mean',
        'weather_wind_speed_m_s': 'mean',
        'weather_precip_mm': 'sum',
        'weather_etr': 'sum',
        'irrigation_depth_mm': 'sum',
        'Solar_noon_Avg': 'mean',
        'CWSI': ['mean', 'max', 'std'],
        'DANS_aboveAvg': 'mean',
        'DANS_aboveLowest': 'mean',
        'Ks_obs': 'mean',
        'Ks_linear': 'mean',
        'Ks_curvilinear': 'mean',
        'Cum_CWSI': range_diff,
        'Cum_DANS': range_diff,
        'Cum_Ks': range_diff,
        'grain_yield': 'first',
        'treatment': 'first'
    }).reset_index()
    
    phase_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in phase_agg.columns]
    phase_agg.rename(columns={
        'Cum_CWSI_range_diff': 'Cum_CWSI_diff',
        'Cum_DANS_range_diff': 'Cum_DANS_diff',
        'Cum_Ks_range_diff': 'Cum_Ks_diff'
    }, inplace=True)
    
    pivot_cols = [
        'swc_rz_mean', 'weather_air_temp_c_mean', 'weather_rh_mean',
        'weather_solar_rad_w_m2_mean', 'weather_wind_speed_m_s_mean',
        'weather_precip_mm_sum', 'weather_etr_sum', 'irrigation_depth_mm_sum',
        'Solar_noon_Avg_mean', 'CWSI_mean', 'CWSI_max', 'CWSI_std',
        'DANS_aboveAvg_mean', 'DANS_aboveLowest_mean', 'Ks_obs_mean', 'Ks_linear_mean',
        'Ks_curvilinear_mean', 'Cum_CWSI_diff', 'Cum_DANS_diff', 'Cum_Ks_diff'
    ]
    
    print("Pivoting growth stages to wide format...")
    pivoted = phase_agg.pivot(index=['year', 'plot', 'grain_yield_first', 'treatment_first'], columns='phase', values=pivot_cols)
    pivoted.columns = [f"{col}_{phase}" for col, phase in pivoted.columns]
    pivoted = pivoted.reset_index()
    pivoted.rename(columns={'grain_yield_first': 'grain_yield', 'treatment_first': 'treatment'}, inplace=True)
    
    df_plot_level = pivoted.dropna(subset=['grain_yield']).copy()
    
    print("Calculating realized vegetative and reproductive irrigation fractions dynamically...")
    df_irrig_phase = daily_df.groupby(['year', 'plot', 'phase'])['irrigation_depth_mm'].sum().unstack(fill_value=0.0)
    veg_phases = ['Early_Veg', 'Late_Veg']
    rep_phases = ['Flowering', 'GrainFill']
    df_irrig_phase['veg_irrig'] = sum(df_irrig_phase[p] for p in veg_phases if p in df_irrig_phase.columns)
    df_irrig_phase['rep_irrig'] = sum(df_irrig_phase[p] for p in rep_phases if p in df_irrig_phase.columns)
    
    max_veg = df_irrig_phase.groupby('year')['veg_irrig'].transform('max')
    max_rep = df_irrig_phase.groupby('year')['rep_irrig'].transform('max')
    
    df_irrig_phase['veg'] = (df_irrig_phase['veg_irrig'] / max_veg * 100.0).round(1).fillna(100.0)
    df_irrig_phase['rep'] = (df_irrig_phase['rep_irrig'] / max_rep * 100.0).round(1).fillna(100.0)
    df_irrig_final = df_irrig_phase[['veg', 'rep']].reset_index()
    
    df_plot_level = pd.merge(df_plot_level, df_irrig_final, on=['year', 'plot'], how='left')
    
    max_yields = df_plot_level.groupby('year')['grain_yield'].transform('max')
    df_plot_level['relative_yield_reduction'] = (max_yields - df_plot_level['grain_yield']) / max_yields
    
    group_cols = ['year', 'treatment', 'veg', 'rep']
    numeric_cols = [c for c in df_plot_level.columns if c not in group_cols and c != 'plot']
    df_averaged = df_plot_level.groupby(group_cols)[numeric_cols].mean().reset_index()
    
    # Impute missing values safely using pandas median (zero fallback for 100% NaN columns)
    avg_num = df_averaged.select_dtypes(include=[np.number])
    avg_num_imp = avg_num.fillna(avg_num.median()).fillna(0.0)
    for col in avg_num_imp.columns:
        df_averaged[col] = avg_num_imp[col]

    plot_num = df_plot_level.select_dtypes(include=[np.number])
    plot_num_imp = plot_num.fillna(plot_num.median()).fillna(0.0)
    for col in plot_num_imp.columns:
        df_plot_level[col] = plot_num_imp[col]
    
    print(f"Saving Master ML Plot-Level Parquet: shape {df_plot_level.shape}")
    df_plot_level.to_parquet('/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.parquet', index=False)
    df_plot_level.to_csv('/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.csv', index=False)
    
    print(f"Saving Master ML Replicate-Averaged Parquet: shape {df_averaged.shape}")
    df_averaged.to_parquet('/mnt/Data/LIRF/Scripts/master_data_ml_averaged.parquet', index=False)
    
    print("Master datasets successfully generated and saved!")

if __name__ == '__main__':
    main()
