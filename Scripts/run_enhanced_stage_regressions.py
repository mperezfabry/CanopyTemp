import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

def prepare_enhanced_4stage_dataset():
    """
    Builds the stage-by-stage stress features for Early_Veg, Late_Veg, Flowering, GrainFill.
    Computes 4 formulations per metric:
    1. Seasonal / Stage Mean Stress
    2. Threshold Excess Integral (Stress Degree Days)
    3. Peak / Upper Quantile Stress (90th Percentile)
    4. Kc Stage-Sensitivity Weighted Integral
    """
    print("=== PREPARING ENHANCED 4-STAGE DATASET WITH FEATURE FORMULATIONS ===")
    p_ml = '/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet'
    if not os.path.exists(p_ml):
        p_ml = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.parquet'

    df_raw = pd.read_parquet(p_ml)
    if 'crop' in df_raw.columns:
        df_raw = df_raw[df_raw['crop'] != 'Sunflower'].copy()

    # Define stage phase boundaries
    def get_phase(doy):
        if doy <= 175:
            return 'Early_Veg'
        elif doy <= 210:
            return 'Late_Veg'
        elif doy <= 230:
            return 'Flowering'
        else:
            return 'GrainFill'

    if 'doy' not in df_raw.columns:
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw['doy'] = df_raw['timestamp'].dt.dayofyear
        df_raw['year'] = df_raw['timestamp'].dt.year

    df_raw['phase'] = df_raw['doy'].apply(get_phase)

    # SWC and Ks calculation
    swc_cols = [c for c in ['swc_depth_30', 'swc_depth_60', 'swc_depth_90', 'swc_depth_120'] if c in df_raw.columns]
    if not swc_cols:
        swc_cols = [c for c in df_raw.columns if 'swc' in c.lower() or 'vwc' in c.lower()]

    if swc_cols:
        df_raw['swc_rz'] = df_raw[swc_cols].mean(axis=1)
    else:
        df_raw['swc_rz'] = 0.20

    p_frac, fc, wp = 0.55, 0.33, 0.13
    denom = (1.0 - p_frac) * (fc - wp)
    df_raw['Ks_obs'] = ((df_raw['swc_rz'] - wp) / denom).clip(0.0, 1.0)

    # Compute CWSI and DANS if missing
    if 'air_temp_c' in df_raw.columns and 'weather_air_temp_c' not in df_raw.columns:
        df_raw['weather_air_temp_c'] = df_raw['air_temp_c']

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

    if 'CWSI' not in df_raw.columns:
        df_raw['CWSI'] = 0.0
    if 'Solar_noon_Avg' not in df_raw.columns:
        df_raw['Solar_noon_Avg'] = df_raw.get('canopy_temp', 25.0)
    if 'DANS_aboveAvg' not in df_raw.columns:
        df_raw['DANS_aboveAvg'] = 0.0
    if 'Ks_linear' not in df_raw.columns:
        df_raw['Ks_linear'] = df_raw['Ks_obs']
    if 'Ks_curvilinear' not in df_raw.columns:
        df_raw['Ks_curvilinear'] = df_raw['Ks_obs']

    # Target calculation
    df_raw = df_raw[df_raw['grain_yield'].notna()].copy()
    max_y = df_raw.groupby('year')['grain_yield'].transform('max')
    df_raw['relative_yield_reduction'] = (max_y - df_raw['grain_yield']) / max_y

    # Kc weight mapping per phase
    kc_phase_weights = {
        'Early_Veg': 0.45,
        'Late_Veg': 1.05,
        'Flowering': 1.20,
        'GrainFill': 0.85
    }
    df_raw['kc_weight'] = df_raw['phase'].map(kc_phase_weights)

    # Calculate daily metric transformations for threshold excess
    df_raw['ex_CWSI'] = np.maximum(0.0, df_raw['CWSI'] - 0.35)
    df_raw['ex_Solar_noon_Avg'] = np.maximum(0.0, df_raw['Solar_noon_Avg'] - 25.0)
    df_raw['ex_DANS_aboveAvg'] = np.maximum(0.0, df_raw['DANS_aboveAvg'])
    df_raw['ex_swc_rz'] = np.maximum(0.0, 0.33 - df_raw['swc_rz'])
    df_raw['ex_Ks_obs'] = np.maximum(0.0, 1.0 - df_raw['Ks_obs'])
    df_raw['ex_Ks_linear'] = np.maximum(0.0, 1.0 - df_raw['Ks_linear'])
    df_raw['ex_Ks_curvilinear'] = np.maximum(0.0, 1.0 - df_raw['Ks_curvilinear'])

    metrics = ['CWSI', 'DANS_aboveAvg', 'swc_rz', 'Solar_noon_Avg', 'Ks_obs', 'Ks_linear', 'Ks_curvilinear']

    daily_aggs = df_raw.groupby(['year', 'plot', 'phase', 'doy']).agg({
        'CWSI': 'mean', 'ex_CWSI': 'mean',
        'DANS_aboveAvg': 'mean', 'ex_DANS_aboveAvg': 'mean',
        'swc_rz': 'mean', 'ex_swc_rz': 'mean',
        'Solar_noon_Avg': 'mean', 'ex_Solar_noon_Avg': 'mean',
        'Ks_obs': 'mean', 'ex_Ks_obs': 'mean',
        'Ks_linear': 'mean', 'ex_Ks_linear': 'mean',
        'Ks_curvilinear': 'mean', 'ex_Ks_curvilinear': 'mean',
        'kc_weight': 'first',
        'relative_yield_reduction': 'first',
        'grain_yield': 'first'
    }).reset_index()

    p90 = lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan

    phase_summary = daily_aggs.groupby(['year', 'plot', 'phase']).agg({
        # 1. Mean
        'CWSI': 'mean', 'DANS_aboveAvg': 'mean', 'swc_rz': 'mean',
        'Solar_noon_Avg': 'mean', 'Ks_obs': 'mean', 'Ks_linear': 'mean', 'Ks_curvilinear': 'mean',
        # 2. Excess Threshold Sum
        'ex_CWSI': 'sum', 'ex_DANS_aboveAvg': 'sum', 'ex_swc_rz': 'sum',
        'ex_Solar_noon_Avg': 'sum', 'ex_Ks_obs': 'sum', 'ex_Ks_linear': 'sum', 'ex_Ks_curvilinear': 'sum',
        # 3. Peak 90th Percentile
        'CWSI_p90': ('CWSI', p90), 'DANS_aboveAvg_p90': ('DANS_aboveAvg', p90), 'swc_rz_p90': ('swc_rz', p90),
        'Solar_noon_Avg_p90': ('Solar_noon_Avg', p90), 'Ks_obs_p90': ('Ks_obs', p90),
        'Ks_linear_p90': ('Ks_linear', p90), 'Ks_curvilinear_p90': ('Ks_curvilinear', p90),
        # Meta targets
        'relative_yield_reduction': 'first', 'grain_yield': 'first'
    }).reset_index()

    # Flatten tuple column names
    phase_summary.columns = [col[0] if isinstance(col, tuple) else col for col in phase_summary.columns]

    # Calculate 4. Kc Weighted Stress Integral
    for m in metrics:
        daily_aggs[f'kc_weighted_{m}'] = daily_aggs[f'ex_{m}'] * daily_aggs['kc_weight']

    kc_summary = daily_aggs.groupby(['year', 'plot', 'phase'])[[f'kc_weighted_{m}' for m in metrics]].sum().reset_index()
    phase_summary = pd.merge(phase_summary, kc_summary, on=['year', 'plot', 'phase'], how='left')

    phases = ['Early_Veg', 'Late_Veg', 'Flowering', 'GrainFill']
    formulations = ['mean', 'excess_sum', 'p90', 'kc_weighted_sum']
    
    pivoted_dfs = {}
    for form in formulations:
        if form == 'mean':
            val_cols = metrics
        elif form == 'excess_sum':
            val_cols = [f'ex_{m}' for m in metrics]
        elif form == 'p90':
            val_cols = [f'{m}_p90' for m in metrics]
        elif form == 'kc_weighted_sum':
            val_cols = [f'kc_weighted_{m}' for m in metrics]

        piv = phase_summary.pivot(index=['year', 'plot', 'relative_yield_reduction', 'grain_yield'], columns='phase', values=val_cols)
        piv.columns = [f"{m}_{p}" for m, p in piv.columns]
        piv = piv.reset_index()
        pivoted_dfs[form] = piv

    return pivoted_dfs

def run_stage_regressions(df, form_name):
    """
    Fits Fixed Effects OLS and ElasticNetCV models for a given formulation.
    """
    feature_cols = [c for c in df.columns if c not in ['year', 'plot', 'relative_yield_reduction', 'grain_yield']]
    
    df_clean = df.dropna(subset=['relative_yield_reduction']).copy()
    X = df_clean[feature_cols].copy()
    X = X.fillna(X.median()).fillna(0.0)
    y = df_clean['relative_yield_reduction'].values

    df_yfe = pd.get_dummies(df_clean['year'], prefix='year', drop_first=True).astype(float)
    X_fe = pd.concat([X, df_yfe], axis=1)
    X_fe = sm.add_constant(X_fe)

    # 1. OLS Fixed Effects
    ols_model = sm.OLS(y, X_fe).fit()
    y_pred_ols = ols_model.predict(X_fe)
    r2_ols = r2_score(y, y_pred_ols)
    rmse_ols = np.sqrt(mean_squared_error(y, y_pred_ols))

    # 2. ElasticNetCV (Standardized Features + YFE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_enet = np.column_stack([X_scaled, df_yfe.values])

    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0], cv=5, random_state=42, max_iter=5000)
    enet.fit(X_enet, y)
    y_pred_enet = enet.predict(X_enet)
    r2_enet = r2_score(y, y_pred_enet)
    rmse_enet = np.sqrt(mean_squared_error(y, y_pred_enet))

    coef_df = pd.DataFrame({
        'Feature': list(feature_cols) + list(df_yfe.columns),
        'OLS_Coef': ols_model.params[1:].values[:len(feature_cols) + len(df_yfe.columns)],
        'OLS_Pval': ols_model.pvalues[1:].values[:len(feature_cols) + len(df_yfe.columns)],
        'ElasticNet_Coef': enet.coef_
    })

    return {
        'Formulation': form_name,
        'OLS_R2': round(r2_ols, 4),
        'OLS_RMSE': round(rmse_ols, 4),
        'ElasticNet_R2': round(r2_enet, 4),
        'ElasticNet_RMSE': round(rmse_enet, 4),
        'Coefficients': coef_df
    }

def main():
    pivoted_dfs = prepare_enhanced_4stage_dataset()
    results = []
    
    print("\n--- ENHANCED 4-STAGE STRESS REGRESSION RESULTS ---")
    for form_name, df_form in pivoted_dfs.items():
        res = run_stage_regressions(df_form, form_name)
        results.append(res)
        print(f"[{form_name.upper()}] OLS R² = {res['OLS_R2']} (RMSE = {res['OLS_RMSE']}) | ElasticNet R² = {res['ElasticNet_R2']} (RMSE = {res['ElasticNet_RMSE']})")

    out_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Formulation': r['Formulation'],
            'OLS_R2': r['OLS_R2'],
            'OLS_RMSE': r['OLS_RMSE'],
            'ElasticNet_R2': r['ElasticNet_R2'],
            'ElasticNet_RMSE': r['ElasticNet_RMSE']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('/mnt/Data/LIRF/Scripts/enhanced_stage_regressions_metrics.csv', index=False)
    print(f"\nSaved regression metrics summary to: /mnt/Data/LIRF/Scripts/enhanced_stage_regressions_metrics.csv")

if __name__ == '__main__':
    main()
