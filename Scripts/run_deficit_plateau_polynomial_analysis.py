import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def main():
    print("=" * 80)
    print("STARTING DEFICIT PLATEAU ANALYSIS (CUBIC VS. SPLINE/GAM FIT WITH YFE)")
    print("=" * 80)
    
    excel_path = '/mnt/Data/LIRF/output_021926.xlsx'
    if os.path.exists(excel_path):
        df_raw = pd.read_excel(excel_path, sheet_name='Annual Plot Data')
    else:
        df_raw = pd.read_parquet('/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet')

    df_raw = df_raw.rename(columns={c: str(c).strip() for c in df_raw.columns})
    y_col = [c for c in df_raw.columns if 'year' in c.lower()][0]
    g_col = [c for c in df_raw.columns if 'yield' in c.lower()][0]
    t_col = [c for c in df_raw.columns if 'treatment' in c.lower()][0]

    if 'crop' in [c.lower() for c in df_raw.columns]:
        c_col = [c for c in df_raw.columns if 'crop' in c.lower()][0]
        plot_df = df_raw[(df_raw[y_col].notna()) & (df_raw[c_col].astype(str).str.lower().str.contains('corn'))].copy()
    else:
        plot_df = df_raw[df_raw[y_col].notna()].copy()

    plot_df['year'] = plot_df[y_col].astype(int)
    plot_df = plot_df[plot_df[g_col].notna()].copy()
    plot_df['grain_yield'] = plot_df[g_col]
    plot_df['treatment'] = plot_df[t_col]

    max_y = plot_df.groupby('year')['grain_yield'].transform('max')
    plot_df['yield_loss_pct'] = ((max_y - plot_df['grain_yield']) / max_y) * 100.0

    map_path = '/mnt/Data/LIRF/Scripts/treatment_et_conversion_map.csv'
    trt_lookup = {}
    if os.path.exists(map_path):
        map_df = pd.read_csv(map_path)
        for _, r in map_df.iterrows():
            t_name = str(r['Treatment']).strip()
            try:
                veg_eq = float(r['ET Eq Veg'])
                rep_eq = float(r['ET Eq Rep'])
                trt_lookup[t_name] = (veg_eq + rep_eq) / 2.0
            except:
                pass

    def get_et_target(row):
        trt = str(row['treatment']).strip().upper()
        if 'SWB_FI' in trt: return 100.0
        elif 'DANS_HI' in trt: return 65.0
        elif 'RSRZ_LO' in trt or 'RSRZ' in trt: return 55.0
        elif 'DANS_LO' in trt or 'DANS' in trt: return 40.0
        elif 'CLASSIFY' in trt or 'WISE' in trt: return 80.0

        if trt in trt_lookup: return trt_lookup[trt]
        try:
            val = float(trt)
            if val <= 1.0: return val * 100.0
            else: return val
        except: pass
        try:
            parts = trt.split('/')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except: pass
        return 100.0

    plot_df['et_target'] = plot_df.apply(get_et_target, axis=1).clip(40.0, 100.0)
    clean_df = plot_df.dropna(subset=['et_target', 'yield_loss_pct']).copy()

    # 3-sigma studentized residual filter
    X_base = sm.add_constant(clean_df[['et_target']])
    ols_base = sm.OLS(clean_df['yield_loss_pct'], X_base).fit()
    infl = ols_base.get_influence()
    clean_df['studentized_resid'] = infl.resid_studentized_internal
    clean_df = clean_df[clean_df['studentized_resid'].abs() <= 3.0].copy()

    clean_df['et_target_sq'] = clean_df['et_target'] ** 2
    clean_df['et_target_cu'] = clean_df['et_target'] ** 3
    df_yfe = pd.get_dummies(clean_df['year'], prefix='year', drop_first=True).astype(float)
    
    # 1. Cubic Model with Year Fixed Effects
    X_cubic = pd.concat([
        pd.DataFrame({
            'const': 1.0,
            'et_target': clean_df['et_target'],
            'et_target_sq': clean_df['et_target_sq'],
            'et_target_cu': clean_df['et_target_cu']
        }, index=clean_df.index),
        df_yfe
    ], axis=1)

    m_cubic_yfe = sm.OLS(clean_df['yield_loss_pct'], X_cubic).fit()

    target_grid = np.linspace(40, 100, 200)
    X_grid_const = pd.DataFrame(0.0, index=range(len(target_grid)), columns=X_cubic.columns)
    X_grid_const['const'] = 1.0
    X_grid_const['et_target'] = target_grid
    X_grid_const['et_target_sq'] = target_grid ** 2
    X_grid_const['et_target_cu'] = target_grid ** 3
    for col in X_cubic.columns:
        if col.startswith('year_'):
            X_grid_const[col] = df_yfe[col].mean()

    pred_res = m_cubic_yfe.get_prediction(X_grid_const).summary_frame(alpha=0.05)
    pred_fit_cubic = pred_res['mean']

    # Equi-Loss threshold dynamically for cubic
    grid_eval = np.linspace(40, 100, 1000)
    X_grid_eval = pd.DataFrame(0.0, index=range(len(grid_eval)), columns=X_cubic.columns)
    X_grid_eval['const'] = 1.0
    X_grid_eval['et_target'] = grid_eval
    X_grid_eval['et_target_sq'] = grid_eval ** 2
    X_grid_eval['et_target_cu'] = grid_eval ** 3
    for col in X_cubic.columns:
        if col.startswith('year_'):
            X_grid_eval[col] = df_yfe[col].mean()
    pred_eval = m_cubic_yfe.predict(X_grid_eval)
    loss_100 = pred_eval.iloc[-1]
    sub_m = grid_eval < 90.0
    idx_eq = np.argmin(np.abs(pred_eval[sub_m] - loss_100))
    et_equal = grid_eval[sub_m][idx_eq]

    # 2. Penalized B-Spline GAM Fit WITH Year Fixed Effects (YFE)
    x_vec = clean_df['et_target'].values
    y_vec = clean_df['yield_loss_pct'].values
    
    spline_trans = SplineTransformer(degree=3, n_knots=5, include_bias=False)
    spline_basis = spline_trans.fit_transform(x_vec.reshape(-1, 1))
    spline_cols = {f'spline_b{i}': spline_basis[:, i] for i in range(spline_basis.shape[1])}
    spline_df = pd.DataFrame(spline_cols, index=clean_df.index)
    spline_df['const'] = 1.0

    X_spline = pd.concat([spline_df, df_yfe], axis=1)
    m_spline_yfe = sm.OLS(y_vec, X_spline).fit()

    # Grid Prediction for Spline GAM YFE
    spline_grid_basis = spline_trans.transform(target_grid.reshape(-1, 1))
    X_spline_grid = pd.DataFrame(0.0, index=range(len(target_grid)), columns=X_spline.columns)
    X_spline_grid['const'] = 1.0
    for i in range(spline_grid_basis.shape[1]):
        X_spline_grid[f'spline_b{i}'] = spline_grid_basis[:, i]
    for col in X_spline.columns:
        if col.startswith('year_'):
            X_spline_grid[col] = df_yfe[col].mean()

    pred_res_spline = m_spline_yfe.get_prediction(X_spline_grid).summary_frame(alpha=0.05)
    pred_fit_gam = pred_res_spline['mean']

    metrics = {
        'Cubic_YFE': {
            'R2': round(m_cubic_yfe.rsquared, 4),
            'Adj_R2': round(m_cubic_yfe.rsquared_adj, 4),
            'RMSE': round(np.sqrt(np.mean(m_cubic_yfe.resid ** 2)), 2),
            'AIC': round(m_cubic_yfe.aic, 2)
        },
        'Spline_GAM_YFE': {
            'R2': round(m_spline_yfe.rsquared, 4),
            'Adj_R2': round(m_spline_yfe.rsquared_adj, 4),
            'RMSE': round(np.sqrt(np.mean(m_spline_yfe.resid ** 2)), 2),
            'AIC': round(m_spline_yfe.aic, 2)
        }
    }

    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)
    json_path = os.path.join(fig_dir, 'deficit_plateau_response_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    fig_path = os.path.join(fig_dir, 'low_deficit_plateau_curve.png')

    plt.figure(figsize=(10, 6.5))

    # All Data Points in Solid Black
    plt.scatter(clean_df['et_target'], clean_df['yield_loss_pct'],
                color='#111111', label=f'Trial Plot Observations (N={len(clean_df)})',
                s=45, alpha=0.65, edgecolor='k', linewidth=0.4, zorder=5)

    # Cubic Polynomial Fit Curve
    plt.plot(target_grid, pred_fit_cubic, color='#1A365D', linewidth=2.8, linestyle='--',
             label=f'Cubic Polynomial Fit (YFE, $R^2={m_cubic_yfe.rsquared:.3f}$)', zorder=9)

    # Spline GAM Fit Curve
    plt.plot(target_grid, pred_fit_gam, color='#d7191c', linewidth=3.5,
             label=f'Spline GAM Fit (YFE, $R^2={m_spline_yfe.rsquared:.3f}$)', zorder=10)

    # Shading and Equi-Loss threshold
    plt.axvspan(et_equal, 100, color='green', alpha=0.10, label=f'Low-Deficit Plateau ({et_equal:.1f}–100% ET)')
    plt.axvspan(40, et_equal, color='red', alpha=0.07, label=f'Accelerating Loss Region (<{et_equal:.1f}% ET)')
    plt.axvline(et_equal, color='darkgreen', linestyle='--', linewidth=1.8, label=f'Equi-Loss Threshold ({et_equal:.1f}%)', zorder=12)

    plt.xlabel('Irrigation Target (% Seasonal ET Demand)', fontweight='bold', fontsize=11)
    plt.ylabel('Yield Reduction (%)', fontweight='bold', fontsize=11)
    plt.title('Low-Deficit Plateau Effect: Polynomial vs. Spline GAM Fit (With Year Fixed Effects)', fontweight='bold', fontsize=12, pad=12)
    
    # FLIPPED: High ET (100%) on the LEFT, Low ET (40%) on the RIGHT
    plt.gca().set_xlim(101.0, 39.0)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.90, edgecolor='#cccccc', fontsize=8.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\nSaved updated deficit plateau plot comparing Cubic vs. Spline GAM to {fig_path}")

if __name__ == '__main__':
    main()
