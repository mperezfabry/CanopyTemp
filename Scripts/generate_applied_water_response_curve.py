import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=== GENERATING TOTAL APPLIED WATER VS GRAIN YIELD RESPONSE CURVES (APPLES-TO-APPLES YFE FITS) ===")
    
    csv_path = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.csv'
    parquet_path = '/mnt/Data/LIRF/lirf_merged_dataset_ml.parquet'
    excel_path = '/mnt/Data/LIRF/output_021926.xlsx'
    
    if os.path.exists(csv_path):
        print(f"Loading plot-level dataset from: {csv_path}")
        df_raw = pd.read_csv(csv_path)
    elif os.path.exists(parquet_path):
        print(f"Loading dataset from: {parquet_path}")
        df_raw = pd.read_parquet(parquet_path)
    else:
        print(f"Loading dataset from: {excel_path}")
        df_raw = pd.read_excel(excel_path, sheet_name='Annual Plot Data')
        
    df = df_raw.copy()
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    
    y_col = 'grain_yield' if 'grain_yield' in df.columns else [c for c in df.columns if 'yield' in c.lower()][0]
    year_col = 'year' if 'year' in df.columns else [c for c in df.columns if 'year' in c.lower()][0]
    
    df['year'] = pd.to_numeric(df[year_col], errors='coerce').astype(int)
    df['grain_yield'] = pd.to_numeric(df[y_col], errors='coerce')
    
    stages = ['Early_Veg', 'Late_Veg', 'Flowering', 'GrainFill']
    p_stage_cols = [f'weather_precip_mm_sum_{s}' for s in stages if f'weather_precip_mm_sum_{s}' in df.columns]
    i_stage_cols = [f'irrigation_depth_mm_sum_{s}' for s in stages if f'irrigation_depth_mm_sum_{s}' in df.columns]
    
    if p_stage_cols and i_stage_cols:
        df_p = df[p_stage_cols].copy()
        for col in p_stage_cols:
            df_p[col] = df_p[col].apply(lambda v: v if v <= 350.0 else np.nan)
            
        df['total_precip_mm'] = df_p.sum(axis=1, skipna=True)
        df['total_irrig_mm'] = df[i_stage_cols].sum(axis=1, skipna=True)
        df['total_applied_water_mm'] = df['total_precip_mm'] + df['total_irrig_mm']
    else:
        p_cols = [c for c in df.columns if 'precip' in c.lower()]
        i_cols = [c for c in df.columns if 'irrig' in c.lower()]
        df['total_precip_mm'] = df[p_cols].sum(axis=1) if p_cols else 0.0
        df['total_irrig_mm'] = df[i_cols].sum(axis=1) if i_cols else 0.0
        df['total_applied_water_mm'] = df['total_precip_mm'] + df['total_irrig_mm']
        
    clean_df = df.dropna(subset=['total_applied_water_mm', 'grain_yield', 'year']).copy()
    clean_df = clean_df[(clean_df['total_applied_water_mm'] >= 150.0) & (clean_df['total_applied_water_mm'] <= 1200.0)].copy()
    
    # 3-sigma studentized residual outlier filtering
    X_base = sm.add_constant(clean_df[['total_applied_water_mm']])
    ols_base = sm.OLS(clean_df['grain_yield'], X_base).fit()
    infl = ols_base.get_influence()
    clean_df['studentized_resid'] = infl.resid_studentized_internal
    clean_df = clean_df[clean_df['studentized_resid'].abs() <= 3.0].copy()
    
    print(f"\nFinal dataset for response curves: N = {len(clean_df)} plots across {clean_df['year'].nunique()} seasons ({clean_df['year'].min()}–{clean_df['year'].max()}).")
    
    W = clean_df['total_applied_water_mm'].values
    y = clean_df['grain_yield'].values
    df_yfe = pd.get_dummies(clean_df['year'], prefix='year', drop_first=True).astype(float)
    
    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    years = sorted(clean_df['year'].unique())
    cmap = matplotlib.colormaps.get_cmap('tab20')
    year_color_map = {yr: cmap(i % 20) for i, yr in enumerate(years)}
    
    w_min, w_max = W.min(), W.max()
    grid_w = np.linspace(w_min, w_max, 500)
    
    degree_info = {
        1: ('Linear', 'total_applied_water_linear.png'),
        2: ('Quadratic', 'total_applied_water_quadratic.png'),
        3: ('Cubic', 'total_applied_water_cubic.png'),
        4: ('Spline GAM', 'total_applied_water_gam_spline.png')
    }
    
    panel_fits = {}
    metrics_summary = {}

    # Polynomial Fits with YFE (Linear, Quadratic, Cubic)
    for deg in [1, 2, 3]:
        deg_name, filename = degree_info[deg]
        
        poly_dict = {'const': 1.0}
        for d in range(1, deg + 1):
            poly_dict[f'W_pow_{d}'] = W ** d
            
        X_poly = pd.concat([pd.DataFrame(poly_dict, index=clean_df.index), df_yfe], axis=1)
        m = sm.OLS(y, X_poly).fit()
        
        residuals = m.resid
        rmse = np.sqrt(np.mean(residuals ** 2))
        rse = np.sqrt(m.mse_resid)
        mae = np.mean(np.abs(residuals))
        
        X_grid = pd.DataFrame(0.0, index=range(len(grid_w)), columns=X_poly.columns)
        X_grid['const'] = 1.0
        for d in range(1, deg + 1):
            X_grid[f'W_pow_{d}'] = grid_w ** d
        for col in X_poly.columns:
            if col.startswith('year_'):
                X_grid[col] = df_yfe[col].mean()
                
        pred_res = m.get_prediction(X_grid).summary_frame(alpha=0.05)
        pred_fit = pred_res['mean']
        pi_lower = pred_res['obs_ci_lower']
        pi_upper = pred_res['obs_ci_upper']
        
        panel_fits[deg] = (m, pred_fit, pi_lower, pi_upper, rmse, rse, mae)
        metrics_summary[deg_name] = {
            'Model': f"{deg_name} (YFE)",
            'R2': round(m.rsquared, 4),
            'Adj_R2': round(m.rsquared_adj, 4),
            'RMSE': round(rmse, 2),
            'RSE': round(rse, 2),
            'MAE': round(mae, 2),
            'AIC': round(m.aic, 2),
            'BIC': round(m.bic, 2)
        }

    # B-Spline GAM Fit WITH Year Fixed Effects (YFE) in Statsmodels OLS
    spline_trans = SplineTransformer(degree=3, n_knots=5, include_bias=False)
    spline_basis = spline_trans.fit_transform(W.reshape(-1, 1))
    spline_cols = {f'spline_b{i}': spline_basis[:, i] for i in range(spline_basis.shape[1])}
    spline_df = pd.DataFrame(spline_cols, index=clean_df.index)
    spline_df['const'] = 1.0

    X_spline = pd.concat([spline_df, df_yfe], axis=1)
    m_spline = sm.OLS(y, X_spline).fit()
    
    residuals_spline = m_spline.resid
    rmse_spline = np.sqrt(np.mean(residuals_spline ** 2))
    rse_spline = np.sqrt(m_spline.mse_resid)
    mae_spline = np.mean(np.abs(residuals_spline))

    # Grid Prediction for Spline GAM YFE
    spline_grid_basis = spline_trans.transform(grid_w.reshape(-1, 1))
    X_spline_grid = pd.DataFrame(0.0, index=range(len(grid_w)), columns=X_spline.columns)
    X_spline_grid['const'] = 1.0
    for i in range(spline_grid_basis.shape[1]):
        X_spline_grid[f'spline_b{i}'] = spline_grid_basis[:, i]
    for col in X_spline.columns:
        if col.startswith('year_'):
            X_spline_grid[col] = df_yfe[col].mean()

    pred_res_spline = m_spline.get_prediction(X_spline_grid).summary_frame(alpha=0.05)
    pred_fit_spline = pred_res_spline['mean']
    pi_lower_spline = pred_res_spline['obs_ci_lower']
    pi_upper_spline = pred_res_spline['obs_ci_upper']

    panel_fits[4] = (m_spline, pred_fit_spline, pi_lower_spline, pi_upper_spline, rmse_spline, rse_spline, mae_spline)
    metrics_summary['Spline_GAM'] = {
        'Model': "Spline GAM (YFE)",
        'R2': round(m_spline.rsquared, 4),
        'Adj_R2': round(m_spline.rsquared_adj, 4),
        'RMSE': round(rmse_spline, 2),
        'RSE': round(rse_spline, 2),
        'MAE': round(mae_spline, 2),
        'AIC': round(m_spline.aic, 2),
        'BIC': round(m_spline.bic, 2)
    }

    # Export metrics JSON
    json_path = os.path.join(fig_dir, 'applied_water_response_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Exported metrics JSON to: {json_path}")

    # Standalone Spline GAM Plot (X-axis flipped so HIGH WATER is on the LEFT)
    plt.figure(figsize=(10.5, 6.8))
    for yr in years:
        sub = clean_df[clean_df['year'] == yr]
        plt.scatter(sub['total_applied_water_mm'], sub['grain_yield'],
                    color=year_color_map[yr], label=str(yr),
                    s=55, alpha=0.85, edgecolor='k', linewidth=0.5, zorder=5)
                    
    plt.plot(grid_w, pred_fit_spline, color='#d7191c', linewidth=3.5,
             label=f'Spline GAM Fit (YFE, R²={m_spline.rsquared:.3f}, RMSE={rmse_spline:.1f} kg/ha)', zorder=10)
             
    plt.fill_between(grid_w, pi_lower_spline, pi_upper_spline, color='#d7191c', alpha=0.15,
                     label='95% Prediction Interval', zorder=7)
                     
    plt.xlabel('Total Applied Water (Irrigation + Precipitation, mm)', fontweight='bold', fontsize=11)
    plt.ylabel('Grain Yield (kg/ha)', fontweight='bold', fontsize=11)
    plt.title(f'Grain Yield vs. Total Applied Water: Spline GAM Fit (2008–2024)\n[RMSE = {rmse_spline:.1f} kg/ha, RSE = {rse_spline:.1f} kg/ha]',
              fontweight='bold', fontsize=12.5, pad=12)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.gca().set_xlim(w_max + 20, w_min - 20)  # FLIPPED: High Water on LEFT, Low Water on RIGHT
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92,
               edgecolor='#cccccc', fontsize=8.5, ncol=2)
               
    plt.tight_layout()
    gam_out_path = os.path.join(fig_dir, 'total_applied_water_gam_spline.png')
    plt.savefig(gam_out_path, dpi=300)
    plt.close()
    print(f"Saved Spline GAM standalone plot to {gam_out_path}")

    # 2x2 Eyeballing Panel Plot including Spline GAM (FLIPPED: High Water on LEFT)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    display_names = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 4: 'Spline GAM'}
    
    for deg in range(1, 5):
        ax = axes[deg - 1]
        deg_name = display_names[deg]
        m, pred_fit, pi_lower, pi_upper, rmse, rse, mae = panel_fits[deg]
        
        for yr in years:
            sub = clean_df[clean_df['year'] == yr]
            ax.scatter(sub['total_applied_water_mm'], sub['grain_yield'],
                       color=year_color_map[yr], s=35, alpha=0.75, edgecolor='k', linewidth=0.3)
                       
        line_color = '#d7191c' if deg == 4 else '#08519c'
        ax.plot(grid_w, pred_fit, color=line_color, linewidth=2.8)
        ax.fill_between(grid_w, pi_lower, pi_upper, color=line_color, alpha=0.15)
        
        ax.set_title(f'({chr(96+deg)}) {deg_name} Fit  [R² = {m.rsquared:.3f} | RMSE = {rmse:.1f} kg/ha | AIC = {m.aic:.1f}]',
                     fontweight='bold', fontsize=11.0)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_xlim(w_max + 20, w_min - 20)  # FLIPPED: High Water on LEFT, Low Water on RIGHT
        
    fig.supxlabel('Total Applied Water (Irrigation + Precipitation, mm)', fontweight='bold', fontsize=12, y=0.07)
    fig.supylabel('Grain Yield (kg/ha)', fontweight='bold', fontsize=12, x=0.02)
    fig.suptitle('Grain Yield vs. Total Applied Water: Polynomials vs. Spline GAM Fit (With Year Fixed Effects)',
                 fontweight='bold', fontsize=14, y=0.94)
                 
    panel_path = os.path.join(fig_dir, 'total_applied_water_4panel_response_curves.png')
    plt.savefig(panel_path, dpi=300)
    plt.close()
    print(f"Saved updated 4-panel comparison plot to {panel_path}")

if __name__ == '__main__':
    main()
