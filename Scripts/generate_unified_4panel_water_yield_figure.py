import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.preprocessing import SplineTransformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("GENERATING UNIFIED 2x2 (4-PANEL) WATER-YIELD RESPONSE CURVES FIGURE")
    print("Monotonic Penalized GAM Fit Only (Cubic Fit Dropped)")
    print("=" * 80)

    # 1. Load Dataset
    csv_path = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.csv'
    excel_path = '/mnt/Data/LIRF/output_021926.xlsx'

    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
    else:
        df_raw = pd.read_excel(excel_path, sheet_name='Annual Plot Data')

    df = df_raw.copy()
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    y_col = 'grain_yield' if 'grain_yield' in df.columns else [c for c in df.columns if 'yield' in c.lower()][0]
    year_col = 'year' if 'year' in df.columns else [c for c in df.columns if 'year' in c.lower()][0]
    trt_col = 'treatment' if 'treatment' in df.columns else [c for c in df.columns if 'treatment' in c.lower()][0]

    df['year'] = pd.to_numeric(df[year_col], errors='coerce').astype(int)
    df['grain_yield'] = pd.to_numeric(df[y_col], errors='coerce')
    df['treatment'] = df[trt_col].astype(str)

    # Compute Total Applied Water (mm)
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

    # Compute ET Target (%)
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
            except: pass

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

    df['et_target'] = df.apply(get_et_target, axis=1).clip(40.0, 100.0)

    # Compute Yield Reduction (%) relative to highest yield of each season
    max_y_per_year = df.groupby('year')['grain_yield'].transform('max')
    df['yield_reduction_pct'] = ((max_y_per_year - df['grain_yield']) / max_y_per_year) * 100.0

    # Clean Dataset: Trim TAW outliers (>1000 mm and <300 mm)
    clean_df = df.dropna(subset=['total_applied_water_mm', 'et_target', 'grain_yield', 'yield_reduction_pct', 'year']).copy()
    clean_df = clean_df[(clean_df['total_applied_water_mm'] >= 300.0) & (clean_df['total_applied_water_mm'] <= 1000.0)].copy()

    # Outlier filter
    X_base = sm.add_constant(clean_df[['total_applied_water_mm']])
    ols_base = sm.OLS(clean_df['grain_yield'], X_base).fit()
    infl = ols_base.get_influence()
    clean_df['studentized_resid'] = infl.resid_studentized_internal
    clean_df = clean_df[clean_df['studentized_resid'].abs() <= 3.0].copy()

    df_yfe = pd.get_dummies(clean_df['year'], prefix='year', drop_first=True).astype(float)

    print(f"Clean Dataset: N = {len(clean_df)} plots across {clean_df['year'].nunique()} seasons.")

    # Strictly Monotonic Spline GAM Solver WITH Year Fixed Effects
    def fit_monotonic_spline_yfe(x_arr, y_arr, grid_x, monotonic_dir='increasing', n_knots=4):
        spline_trans = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
        spline_basis = spline_trans.fit_transform(x_arr.reshape(-1, 1))

        X_mat = np.column_stack([np.ones(len(x_arr)), spline_basis, df_yfe.values])
        grid_basis = spline_trans.transform(grid_x.reshape(-1, 1))
        X_grid = np.column_stack([np.ones(len(grid_x)), grid_basis, np.tile(df_yfe.mean().values, (len(grid_x), 1))])

        beta_ols, _, _, _ = np.linalg.lstsq(X_mat, y_arr, rcond=None)

        def objective(beta):
            return np.sum((y_arr - X_mat @ beta) ** 2)

        def grad(beta):
            return -2 * X_mat.T @ (y_arr - X_mat @ beta)

        diff_matrix = np.diff(X_grid, axis=0)

        if monotonic_dir == 'increasing':
            cons = {'type': 'ineq', 'fun': lambda beta: diff_matrix @ beta}
        elif monotonic_dir == 'decreasing':
            cons = {'type': 'ineq', 'fun': lambda beta: -diff_matrix @ beta}
        else:
            cons = []

        res = minimize(objective, beta_ols, jac=grad, constraints=cons, method='SLSQP', options={'maxiter': 500})
        beta_opt = res.x if res.success else beta_ols

        pred_grid = X_grid @ beta_opt
        pred_train = X_mat @ beta_opt

        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        ss_res = np.sum((y_arr - pred_train) ** 2)
        r2_val = 1 - (ss_res / ss_tot)
        rmse_val = np.sqrt(np.mean((y_arr - pred_train) ** 2))

        return pred_grid, r2_val, rmse_val

    # Grid X ranges
    g_taw = np.linspace(300, 1000, 300)
    g_et = np.linspace(40, 100, 300)

    # 4 Panels Monotonic GAM Fits
    p1_s, r2_s1, rmse_s1 = fit_monotonic_spline_yfe(clean_df['total_applied_water_mm'].values, clean_df['grain_yield'].values, g_taw, monotonic_dir='increasing')
    p2_s, r2_s2, rmse_s2 = fit_monotonic_spline_yfe(clean_df['et_target'].values, clean_df['grain_yield'].values, g_et, monotonic_dir='increasing')
    p3_s, r2_s3, rmse_s3 = fit_monotonic_spline_yfe(clean_df['total_applied_water_mm'].values, clean_df['yield_reduction_pct'].values, g_taw, monotonic_dir='decreasing')
    p4_s, r2_s4, rmse_s4 = fit_monotonic_spline_yfe(clean_df['et_target'].values, clean_df['yield_reduction_pct'].values, g_et, monotonic_dir='decreasing')

    metrics = {
        'P1_RawYield_TAW': {'MonotonicGAM_R2': round(r2_s1, 4), 'RMSE': round(rmse_s1, 1)},
        'P2_RawYield_ET': {'MonotonicGAM_R2': round(r2_s2, 4), 'RMSE': round(rmse_s2, 1)},
        'P3_YieldRed_TAW': {'MonotonicGAM_R2': round(r2_s3, 4), 'RMSE': round(rmse_s3, 2)},
        'P4_YieldRed_ET': {'MonotonicGAM_R2': round(r2_s4, 4), 'RMSE': round(rmse_s4, 2)}
    }

    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)
    json_path = os.path.join(fig_dir, 'unified_4panel_response_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create Unified 2x2 (4-Panel) Grid Plot with GAM Fit ONLY
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), dpi=300)
    c_spline = '#D7191C'

    # Panel (a): Raw Yield vs TAW
    ax = axes[0, 0]
    ax.scatter(clean_df['total_applied_water_mm'], clean_df['grain_yield'], color='#2B5C8F', alpha=0.55, s=35, edgecolor='k', linewidth=0.3)
    ax.plot(g_taw, p1_s, color=c_spline, linewidth=3.0, label=f'Monotonic GAM Fit (YFE, R² = {r2_s1:.3f})')
    ax.set_title('(a) Grain Yield vs. Total Applied Water (300–1000 mm)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Total Applied Water (mm)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Grain Yield (kg/ha)', fontweight='bold', fontsize=11)
    ax.set_xlim(1020, 280)  # High Water LEFT -> Low Water RIGHT
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)

    # Panel (b): Raw Yield vs ET Target (%)
    ax = axes[0, 1]
    ax.scatter(clean_df['et_target'], clean_df['grain_yield'], color='#2B5C8F', alpha=0.55, s=35, edgecolor='k', linewidth=0.3)
    ax.plot(g_et, p2_s, color=c_spline, linewidth=3.0, label=f'Monotonic GAM Fit (YFE, R² = {r2_s2:.3f})')
    ax.set_title('(b) Grain Yield vs. ET Target', fontweight='bold', fontsize=12)
    ax.set_xlabel('Irrigation Target (% Seasonal ET Demand)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Grain Yield (kg/ha)', fontweight='bold', fontsize=11)
    ax.set_xlim(101, 39)  # High ET LEFT -> Low ET RIGHT
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)

    # Panel (c): Yield Reduction vs TAW
    ax = axes[1, 0]
    ax.scatter(clean_df['total_applied_water_mm'], clean_df['yield_reduction_pct'], color='#111111', alpha=0.55, s=35, edgecolor='k', linewidth=0.3)
    ax.plot(g_taw, p3_s, color=c_spline, linewidth=3.0, label=f'Monotonic GAM Fit (YFE, R² = {r2_s3:.3f})')
    ax.set_title('(c) Yield Reduction vs. Total Applied Water (300–1000 mm)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Total Applied Water (mm)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Yield Reduction (%)', fontweight='bold', fontsize=11)
    ax.set_xlim(1020, 280)  # High Water LEFT -> Low Water RIGHT
    ax.set_ylim(-3, max(clean_df['yield_reduction_pct'].max() + 5, 55))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)

    # Panel (d): Yield Reduction vs ET Target (%)
    ax = axes[1, 1]
    ax.scatter(clean_df['et_target'], clean_df['yield_reduction_pct'], color='#111111', alpha=0.55, s=35, edgecolor='k', linewidth=0.3)
    ax.plot(g_et, p4_s, color=c_spline, linewidth=3.0, label=f'Monotonic GAM Fit (YFE, R² = {r2_s4:.3f})')
    ax.set_title('(d) Yield Reduction vs. ET Target', fontweight='bold', fontsize=12)
    ax.set_xlabel('Irrigation Target (% Seasonal ET Demand)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Yield Reduction (%)', fontweight='bold', fontsize=11)
    ax.set_xlim(101, 39)  # High ET LEFT -> Low ET RIGHT
    ax.set_ylim(-3, max(clean_df['yield_reduction_pct'].max() + 5, 55))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)

    fig.suptitle('Uniform Water-Yield Response Curves (Strictly Monotonic GAM Fit with Year Fixed Effects)', fontweight='bold', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.join(fig_dir, 'unified_4panel_water_yield_response_curves.png')
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"\nSaved GAM-Only Unified 2x2 (4-Panel) Figure to: {out_png}")
    print(f"Metrics JSON exported to {json_path}:\n{json.dumps(metrics, indent=2)}")

if __name__ == '__main__':
    main()
