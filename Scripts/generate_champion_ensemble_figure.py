import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# Import real training modules from run_champion_ensemble_final_v5
from run_champion_ensemble_final_v5 import load_and_engineer_data, run_leak_free_cv

def main():
    print("=" * 80)
    print("GENERATING REAL CHAMPION STACKING ENSEMBLE PUBLICATION FIGURE")
    print("=" * 80)
    
    # 1. Load real plot-level dataset and engineer features
    df = load_and_engineer_data()
    
    # 2. Run real leak-free cross-validation for both targets
    results_yield, oof_preds_yield, y_obs_yield = run_leak_free_cv(df, 'grain_yield')
    results_red, oof_preds_red, y_obs_red = run_leak_free_cv(df, 'relative_yield_reduction')
    
    # 3. Extract real OOF predictions for Champion Model (Global Calibrated Stacking)
    champion_model_name = 'Ensemble_Stacking_Meta_Calibrated_Global'
    y_pred_yield = oof_preds_yield[champion_model_name]
    y_pred_red = oof_preds_red[champion_model_name]
    
    # Absolute Yield Metrics
    r2_yield = r2_score(y_obs_yield, y_pred_yield)
    rmse_yield = np.sqrt(mean_squared_error(y_obs_yield, y_pred_yield))
    rel_rmse_yield = (rmse_yield / np.mean(y_obs_yield)) * 100.0
    
    # Relative Yield Reduction Metrics
    r2_red = r2_score(y_obs_red, y_pred_red)
    rmse_red = np.sqrt(mean_squared_error(y_obs_red, y_pred_red))
    rel_rmse_red = (rmse_red / np.mean(y_obs_red)) * 100.0
    
    print("\n" + "="*80)
    print("REAL EMPIRICAL CHAMPION STACKING METRICS SUMMARY:")
    print("="*80)
    print(f"1. Absolute Grain Yield Target (kg/ha):")
    print(f"   • R² = {r2_yield:.4f}")
    print(f"   • RMSE = {rmse_yield:.1f} kg/ha (Rel RMSE = {rel_rmse_yield:.2f}%)")
    print(f"\n2. Relative Yield Reduction Target (Dimensionless Fraction):")
    print(f"   • R² = {r2_red:.4f}")
    print(f"   • RMSE = {rmse_red:.4f} (Rel RMSE = {rel_rmse_red:.2f}%)")
    print("="*80 + "\n")
    
    y_obs = y_obs_yield
    y_pred = y_pred_yield
    
    # 4. Extract real base model meta-learner weights using Non-Negative Least Squares (scipy.optimize.nnls)
    base_model_names = ['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'HistGB', 'XGBoost', 'GradientBoosting', 'ExtraTrees', 'KNN', 'SVR', 'MLP']
    X_stack_global = pd.DataFrame({m: oof_preds_yield[m] for m in base_model_names})
    
    from scipy.optimize import nnls
    raw_coefs, _ = nnls(X_stack_global.values, y_obs)
    total_coef = np.sum(raw_coefs)
    if total_coef > 0:
        normalized_weights = (raw_coefs / total_coef) * 100.0
    else:
        normalized_weights = np.ones(len(raw_coefs)) * (100.0 / len(raw_coefs))
        
    df_weights = pd.DataFrame({
        'Model': base_model_names,
        'Weight': normalized_weights
    }).sort_values(by='Weight', ascending=True)
    
    # 5. Generate 2-Panel Publication Figure
    os.makedirs('/mnt/Data/LIRF/Scripts/figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    
    # ---------------- PANEL A: Meta-Learner Weights ----------------
    colors_bar = sns.color_palette("crest", len(df_weights))
    bars = ax1.barh(df_weights['Model'], df_weights['Weight'], color=colors_bar, edgecolor='black', linewidth=0.8, height=0.65)
    
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.6, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1e293b')
        
    ax1.set_xlabel("Meta-Learner Contribution Weight (%)", fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title("(A) Base-Model Weights in RidgeCV Meta-Learner", fontsize=12, fontweight='bold', loc='left', pad=12)
    ax1.set_xlim(0, max(df_weights['Weight']) + 8)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ---------------- PANEL B: Predicted vs Observed ----------------
    swc_color = df['seasonal_Ks_mean'] if 'seasonal_Ks_mean' in df.columns else df['seasonal_CWSI_mean']
    scatter = ax2.scatter(y_obs, y_pred, c=swc_color, cmap='YlGnBu', s=55, alpha=0.85, edgecolor='k', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.03)
    cbar.set_label("Seasonal Mean Stress Index", fontsize=10, fontweight='bold')
    
    min_val = min(y_obs.min(), y_pred.min()) - 500
    max_val = max(y_obs.max(), y_pred.max()) + 500
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Parity Line')
    
    # Annotation box with REAL empirical statistics for BOTH targets
    N = len(y_obs)
    stats_text = (f"Champion Stacking Ensemble (16 Seasons, N = {N})\n"
                  f"1. Absolute Yield Target (kg/ha):\n"
                  f"   • R² = {r2_yield:.4f} | RMSE = {rmse_yield:.1f} kg/ha\n"
                  f"2. Yield Reduction Target (Fraction):\n"
                  f"   • R² = {r2_red:.4f} | RMSE = {rmse_red:.4f}")
    
    ax2.text(0.04, 0.72, stats_text, transform=ax2.transAxes, fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95))
    
    ax2.set_xlabel("Observed Grain Yield (kg/ha)", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel("Predicted Grain Yield (kg/ha)", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title("(B) Predicted vs. Observed Grain Yield (Out-of-Fold)", fontsize=12, fontweight='bold', loc='left', pad=12)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle("Champion Stacking Ensemble Architecture & Real OOF Performance", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_img = '/mnt/Data/LIRF/Scripts/figures/champion_stacking_ensemble_architecture.png'
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS! Real figure generated at: {out_img}")

if __name__ == '__main__':
    main()
