import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

def main():
    print("=" * 80)
    print("GENERATING SHAP & FEATURE IMPORTANCE VISUALIZATIONS FOR EXTRATREES & GRADIENTBOOST")
    print("=" * 80)

    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)

    master_parquet = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.parquet'
    if not os.path.exists(master_parquet):
        master_parquet = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.csv'

    if not os.path.exists(master_parquet):
        print(f"Error: {master_parquet} not found.")
        return

    if master_parquet.endswith('.parquet'):
        df_master = pd.read_parquet(master_parquet)
    else:
        df_master = pd.read_csv(master_parquet)

    print(f"Loaded master plot-level dataset: {len(df_master)} rows.")

    ignore_cols = ['year', 'plot', 'treatment', 'grain_yield', 'relative_yield_reduction', 'veg', 'rep', 'crop']
    feature_cols = [c for c in df_master.columns if c not in ignore_cols]

    X = df_master[feature_cols].copy()
    medians = X.median()
    X = X.fillna(medians).fillna(0.0)
    
    target_col = 'grain_yield' if 'grain_yield' in df_master.columns else 'relative_yield_reduction'
    y = df_master[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. ExtraTrees Feature Importances
    print("\nTraining ExtraTrees Regressor...")
    et_model = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
    et_model.fit(X_scaled, y)
    df_et_fi = pd.DataFrame({'Feature': feature_cols, 'Importance': et_model.feature_importances_}).sort_values('Importance', ascending=False)

    # 2. GradientBoost Feature Importances
    print("Training GradientBoost Regressor...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    gb_model.fit(X_scaled, y)
    df_gb_fi = pd.DataFrame({'Feature': feature_cols, 'Importance': gb_model.feature_importances_}).sort_values('Importance', ascending=False)

    # Plot Top 12 Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top_et = df_et_fi.head(12).sort_values('Importance', ascending=True)
    axes[0].barh(top_et['Feature'], top_et['Importance'], color='#1f77b4', edgecolor='k', linewidth=0.6)
    axes[0].set_title("ExtraTrees Top 12 Feature Importances", fontweight='bold', fontsize=11)
    axes[0].set_xlabel("MDI Feature Importance", fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3, axis='x')

    top_gb = df_gb_fi.head(12).sort_values('Importance', ascending=True)
    axes[1].barh(top_gb['Feature'], top_gb['Importance'], color='#2ca02c', edgecolor='k', linewidth=0.6)
    axes[1].set_title("GradientBoost Top 12 Feature Importances", fontweight='bold', fontsize=11)
    axes[1].set_xlabel("MDI Feature Importance", fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3, axis='x')

    plt.tight_layout()
    out_fig_path = os.path.join(fig_dir, 'extratrees_gradientboost_feature_importance.png')
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSuccessfully generated SHAP / Feature Importance plot at: {out_fig_path}")

    # Print Top 10 Summary
    print("\n--- TOP 10 EXTRATREES FEATURES ---")
    print(df_et_fi.head(10).to_string(index=False))

    print("\n--- TOP 10 GRADIENT BOOST FEATURES ---")
    print(df_gb_fi.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
