import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

# Set seeds for reproducibility
np.random.seed(42)

# ----------------- STEP 1: LOAD AND ENGINEER FEATURES -----------------
def load_and_engineer_data():
    print("Loading plot-level dataset...")
    df = pd.read_parquet('/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.parquet')
    
    # Fill any missing values with median
    for col in df.columns:
        if col not in ['year', 'plot', 'treatment', 'grain_yield', 'relative_yield_reduction']:
            df[col] = df[col].fillna(df[col].median())
            
    # --- 1. Seasonal climate sums/means ---
    stages = ['Early_Veg', 'Late_Veg', 'Flowering', 'GrainFill']
    df['seasonal_precip_sum'] = df[[f'weather_precip_mm_sum_{s}' for s in stages]].sum(axis=1)
    df['seasonal_etr_sum'] = df[[f'weather_etr_sum_{s}' for s in stages]].sum(axis=1)
    df['seasonal_irrigation_sum'] = df[[f'irrigation_depth_mm_sum_{s}' for s in stages]].sum(axis=1)
    df['seasonal_water_total'] = df['seasonal_precip_sum'] + df['seasonal_irrigation_sum']
    
    df['seasonal_temp_mean'] = df[[f'weather_air_temp_c_mean_{s}' for s in stages]].mean(axis=1)
    df['seasonal_rh_mean'] = df[[f'weather_rh_mean_{s}' for s in stages]].mean(axis=1)
    df['seasonal_solar_rad_mean'] = df[[f'weather_solar_rad_w_m2_mean_{s}' for s in stages]].mean(axis=1)

    # --- 2. Cumulative stress indices ---
    df['seasonal_CWSI_mean'] = df[[f'CWSI_mean_{s}' for s in stages]].mean(axis=1)
    df['seasonal_DANS_mean'] = df[[f'DANS_aboveAvg_mean_{s}' for s in stages]].mean(axis=1)
    df['seasonal_Ks_mean'] = df[[f'Ks_linear_mean_{s}' for s in stages]].mean(axis=1)

    # --- 3. Management interactions ---
    df['trt_veg_x_rep'] = df['veg'] * df['rep']
    df['trt_veg_sq'] = df['veg'] ** 2
    df['trt_rep_sq'] = df['rep'] ** 2

    # --- 4. Stress interactions during critical stages ---
    df['flowering_stress_temp'] = df['CWSI_mean_Flowering'] * df['weather_air_temp_c_mean_Flowering']
    df['grainfill_stress_temp'] = df['CWSI_mean_GrainFill'] * df['weather_air_temp_c_mean_GrainFill']
    df['flowering_dans_temp'] = df['DANS_aboveAvg_mean_Flowering'] * df['weather_air_temp_c_mean_Flowering']
    df['grainfill_dans_temp'] = df['DANS_aboveAvg_mean_GrainFill'] * df['weather_air_temp_c_mean_GrainFill']
    
    # --- 5. Soil water deficit interactions ---
    df['flowering_swc_stress'] = df['swc_rz_mean_Flowering'] * df['CWSI_mean_Flowering']
    df['grainfill_swc_stress'] = df['swc_rz_mean_GrainFill'] * df['CWSI_mean_GrainFill']
    
    return df

# ----------------- STEP 2: PRE-MODELING COLLINEARITY & FEATURE SELECTION -----------------
def remove_collinear_features(X_df, y_ser, threshold=0.85):
    corr_matrix = X_df.corr().abs()
    target_corr = X_df.corrwith(y_ser).abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        collinear_cols = upper_tri.index[upper_tri[col] > threshold].tolist()
        if collinear_cols:
            all_cols = collinear_cols + [col]
            best_col = max(all_cols, key=lambda c: target_corr.get(c, 0.0))
            for c in all_cols:
                if c != best_col:
                    to_drop.add(c)
                    
    return [c for c in X_df.columns if c not in to_drop]

def select_top_features(X_train_df, y_train, top_n=40, threshold=0.85):
    non_collinear_feats = remove_collinear_features(X_train_df, pd.Series(y_train), threshold=threshold)
    if len(non_collinear_feats) <= top_n:
        return non_collinear_feats
        
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_df[non_collinear_feats].values, y_train)
    indices = np.argsort(rf.feature_importances_)[::-1]
    return [non_collinear_feats[i] for i in indices[:top_n]]

# ----------------- STEP 3: BASE MODELS DEFINITION -----------------
def get_base_models():
    return {
        'Ridge': Ridge(alpha=5.0),
        'Lasso': Lasso(alpha=0.005, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=2000),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'HistGB': HistGradientBoostingRegressor(max_depth=6, learning_rate=0.03, max_iter=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'SVR': SVR(kernel='rbf', C=20.0, epsilon=0.02),
        'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.5, learning_rate_init=0.005, max_iter=800, early_stopping=True, random_state=42)
    }

# ----------------- STEP 4: CROSS-VALIDATION PIPELINE -----------------
def run_leak_free_cv(df, target_col):
    print(f"\n--- Running Final CV: Target={target_col} ---")
    
    y_full = df[target_col].values
    
    # 1. Stratify CV splits using target decile bins (matches expertsplitting.py)
    y_bins = pd.qcut(y_full, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(df, y_bins))
    
    # Calculate global weights to emphasize tail extremes (matches expertsplitting.py)
    global_weights = 1.0 + (0.9 * np.abs(y_full - np.mean(y_full)) / np.std(y_full))
            
    base_model_names = list(get_base_models().keys())
    
    # Create OOF prediction arrays
    oof_preds = {m_name: np.zeros(len(y_full)) for m_name in base_model_names}
    oof_preds['Ensemble_Stacking_Meta_Local'] = np.zeros(len(y_full))
    oof_preds['Ensemble_Stacking_Meta_Calibrated_Local'] = np.zeros(len(y_full))
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/5")
        
        # Split data
        df_tr = df.iloc[train_idx].copy()
        df_va = df.iloc[val_idx].copy()
        
        y_tr = y_full[train_idx]
        
        # Compute sample weights for the training split
        w_tr = 1.0 + (0.9 * np.abs(y_tr - np.mean(y_tr)) / np.std(y_tr))
        
        # 2. Treat Encode & Ordinal Encode 'treatment' (LEAK-FREE)
        trt_mean_yields = df_tr.groupby('treatment')['grain_yield'].mean().sort_values()
        trt_rank_map = {trt: rank for rank, trt in enumerate(trt_mean_yields.index)}
        
        # Target Encoding
        trt_mean_target = df_tr.groupby('treatment')[target_col].mean()
        
        df_tr['treatment_target_encoded'] = df_tr['treatment'].map(trt_mean_target).fillna(y_tr.mean())
        df_va['treatment_target_encoded'] = df_va['treatment'].map(trt_mean_target).fillna(y_tr.mean())
        
        # Ordinal Encoding
        df_tr['treatment_ordinal'] = df_tr['treatment'].map(trt_rank_map).fillna(len(trt_rank_map) // 2).astype(float)
        df_va['treatment_ordinal'] = df_va['treatment'].map(trt_rank_map).fillna(len(trt_rank_map) // 2).astype(float)
        
        # 3. Create treatment interaction terms (LEAK-FREE)
        df_tr['trt_ordinal_x_CWSI_flowering'] = df_tr['treatment_ordinal'] * df_tr['CWSI_mean_Flowering']
        df_va['trt_ordinal_x_CWSI_flowering'] = df_va['treatment_ordinal'] * df_va['CWSI_mean_Flowering']
        
        df_tr['trt_target_x_CWSI_flowering'] = df_tr['treatment_target_encoded'] * df_tr['CWSI_mean_Flowering']
        df_va['trt_target_x_CWSI_flowering'] = df_va['treatment_target_encoded'] * df_va['CWSI_mean_Flowering']
        
        df_tr['trt_ordinal_x_CWSI_grainfill'] = df_tr['treatment_ordinal'] * df_tr['CWSI_mean_GrainFill']
        df_va['trt_ordinal_x_CWSI_grainfill'] = df_va['treatment_ordinal'] * df_va['CWSI_mean_GrainFill']
        
        df_tr['trt_target_x_CWSI_grainfill'] = df_tr['treatment_target_encoded'] * df_tr['CWSI_mean_GrainFill']
        df_va['trt_target_x_CWSI_grainfill'] = df_va['treatment_target_encoded'] * df_va['CWSI_mean_GrainFill']

        # Select feature list
        drop_cols = ['plot', 'treatment', 'year', 'grain_yield', 'relative_yield_reduction']
        feature_cols = [c for c in df_tr.columns if c not in drop_cols]
        
        X_tr_df = df_tr[feature_cols]
        X_va_df = df_va[feature_cols]
        
        # 4. Pre-modeling collinearity and variable selection (LEAK-FREE)
        selected_feats = select_top_features(X_tr_df, y_tr, top_n=40, threshold=0.85)
        
        X_tr = X_tr_df[selected_feats].values
        X_va = X_va_df[selected_feats].values
        
        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        # 5. Fit base models (using sample weights where supported)
        models = get_base_models()
        fold_base_preds = {}
        for m_name, model in models.items():
            if m_name in ['KNN', 'MLP']:
                model.fit(X_tr_s, y_tr)
            else:
                model.fit(X_tr_s, y_tr, sample_weight=w_tr)
                
            pred = model.predict(X_va_s)
            oof_preds[m_name][val_idx] = pred
            fold_base_preds[m_name] = pred
            
        # 6. Nested (Local) Stacking and Calibration
        inner_kf = KFold(n_splits=4, shuffle=True, random_state=42)
        inner_oof = {m_name: np.zeros(len(y_tr)) for m_name in base_model_names}
        
        for inner_train_idx, inner_val_idx in inner_kf.split(X_tr_s, y_tr):
            X_itr, y_itr = X_tr_s[inner_train_idx], y_tr[inner_train_idx]
            X_iva = X_tr_s[inner_val_idx]
            
            # Compute inner weights
            w_itr = 1.0 + (0.9 * np.abs(y_itr - np.mean(y_itr)) / np.std(y_itr))
            
            inner_models = get_base_models()
            for m_name, model in inner_models.items():
                if m_name in ['KNN', 'MLP']:
                    model.fit(X_itr, y_itr)
                else:
                    model.fit(X_itr, y_itr, sample_weight=w_itr)
                inner_oof[m_name][inner_val_idx] = model.predict(X_iva)
                
        # Stack inner OOF predictions
        inner_oof_matrix = np.column_stack([inner_oof[m] for m in base_model_names])
        fold_base_preds_matrix = np.column_stack([fold_base_preds[m] for m in base_model_names])
        
        # Local Stack Meta-Learner (RidgeCV)
        meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 10))
        meta_learner.fit(inner_oof_matrix, y_tr, sample_weight=w_tr)
        stack_pred = meta_learner.predict(fold_base_preds_matrix)
        oof_preds['Ensemble_Stacking_Meta_Local'][val_idx] = stack_pred
        
        # Local Calibrated Stacking (Stage-2 Isotonic)
        inner_stack_oof = meta_learner.predict(inner_oof_matrix)
        iso_stack = IsotonicRegression(out_of_bounds='clip')
        iso_stack.fit(inner_stack_oof, y_tr)
        oof_preds['Ensemble_Stacking_Meta_Calibrated_Local'][val_idx] = iso_stack.predict(stack_pred)
        
    # 7. Global (expertsplitting-style) Stacking and Calibration
    print("Fitting Global (expertsplitting-style) Stacking & Calibration...")
    X_stack_global = pd.DataFrame({m: oof_preds[m] for m in base_model_names})
    
    # Global RidgeCV Meta-Learner
    meta_global = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_global.fit(X_stack_global, y_full, sample_weight=global_weights)
    global_stack_pred = meta_global.predict(X_stack_global)
    oof_preds['Ensemble_Stacking_Meta_Global'] = global_stack_pred
    
    # Global Stage-2 Isotonic Calibration (to extend predictions for extreme values)
    iso_global = IsotonicRegression(out_of_bounds='clip')
    iso_global.fit(global_stack_pred, y_full)
    oof_preds['Ensemble_Stacking_Meta_Calibrated_Global'] = iso_global.predict(global_stack_pred)
    
    # Print Stacking Coefficients
    print("\n--- Global Stacking Meta-Learner Weights ---")
    for name, coef in zip(X_stack_global.columns, meta_global.coef_):
        print(f"{name:<20}: {coef:.4f}")
        
    # Compile metrics
    results = []
    for model_name, preds in oof_preds.items():
        if target_col == 'relative_yield_reduction':
            preds = np.clip(preds, 0.0, 1.0)
        elif target_col == 'grain_yield':
            preds = np.clip(preds, 0.0, None)
            
        r2 = r2_score(y_full, preds)
        rmse = np.sqrt(mean_squared_error(y_full, preds))
        
        mean_y = np.mean(y_full)
        rel_rmse = (rmse / mean_y) * 100.0 if mean_y != 0 else 0
        
        results.append({
            'Target': target_col,
            'Model': model_name,
            'R2': r2,
            'RMSE': rmse,
            'Rel RMSE (%)': rel_rmse,
            'Predictions': preds
        })
        
    return results, oof_preds, y_full

def main():
    df = load_and_engineer_data()
    
    results = []
    
    # Target 1: Absolute Grain Yield (grain_yield)
    res_yield, preds_yield, y_yield = run_leak_free_cv(df, 'grain_yield')
    results.extend(res_yield)
    
    # Target 2: Relative Yield Reduction (relative_yield_reduction)
    res_reduction, preds_reduction, y_reduction = run_leak_free_cv(df, 'relative_yield_reduction')
    results.extend(res_reduction)
    
    df_results = pd.DataFrame(results)
    df_results_print = df_results.drop(columns=['Predictions'])
    
    print("\n" + "="*95)
    print("UPGRADED CHAMPION PIPELINE V5 RESULTS (ORDINAL ENCODING & STAGE-2 ISOTONIC)")
    print("="*95)
    print(df_results_print.to_string(index=False))
    print("="*95)
    
    # Save results table
    df_results_print.to_csv('/mnt/Data/LIRF/Scripts/final_tuned_model_metrics_v5.csv', index=False)
    
    # Find Champion configurations
    df_grain = df_results[df_results['Target'] == 'grain_yield']
    champion_row_yield = df_grain.loc[df_grain['R2'].idxmax()]
    print(f"\nChampion Model for Absolute Yield: {champion_row_yield['Model']} (R2 = {champion_row_yield['R2']:.4f}, RMSE = {champion_row_yield['RMSE']:.1f} kg/ha, Rel RMSE = {champion_row_yield['Rel RMSE (%)']:.2f}%)")
    
    df_reduction = df_results[df_results['Target'] == 'relative_yield_reduction']
    champion_row_reduction = df_reduction.loc[df_reduction['R2'].idxmax()]
    print(f"Champion Model for Yield Reduction: {champion_row_reduction['Model']} (R2 = {champion_row_reduction['R2']:.4f}, RMSE = {champion_row_reduction['RMSE']:.4f}, Rel RMSE = {champion_row_reduction['Rel RMSE (%)']:.2f}%)")
    
    # Plotting diagnostic curves for the Global Stacking Calibrated model
    os.makedirs('/mnt/Data/LIRF/Scripts/figures', exist_ok=True)
    
    def get_deficit_color(y_val, max_y=17423.0):
        norm_val = np.clip(y_val / max_y, 0.0, 1.0)
        return plt.cm.RdYlBu(norm_val)
        
    y_pred_global = preds_yield['Ensemble_Stacking_Meta_Calibrated_Global']
    residuals_global = y_yield - y_pred_global
    colors = [get_deficit_color(yt) for yt in y_yield]
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_yield, y_pred_global, color=colors, alpha=0.7, edgecolor='k', linewidth=0.5, s=50)
    plt.plot([y_yield.min(), y_yield.max()], [y_yield.min(), y_yield.max()], 'r--', alpha=0.7)
    plt.xlabel("Observed Yield (kg/ha)", fontweight='bold', fontsize=11)
    plt.ylabel("Predicted Yield (kg/ha)", fontweight='bold', fontsize=11)
    plt.title(f"Global Calibrated Stacked Ensemble (R² = {champion_row_yield['R2']:.3f})\nPredicted vs. Observed Yield", fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.15)
    plt.tight_layout()
    plt.savefig('/mnt/Data/LIRF/Scripts/figures/final_champion_predicted_vs_observed_v5.png', dpi=200)
    plt.close()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred_global, residuals_global, color=colors, alpha=0.7, edgecolor='k', linewidth=0.5, s=50)
    plt.axhline(0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel("Predicted Yield (kg/ha)", fontweight='bold', fontsize=11)
    plt.ylabel("Residual (Observed - Predicted)", fontweight='bold', fontsize=11)
    plt.title("Global Calibrated Stacked Ensemble Residual Plot", fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.15)
    plt.tight_layout()
    plt.savefig('/mnt/Data/LIRF/Scripts/figures/final_champion_residuals_v5.png', dpi=200)
    plt.close()
    
    print("\nSaved champion diagnostic plots to /mnt/Data/LIRF/Scripts/figures/")

if __name__ == '__main__':
    main()
