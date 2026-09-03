import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def build_5panel_model_comparison(csv_path, target_title, save_path, y_metric='RMSE'):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_parquet(csv_path) if csv_path.endswith('.parquet') else pd.read_csv(csv_path)
    
    # Sort stages in chronological agronomic order
    stage_order = ['V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
                   'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'VT', 
                   'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    
    if 'Stage' not in df.columns and 'stage' in df.columns:
        df['Stage'] = df['stage']
        
    df['Stage'] = pd.Categorical(df['Stage'], categories=stage_order, ordered=True)
    df = df.sort_values('Stage')
    
    models = ['ElasticNet', 'ExtraTrees', 'Gradient_Boost', '1D_CNN', 'MLP']
    model_titles = ['Elastic Net', 'ExtraTrees', 'Gradient Boost', '1D-CNN', 'MLP']
    
    metrics = sorted(df['Metric'].unique()) if 'Metric' in df.columns else ['Default']
    palette = sns.color_palette("tab10", n_colors=len(metrics))
    color_map = dict(zip(metrics, palette))
    
    fig, axs = plt.subplots(1, 5, figsize=(24, 5.2), sharey=True, dpi=300)
    fig.suptitle(f"LOYO Season Simulation Stage Progression by Model Class ({target_title})", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (m, m_title) in enumerate(zip(models, model_titles)):
        ax = axs[idx]
        col_name = f"{y_metric}_{m}"
        
        if col_name not in df.columns:
            # Try alternate col naming
            alt_cols = [c for c in df.columns if y_metric.lower() in c.lower() and m.lower().replace('_', '') in c.lower().replace('_', '')]
            if alt_cols:
                col_name = alt_cols[0]
            else:
                ax.set_title(f"{m_title} (N/A)", fontweight='bold', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.4)
                continue
            
        for metric_name in metrics:
            if 'Metric' in df.columns:
                df_sub = df[df['Metric'] == metric_name]
            else:
                df_sub = df
            ax.plot(df_sub['Stage'], df_sub[col_name], marker='o', ms=4, 
                    label=metric_name, color=color_map[metric_name], linewidth=1.8)
            
        ax.set_title(m_title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Growth Stage", fontweight='bold', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        if idx == 0:
            y_label = "RMSE (kg/ha)" if y_metric == 'RMSE' else "$R^2$"
            ax.set_ylabel(y_label, fontweight='bold', fontsize=11)
            
        if idx == 4:
            ax.legend(title="Feature Set / Metric", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated 5-panel figure at: {save_path}")

def main():
    print("=" * 80)
    print("GENERATING 5-PANEL HORIZONTAL MODEL COMPARISON MANUSCRIPT FIGURES")
    print("=" * 80)
    
    # 1. Grain Yield RMSE
    build_5panel_model_comparison(
        '/mnt/Data/LIRF/Scripts/season_simulation_results_loyo.csv',
        'Grain Yield RMSE',
        '/mnt/Data/LIRF/Scripts/figures/season_simulation_5panel_yield_rmse.png',
        y_metric='RMSE'
    )
    
    # 2. Grain Yield R2
    build_5panel_model_comparison(
        '/mnt/Data/LIRF/Scripts/season_simulation_results_loyo.csv',
        'Grain Yield R2',
        '/mnt/Data/LIRF/Scripts/figures/season_simulation_5panel_yield_r2.png',
        y_metric='R2'
    )

    # 3. Relative Yield Loss RMSE
    build_5panel_model_comparison(
        '/mnt/Data/LIRF/Scripts/season_simulation_results_yield_loss_loyo.csv',
        'Relative Yield Loss RMSE',
        '/mnt/Data/LIRF/Scripts/figures/season_simulation_5panel_yield_loss_rmse.png',
        y_metric='RMSE'
    )

if __name__ == '__main__':
    main()
