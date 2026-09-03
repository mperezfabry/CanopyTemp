import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def parse_treatment(t):
    if '/' in str(t):
        p = t.split('/')
        return float(p[0]), float(p[1])
    try:
        val = float(t)
        return val, val
    except:
        return None, None

def get_raw_treatment(row):
    raw_trt = row.get('treatment', '')
    if pd.isna(raw_trt) or str(raw_trt).strip().upper() in ['NAN', 'NONE', '', 'NULL', 'UNKNOWN', 'NP.NAN']:
        raw_trt = row.get('trt_code', row.get('treatment_code', ''))
    return str(raw_trt).strip()

def clean_treatment_string(row):
    raw_s = get_raw_treatment(row)
    if not raw_s or raw_s.upper() in ['NAN', 'NONE', 'NULL', 'UNKNOWN']:
        return None
        
    s = raw_s.upper().replace(' ', '')
    if s in ['100', '100/100', '1.0', '1.0/1.0', 'FI', 'FULL', 'FULLIRRIGATION']:
        return '100/100'
        
    if '/' in s:
        parts = s.split('/')
        try:
            p0 = float(parts[0])
            p1 = float(parts[1])
            v = p0 * 100.0 if p0 <= 1.0 else p0
            r = p1 * 100.0 if p1 <= 1.0 else p1
            return f"{int(round(v))}/{int(round(r))}"
        except:
            return s
            
    try:
        val = float(s)
        v = val * 100.0 if val <= 1.0 else val
        return f"{int(round(v))}/{int(round(v))}"
    except:
        return s

def classify_strategy(row):
    trt = str(row.get('treatment', '')).strip().upper()
    if any(k in trt for k in ['_LO', '_HI', '_FI', 'DANS', 'CWSI', 'WISE', 'TCR', 'DACT', 'SWB']):
        return None

    trt_clean = row.get('treatment_clean')
    if not trt_clean or str(trt_clean).upper() in ['NAN', 'NONE', 'NULL']:
        return None
        
    if trt_clean in ['100/100', '100']:
        return 'Full Irrigation (Control)'
        
    if '/' in str(trt_clean):
        parts = str(trt_clean).split('/')
        try:
            v = float(parts[0])
            r = float(parts[1])
            if v >= 95.0 and r >= 95.0:
                return 'Full Irrigation (Control)'
            elif v > r:
                return 'Reproductive Deficit (Early watering, late stress)'
            elif v < r:
                return 'Vegetative Deficit (Early stress, late watering)'
            elif v == r:
                return 'Dual/Continuous Deficit (Season-long stress)'
        except:
            pass
            
    return 'Dual/Continuous Deficit (Season-long stress)'

def main():
    excel_path = '/mnt/Data/LIRF/output_021926.xlsx'
    csv_path = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.csv'
    parquet_path = '/mnt/Data/LIRF/Scripts/master_data_ml_plot_level.parquet'
    
    if os.path.exists(excel_path):
        print(f"Loading yield data directly from {excel_path} (Sheet: 'Annual Plot Data')...")
        df_raw = pd.read_excel(excel_path, sheet_name='Annual Plot Data')
    elif os.path.exists(csv_path):
        print(f"Loading master yield dataset directly from {csv_path}...")
        df_raw = pd.read_csv(csv_path)
    else:
        print(f"Loading master yield dataset directly from {parquet_path}...")
        df_raw = pd.read_parquet(parquet_path)
    
    # Standardize column names
    df_raw = df_raw.rename(columns={c: str(c).strip() for c in df_raw.columns})
    
    y_col = [c for c in df_raw.columns if 'year' in c.lower()][0]
    g_col = [c for c in df_raw.columns if 'yield' in c.lower()][0]
    t_col = [c for c in df_raw.columns if 'treatment' in c.lower()][0]
    
    df_raw[y_col] = pd.to_numeric(df_raw[y_col], errors='coerce')
    
    # Filter for valid rows
    if 'crop' in [c.lower() for c in df_raw.columns]:
        c_col = [c for c in df_raw.columns if 'crop' in c.lower()][0]
        df = df_raw[(df_raw[y_col].notna()) & (df_raw[c_col].astype(str).str.lower().str.contains('corn'))].copy()
    else:
        df = df_raw[df_raw[y_col].notna()].copy()
        
    df['year'] = df[y_col].astype(int)
    df = df[df[g_col].notna()].copy()
    df['grain_yield'] = df[g_col]
    df['treatment'] = df[t_col]
    
    # Clean and group treatment strings (single numbers e.g. "40" -> "40/40", fallback to trt_code if blank)
    df['treatment_clean'] = df.apply(clean_treatment_string, axis=1)
    df = df[df['treatment_clean'].notna()].copy()

    min_yr = int(df['year'].min())
    max_yr = int(df['year'].max())
    print(f"Loaded unified dataset with {len(df)} corn plot-year observations for {min_yr}–{max_yr}.")
    
    # Compute relative yield reduction per year based on max yield in that year
    if 'relative_yield_reduction' in df.columns:
        df['rel_yield_red'] = df['relative_yield_reduction']
    else:
        max_yields = df.groupby('year')['grain_yield'].transform('max')
        df['rel_yield_red'] = 1.0 - (df['grain_yield'] / max_yields)
    
    df['strategy'] = df.apply(classify_strategy, axis=1)
    df = df[df['strategy'].notna()].copy()
    
    # Configure plotting style matching swapped_treatments_comparison
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14
    })
    
    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)

    # --- Figure 1: Horizontal Bar Chart of Treatments Sorted by Yield Reduction (Styled matching irrigation_wue_bar) ---
    # Calculate group stats and drop single-observation (n=1) treatments
    grp_stats = df.groupby('treatment_clean')['rel_yield_red'].agg(['mean', 'sem', 'count']).reset_index()
    grp_stats = grp_stats[grp_stats['count'] > 1].copy()
    grp_stats['sem'] = grp_stats['sem'].fillna(0.0)
    grp_stats = grp_stats.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(grp_stats)))
    
    bars = ax.barh(grp_stats['treatment_clean'], grp_stats['mean'] * 100, 
                    xerr=grp_stats['sem'] * 100, 
                    color=colors, edgecolor='k', linewidth=0.5, capsize=3, 
                    error_kw={'ecolor': '#333333', 'lw': 1.0})
    
    max_bar_extent = max([(bar.get_width() + (grp_stats.iloc[idx]['sem'] * 100 if pd.notna(grp_stats.iloc[idx]['sem']) else 0.0)) for idx, bar in enumerate(bars)])
    ax.set_xlim(0, max(65.0, max_bar_extent * 1.25))
    
    for idx, (bar, count) in enumerate(zip(bars, grp_stats['count'])):
        width = bar.get_width()
        sem_val = grp_stats.iloc[idx]['sem'] * 100
        val = grp_stats.iloc[idx]['mean'] * 100
        ax.text(width + sem_val + 0.5, bar.get_y() + bar.get_height()/2., 
                f'{val:.1f}% (n={count})', 
                ha='left', va='center', fontsize=8.5, color='#222222', fontweight='bold')
        
    ax.set_xlabel('Relative Yield Reduction (%)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Irrigation Treatment', fontweight='bold', fontsize=11)
    ax.set_title(f'Relative Yield Reduction Across Irrigation Deficit Treatments ({min_yr}–{max_yr})', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'deficit_treatment_bar.png'), dpi=300)
    plt.close()
    print("Saved deficit_treatment_bar.png")
    
    # --- Figure 2: Boxplot of Strategies (Styled matching swapped_treatments_comparison) ---
    print("Generating Strategy Boxplot...")
    strategy_order = [
        'Full Irrigation (Control)',
        'Vegetative Deficit (Early stress, late watering)',
        'Reproductive Deficit (Early watering, late stress)'
    ]
    
    plt.figure(figsize=(8.5, 5.5))
    palette = [
        '#1a365d', # Full Irrigation - Deep blue
        '#166534', # Vegetative Deficit - Deep green
        '#ea580c'  # Reproductive Deficit - Deep orange
    ]
    
    df_box = df.copy()
    
    # Prepare data for boxplot (Only first 3 strategies plotted)
    data_to_plot = [df_box[df_box['strategy'] == strat]['rel_yield_red'].dropna().values * 100 for strat in strategy_order]
    
    # Create boxplot matching swapped_treatments_comparison style
    bp = plt.boxplot(data_to_plot, patch_artist=True, widths=0.4)
                      
    # Color and format the boxes
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    # Overlay jittered data points and mean text labels
    for i, vals in enumerate(data_to_plot):
        if len(vals) > 0:
            np.random.seed(42 + i)
            x = np.random.normal(i + 1, 0.05, size=len(vals))
            plt.scatter(x, vals, color='black', alpha=0.2, s=12)
            mean_val = np.mean(vals)
            plt.text(i + 1, mean_val + 2, f"{mean_val:.1f}%", 
                     ha='center', va='bottom', color='black', weight='bold', fontsize=8)
                   
    plt.ylabel('Relative Yield Reduction (%)', fontweight='bold')
    plt.xlabel('Irrigation Scheduling Strategy', fontweight='bold')
    plt.title(f'Corn Yield Sensitivity Across Growth-Stage Deficit Classes ({min_yr}–{max_yr})', fontweight='bold', pad=12, fontsize=12)
    
    # Format X ticks for readability
    labels = [
        f'Full Irrigation\n(Control)\n[n={len(data_to_plot[0])}]',
        f'Vegetative Deficit\n(Early Stress / Late Water)\n[n={len(data_to_plot[1])}]',
        f'Reproductive Deficit\n(Early Water / Late Stress)\n[n={len(data_to_plot[2])}]'
    ]
    plt.xticks(range(1, 4), labels, fontsize=9)
    plt.ylim(-15, 105)
    plt.grid(True, linestyle='--', alpha=0.15, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'deficit_strategy_boxplot.png'), dpi=300)
    plt.close()
    print("Saved deficit_strategy_boxplot.png (3 boxplots only)")
    
    # --- Figure 3: Yearly Maximum Yields (Vertical Bar Chart starting at y=0) ---
    print("Generating Yearly Maximum Yields Bar Plot...")
    yearly_max = df.groupby('year')['grain_yield'].max().reset_index()
    yearly_max['year'] = yearly_max['year'].astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(yearly_max['year'], yearly_max['grain_yield'], color='#1f77b4', edgecolor='k', linewidth=0.6, alpha=0.85)
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=10)
    ax.set_ylabel('Maximum Grain Yield (kg/ha)', fontweight='bold', fontsize=10)
    ax.set_title(f'Inter-annual Variation in Maximum Corn Yield (LIRF {min_yr}–{max_yr})', fontweight='bold', pad=12, fontsize=12)
    ax.set_ylim(bottom=0, top=yearly_max['grain_yield'].max() * 1.15)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_xticks(yearly_max['year'])
    ax.set_xticklabels(yearly_max['year'], rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 150, f'{height:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#333333')
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'yearly_maximum_yields.png'), dpi=300)
    plt.close()
    print("Saved yearly_maximum_yields.png as bar plot starting at y=0")
    
    # --- Print descriptive statistics summary ---
    print("\n--- DESCRIPTIVE STATISTICS TABLE ---")
    strat_summary = df.groupby('strategy')['rel_yield_red'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(strat_summary)
    
    strat_summary.to_csv('/mnt/Data/LIRF/Scripts/strategy_summary_stats.csv')
    print("\nDescriptive statistics written to strategy_summary_stats.csv")

if __name__ == '__main__':
    main()
