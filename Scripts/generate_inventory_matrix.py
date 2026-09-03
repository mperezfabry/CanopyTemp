import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=== GENERATING CUSTOM STYLED INVENTORY MATRIX CHART & TABLE ===")
    
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # Treatment specs ordering:
    # 1. Sensor/Model treatments grouped at top (F56, EBAL, RSRZ, DANS, SWB, CWSI, DACT, WISE, TCR)
    # 2. Fixed ET target deficit section starting with 100/100 right below TCR
    # 3. Stage deficits and earlier fixed deficits
    treatment_specs = [
        ("F56 (FI)", [2022, 2023, 2024]),
        ("EBAL (LO)", [2020, 2021, 2022, 2023, 2024]),
        ("RSRZ (HI, LO)", [2020, 2021, 2022, 2023, 2024]),
        ("DANS (HI, LO)", [2019, 2020, 2021, 2022, 2023, 2024]),
        ("SWB (FI, LO)", [2019, 2020, 2021, 2022, 2023, 2024]),
        ("CWSI (HI, LO), CWSIT, CWSIB", [2017, 2019, 2021]),
        ("DACT (HI, LO)", [2017]),
        ("WISE (HI, LO)", [2017]),
        ("TCR (HI, LO)", [2017]),
        
        ("100/100", [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]),
        ("80/80", [2012, 2013, 2014, 2015, 2016, 2017, 2018]),
        ("70/70", [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]),
        ("40/40", [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]),
        
        ("65/65", [2012, 2013, 2014, 2015, 2016]),
        ("50/50", [2012, 2013, 2014, 2015, 2016]),
        ("100/50", [2012, 2013, 2014, 2015, 2016]),
        ("80/65", [2012, 2013, 2014, 2015, 2016]),
        ("80/40", [2012, 2013, 2014, 2015, 2016]),
        ("65/80", [2012, 2013, 2014, 2015, 2016]),
        ("65/50", [2012, 2013, 2014, 2015, 2016]),
        ("65/40", [2012, 2013, 2014, 2015, 2016]),
        ("80/50", [2012, 2013]),
        ("85/85", [2008, 2009, 2010, 2011]),
        ("75/75", [2008, 2009, 2010, 2011]),
        ("55/55", [2008, 2009, 2010, 2011])
    ]
    
    # 1. Build DataFrame
    matrix_data = []
    for trt, active_yrs in treatment_specs:
        row = {'Treatment': trt}
        for y in years:
            row[str(y)] = 'X' if y in active_yrs else ''
        matrix_data.append(row)
        
    df_matrix = pd.DataFrame(matrix_data)
    
    # 2. Write Markdown Tables
    md_lines = ["# LIRF Experimental Plot Treatment Inventory Matrix (2008–2024)", ""]
    md_lines.append("Treatments ordered with named sensor/model treatments first, followed by fixed ET targets starting with 100/100 below TCR:")
    md_lines.append("")
    md_lines.append(df_matrix.to_markdown(index=False))
    md_content = "\n".join(md_lines)
    
    out_md1 = '/mnt/Data/LIRF/Scripts/plot_treatments_matrix.md'
    out_md2 = '/mnt/Data/LIRF/Scripts/plot_treatments_2col.md'
    
    with open(out_md1, 'w') as f:
        f.write(md_content)
    with open(out_md2, 'w') as f:
        f.write(md_content)
        
    print(f"Saved updated matrix Markdown table to {out_md1} and {out_md2}")
    
    # 3. Create Matrix Figure
    fig_dir = '/mnt/Data/LIRF/Scripts/figures'
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'plot_year_inventory_matrix.png')
    
    n_rows = len(treatment_specs)
    n_cols = len(years)
    
    grid = np.zeros((n_rows, n_cols))
    
    fig, ax = plt.subplots(figsize=(10.0, 10.5))
    
    cmap = matplotlib.colors.ListedColormap(['#ffffff'])
    ax.imshow(grid, cmap=cmap, aspect='auto', extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])
    
    # Add Black 'X' inside active treatment cells
    for i, (trt, active_yrs) in enumerate(treatment_specs):
        for j, y in enumerate(years):
            if y in active_yrs:
                ax.text(j, i, 'X', ha='center', va='center',
                        color='black', fontweight='bold', fontsize=11)
                        
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    
    # Year labels ONLY on top, rotated 45 degrees
    ax.set_xticklabels([str(y) for y in years], fontweight='bold', fontsize=10.5, rotation=45, ha='left', rotation_mode='anchor')
    ax.set_yticklabels([t[0] for t in treatment_specs], fontweight='bold', fontsize=9.5)
    
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_ticks_position('top')
    
    # Gridlines: Transparent horizontal gridlines, Darker vertical gridlines
    for y_val in np.arange(-0.5, n_rows, 1):
        ax.axhline(y_val, color='#cbd5e0', linestyle='-', linewidth=0.7, alpha=0.45)
    for x_val in np.arange(-0.5, n_cols, 1):
        ax.axvline(x_val, color='#1a202c', linestyle='-', linewidth=1.3, alpha=0.85)
    # Right border line
    ax.axvline(n_cols - 0.5, color='#1a202c', linestyle='-', linewidth=1.3, alpha=0.85)
    ax.axhline(n_rows - 0.5, color='#cbd5e0', linestyle='-', linewidth=0.7, alpha=0.45)
    
    plt.title('LIRF Experimental Plot Treatment Inventory Matrix (2008–2024)',
              fontweight='bold', fontsize=13, pad=38)
              
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Saved custom matrix chart figure to {fig_path}")

if __name__ == '__main__':
    main()
