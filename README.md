# Machine Learning & Thermal Stress Modeling for Deficit-Irrigated Maize

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

This repository is the official code accompaniment for the manuscript:

> **"Multi-Modal Thermal Stress & Microclimate Machine Learning for Season-Long Yield Deficit Simulation in Maize"** *(Currently under peer review)*.

---

## 📢 Data Availability Statement

> **Note:** The underlying high-resolution microclimate, infrared canopy temperature (IRT), soil water content, and crop yield datasets spanning 16 experimental field seasons are **not yet publicly included in this preliminary repository**.
>
> The complete dataset is scheduled for public release and hosting on **[AgriCommons](https://agricommons.org/)** upon final acceptance of the manuscript. Once data is published, standard download paths will be provided to automatically run the pipeline.

---

## 📁 Repository Structure

```text
LIRF/
├── dataset/                        # Data directory (populated upon AgriCommons release)
│   ├── Long/                       # Native parquet time series
│   └── weather/                    # High-resolution CoAgMet weather data
├── Scripts/                        # Core Python scripts for modeling & figures
│   ├── build_full_coagmet_native_dataset.py     # Data ingestion & 5-min weather aggregation
│   ├── prepare_master_ml_data.py                # Plot-level feature engineering
│   ├── generate_inventory_matrix.py             # Inventory matrix (Fig 1 / Table 1)
│   ├── generate_descriptive_stats.py            # Yield boxplots & stats (Fig 2)
│   ├── generate_applied_water_response_curve.py # Monotonic GAM water curves (Fig 3)
│   ├── generate_unified_4panel_water_yield_figure.py # Unified GAM 4-panel curve
│   ├── run_deficit_plateau_polynomial_analysis.py    # Deficit plateau GAM curve (Fig 4)
│   ├── run_enhanced_stage_regressions.py        # Stage-by-stage OLS/ElasticNet (Fig 5)
│   ├── run_season_simulation_loyo.py            # 25-stage LOYO ML simulation (V3 -> R6)
│   ├── run_deep_learning_season_simulation_loyo.py # 3D sequence PyTorch DL simulation
│   ├── plot_season_simulation_results.py        # 5-panel stage progression plots (Fig 6/7)
│   ├── plot_season_simulation_residuals.py      # 30-subplot publication residual grid (Fig 8)
│   ├── generate_shap_feature_importance.py      # SHAP feature importance plot (Fig 9)
│   ├── figures/                    # Exported high-resolution publication PNGs
│   └── legacy/                     # Archival exploratory & diagnostic code
└── README.md                       # Project documentation
```

---

## 💻 Environment Setup & Installation

Clone the repository and install required dependencies using Python 3.10+ and standard PyTorch stack:

```bash
git clone https://github.com/mperezfabry/CanopyTemp.git
cd CanopyTemp

# Create conda environment
conda create -n lirf_env python=3.11 -y
conda activate lirf_env

# Install dependencies
pip install numpy pandas scikit-learn pytorch torchvision matplotlib seaborn shap statsmodels pygam
```

---

## 🚀 Execution & Pipeline Guide

Once dataset files are placed into the `/dataset/` directory, execute the full end-to-end analytical pipeline to reproduce all manuscript tables, ML model cross-validations, and figures:

### Step 1: Data Processing & Feature Engineering
```bash
# 1. Ingest high-resolution weather & resample 5-min precipitation
python3 Scripts/build_full_coagmet_native_dataset.py

# 2. Engineer plot-level thermal stress (CWSI, DANS), SWC, and growth stage features
python3 Scripts/prepare_master_ml_data.py
```

### Step 2: Descriptive Statistics & Water Response Curves
```bash
# 3. Generate experimental inventory matrix & yield boxplots (Figures 1 & 2)
python3 Scripts/generate_inventory_matrix.py
python3 Scripts/generate_descriptive_stats.py

# 4. Fit strictly monotonic GAM water & ET response curves (Figures 3 & 4)
python3 Scripts/generate_applied_water_response_curve.py
python3 Scripts/run_deficit_plateau_polynomial_analysis.py
```

### Step 3: Stage Regressions & Season Simulations
```bash
# 5. Run stage-by-stage OLS & ElasticNet regressions (Figure 5)
python3 Scripts/run_enhanced_stage_regressions.py

# 6. Run 25-stage Leave-One-Year-Out (LOYO) cross-validation ML simulation (V3 -> R6)
python3 Scripts/run_season_simulation_loyo.py

# 7. Run 3D-sequence PyTorch Deep Learning LOYO cross-validation simulation
python3 Scripts/run_deep_learning_season_simulation_loyo.py
```

### Step 4: Figure Generation & Model Diagnostics
```bash
# 8. Render 5-panel stage progression model comparison line plots (Figures 6 & 7)
python3 Scripts/plot_season_simulation_results.py

# 9. Render 30-subplot publication residual grid across models & metrics (Figure 8)
python3 Scripts/plot_season_simulation_residuals.py

# 10. Generate SHAP feature importance plot (Figure 9)
python3 Scripts/generate_shap_feature_importance.py
```

---

## 📊 Manuscript Figures & Script Mapping

| Manuscript Figure | Description | Generating Script | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Figure 1** | Experimental Season Inventory & Treatment Matrix | `generate_inventory_matrix.py` | `plot_year_inventory_matrix.png` |
| **Figure 2** | Deficit Strategy Yield Distributions & Max Yields | `generate_descriptive_stats.py` | `deficit_strategy_boxplot.png` |
| **Figure 3** | Monotonic GAM Water Response Curves | `generate_applied_water_response_curve.py` | `total_applied_water_gam_spline.png` |
| **Figure 4** | Deficit Plateau Response Curve | `run_deficit_plateau_polynomial_analysis.py` | `low_deficit_plateau_curve.png` |
| **Figure 5** | Stage-by-Stage Stress Metrics Regression | `run_enhanced_stage_regressions.py` | `enhanced_stage_regressions_metrics.csv` |
| **Figure 6** | 5-Panel Stage Progression RMSE ($V_3 \to R_6$) | `plot_season_simulation_results.py` | `season_simulation_5panel_yield_loss_rmse.png` |
| **Figure 7** | 5-Panel Stage Progression $R^2$ ($V_3 \to R_6$) | `plot_season_simulation_results.py` | `season_simulation_5panel_yield_r2.png` |
| **Figure 8** | 30-Subplot Publication Residual Grid | `plot_season_simulation_residuals.py` | `season_simulation_loyo_residuals_all_years.png` |
| **Figure 9** | SHAP Feature Importance Summary | `generate_shap_feature_importance.py` | `shap_feature_importance_summary.png` |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
