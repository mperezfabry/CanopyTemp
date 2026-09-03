import os
import shutil

def main():
    scripts_dir = '/mnt/Data/LIRF/Scripts'
    legacy_dir = os.path.join(scripts_dir, 'legacy')
    os.makedirs(legacy_dir, exist_ok=True)

    # Core official paper scripts (champion ensemble dropped)
    core_scripts = {
        'build_full_coagmet_native_dataset.py',
        'prepare_master_ml_data.py',
        'generate_inventory_matrix.py',
        'generate_descriptive_stats.py',
        'generate_applied_water_response_curve.py',
        'generate_unified_4panel_water_yield_figure.py',
        'run_deficit_plateau_polynomial_analysis.py',
        'run_enhanced_stage_regressions.py',
        'run_season_simulation_loyo.py',
        'run_deep_learning_season_simulation_loyo.py',
        'plot_season_simulation_results.py',
        'plot_season_simulation_residuals.py',
        'generate_shap_feature_importance.py'
    }

    all_files = os.listdir(scripts_dir)
    moved_count = 0

    for item in all_files:
        full_path = os.path.join(scripts_dir, item)
        if os.path.isfile(full_path):
            if item not in core_scripts and item != 'README.md':
                target_path = os.path.join(legacy_dir, item)
                shutil.move(full_path, target_path)
                moved_count += 1

    print(f"Cleanup Complete: Moved {moved_count} exploratory/test files into {legacy_dir}")
    print(f"Pristine Core Scripts Remaining in {scripts_dir}:")
    for s in sorted(core_scripts):
        print(f"  • {s}")

if __name__ == '__main__':
    main()
