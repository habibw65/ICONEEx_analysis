import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- Data Loading and Correlation Functions (re-used) ---
def load_and_prepare_satellite_data(data_dir, file_name, value_name):
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, parse_dates=['system:time_start'], infer_datetime_format=True)
    df = df.rename(columns={'system:time_start': 'DateTime'})
    df_melted = df.melt(id_vars=['DateTime'], var_name='Tower', value_name=value_name)
    df_melted['Year'] = df_melted['DateTime'].dt.year
    df_melted['Week'] = df_melted['DateTime'].dt.isocalendar().week.astype(int)
    df_weekly = df_melted.groupby(['Tower', 'Year', 'Week'])[value_name].mean().reset_index()
    return df_weekly

def load_all_satellite_data(satellite_data_dir):
    ppfd_df = load_and_prepare_satellite_data(satellite_data_dir, 'daily_mean_modis_ppfd.csv', 'MODIS_PPFD')
    evi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_evi_daily.csv', 'EVI')
    ndmi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndmi_daily.csv', 'NDMI')
    ndvi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndvi_daily.csv', 'NDVI')
    savi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_savi_daily.csv', 'SAVI')
    merged_satellite_df = ppfd_df
    for df_vi in [evi_df, ndmi_df, ndvi_df, savi_df]:
        merged_satellite_df = pd.merge(merged_satellite_df, df_vi, on=['Tower', 'Year', 'Week'], how='outer')
    return merged_satellite_df

def load_lrc_parameters(gurteen_dir):
    lrc_2024_path = os.path.join(gurteen_dir, 'lrc_parameters_2024.csv')
    return pd.read_csv(lrc_2024_path) if os.path.exists(lrc_2024_path) else pd.DataFrame()

def linear_func(x, a, b): return a * x + b
def exponential_func(x, a, b): return a * np.exp(b * x)
def logarithmic_func(x, a, b): return a * np.log(x + 1e-9) + b
def power_func(x, a, b): return a * np.power(x, b)

def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    if np.isclose(denom, 0): return np.full_like(ppfd, np.nan)
    exp_arg = np.clip((-c * ppfd) / denom, -700, 700)
    return 1 - denom * (1 - np.exp(exp_arg)) + b

# --- Main Execution Block ---
if __name__ == '__main__':
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    gurteen_dir = "/Users/habibw/Documents/Gurteen"
    base_directories = {
        "Gurteen": "/Users/habibw/Documents/Gurteen", "Athenry": "/Users/habibw/Documents/Athenry",
        "JC1": "/Users/habibw/Documents/JC1", "JC2": "/Users/habibw/Documents/JC2",
        "Timoleague": "/Users/habibw/Documents/Timoleague"
    }

    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    lrc_params = load_lrc_parameters(gurteen_dir)
    
    lrc_params_2024 = lrc_params[lrc_params['Year'] == 2024].copy()
    satellite_data_2024 = satellite_data_weekly[satellite_data_weekly['Year'] == 2024].copy()
    lrc_params_successful = lrc_params_2024[lrc_params_2024['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)
    merged_data = pd.merge(lrc_params_successful, satellite_data_2024, on=['Tower', 'Year', 'Week'], how='inner')

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func, 'exponential': exponential_func,
        'logarithmic': logarithmic_func, 'power': power_func
    }

    # --- 1. Find Globally Best Predictors ---
    print("--- Finding Globally Best Predictors for LRC Parameters (2024) ---")
    global_best_models = {}
    for lrc_param in lrc_parameters:
        best_r2 = -1
        best_model = None
        for vi in vegetation_indices:
            for func_name, func in correlation_functions.items():
                subset_data = merged_data.dropna(subset=[lrc_param, vi])
                if len(subset_data) < 2: continue
                try:
                    params, _ = curve_fit(func, subset_data[vi].values, subset_data[lrc_param].values, maxfev=5000)
                    r2 = r2_score(subset_data[lrc_param].values, func(subset_data[vi].values, *params))
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = {'VI': vi, 'Type': func_name, 'R2': r2, 'Params': params}
                except (RuntimeError, ValueError): continue
        if best_model:
            global_best_models[lrc_param] = best_model
            print(f"  Best model for '{lrc_param}': {best_model['Type']} with {best_model['VI']} (R²: {best_model['R2']:.3f})")

    # --- 2. Apply Global Model and Evaluate Weekly ---
    print("\n--- Applying Global Model to Predict Weekly NEE ---")
    weekly_results = []
    for tower_name, tower_dir in base_directories.items():
        processed_ec_data_path = os.path.join(tower_dir, "processed_hesseflux_2024.csv")
        if not os.path.exists(processed_ec_data_path): continue
        df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
        df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)

        for week in range(1, 53):
            df_ec_target_week = df_ec[df_ec['Week'] == week].copy()
            tower_satellite_data_week = satellite_data_2024[
                (satellite_data_2024['Tower'] == tower_name) & (satellite_data_2024['Week'] == week)
            ].copy()

            if df_ec_target_week.empty or tower_satellite_data_week.empty: continue

            try:
                a_model = global_best_models['a']
                b_model = global_best_models['b']
                c_model = global_best_models['c']

                a_val = correlation_functions[a_model['Type']](tower_satellite_data_week[a_model['VI']].iloc[0], *a_model['Params'])
                b_val = correlation_functions[b_model['Type']](tower_satellite_data_week[b_model['VI']].iloc[0], *b_model['Params'])
                c_val = correlation_functions[c_model['Type']](tower_satellite_data_week[c_model['VI']].iloc[0], *c_model['Params'])

                y_predicted = mitscherlich_lrc(df_ec_target_week['PPFD'].values, a_val, b_val, c_val)
                y_observed = df_ec_target_week['NEE'].values

                valid_indices = ~np.isnan(y_observed) & ~np.isnan(y_predicted)
                if np.sum(valid_indices) > 0:
                    weekly_results.append({
                        'Tower': tower_name, 'Week': week,
                        'Measured_NEE': np.mean(y_observed[valid_indices]),
                        'Modeled_NEE': np.mean(y_predicted[valid_indices])
                    })
            except Exception: continue

    results_df = pd.DataFrame(weekly_results)

    # --- 3. Visualize and Report ---
    print("\n--- Generating Time Series Comparison Plots ---")
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})
    fig, axes = plt.subplots(len(base_directories), 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Global Model: Measured vs. Modeled Weekly NEE (2024)', fontsize=16, y=0.95)

    for i, (tower_name, ax) in enumerate(zip(base_directories.keys(), axes.flatten())):
        tower_data = results_df[results_df['Tower'] == tower_name]
        if tower_data.empty: 
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        ax.plot(tower_data['Week'], tower_data['Measured_NEE'], 'o-', label='Measured NEE', markersize=4)
        ax.plot(tower_data['Week'], tower_data['Modeled_NEE'], 's--', label='Modeled NEE', markersize=4)
        
        overall_r2 = r2_score(tower_data['Measured_NEE'], tower_data['Modeled_NEE'])
        overall_rmse = np.sqrt(mean_squared_error(tower_data['Measured_NEE'], tower_data['Modeled_NEE']))
        
        ax.set_title(f'{tower_name} (R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f})')
        ax.set_ylabel('Mean NEE')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel('Week of Year')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, 'global_model_weekly_nee_comparison.png'))
    plt.close()
    print(f"Global model comparison plot saved to {combined_plots_dir}")
