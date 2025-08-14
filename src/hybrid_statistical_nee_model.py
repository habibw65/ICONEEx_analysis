import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- Data Loading Functions (re-used) ---
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

def load_weekly_environmental_data(base_directories, year):
    all_env_dfs = []
    env_cols = ['TA', 'VPD', 'SW_IN'] # Environmental columns to load

    for tower_name, tower_dir in base_directories.items():
        processed_file_path = os.path.join(tower_dir, f"processed_hesseflux_{year}.csv")
        if os.path.exists(processed_file_path):
            df_ec = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)
            df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)
            df_ec['Year'] = df_ec.index.year # Add Year column
            df_ec['Tower'] = tower_name # Add tower name for merging
            
            # Select relevant environmental columns and calculate weekly mean
            weekly_env_df = df_ec.groupby(['Tower', 'Year', 'Week'])[env_cols].mean().reset_index()
            all_env_dfs.append(weekly_env_df)
    
    return pd.concat(all_env_dfs, ignore_index=True) if all_env_dfs else pd.DataFrame()

def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    # Handle cases where denom is zero or very close to zero
    if isinstance(denom, pd.Series):
        # For pandas Series, use element-wise check and fill with NaN
        result = pd.Series(np.full_like(ppfd, np.nan), index=ppfd.index)
        valid_indices = ~np.isclose(denom, 0)
        if valid_indices.any():
            exp_arg = (-c[valid_indices] * ppfd[valid_indices]) / denom[valid_indices]
            exp_arg = np.clip(exp_arg, -700, 700)
            result[valid_indices] = 1 - denom[valid_indices] * (1 - np.exp(exp_arg)) + b[valid_indices]
        return result
    else:
        # For single values
        if np.isclose(denom, 0): return np.full_like(ppfd, np.nan)
        exp_arg = np.clip((-c * ppfd) / denom, -700, 700)
        return 1 - denom * (1 - np.exp(exp_arg)) + b

# --- Statistical Functions for LRC Parameters ---
def linear_func(x, p1, p2): return p1 * x + p2
def exponential_func(x, p1, p2): return p1 * np.exp(p2 * x)
def logarithmic_func(x, p1, p2): return p1 * np.log(x + 1e-9) + p2
def power_func(x, p1, p2): return p1 * np.power(x, p2)

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
    weekly_env_data = load_weekly_environmental_data(base_directories, 2024) # Load 2024 environmental data
    
    lrc_params_2024 = lrc_params[lrc_params['Year'] == 2024].copy()
    satellite_data_2024 = satellite_data_weekly[satellite_data_weekly['Year'] == 2024].copy()
    lrc_params_successful = lrc_params_2024[lrc_params_2024['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)

    # Merge all data sources
    merged_data = pd.merge(lrc_params_successful, satellite_data_2024, on=['Tower', 'Year', 'Week'], how='inner')
    merged_data = pd.merge(merged_data, weekly_env_data, on=['Tower', 'Year', 'Week'], how='inner')

    # --- Feature Engineering: Lagged Variables and Seasonal Indicators ---
    print("--- Performing Feature Engineering ---")
    # Sort data for correct lagging
    merged_data = merged_data.sort_values(by=['Tower', 'Year', 'Week'])

    # Lagged VIs and Environmental variables
    lag_features = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    for feature in lag_features:
        merged_data[f'{feature}_lag1'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(1)
        merged_data[f'{feature}_lag2'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(2)

    # Seasonal indicators (sine and cosine of week number)
    merged_data['Week_sin'] = np.sin(2 * np.pi * merged_data['Week'] / 52)
    merged_data['Week_cos'] = np.cos(2 * np.pi * merged_data['Week'] / 52)

    # Define all potential features for statistical modeling
    all_features = ['EVI', 'NDMI', 'NDVI', 'SAVI',
                    'EVI_lag1', 'NDMI_lag1', 'NDVI_lag1', 'SAVI_lag1',
                    'EVI_lag2', 'NDMI_lag2', 'NDVI_lag2', 'SAVI_lag2',
                    'Week_sin', 'Week_cos']

    lrc_parameters = ['a', 'b', 'c']
    statistical_functions = {
        'linear': linear_func,
        'exponential': exponential_func,
        'logarithmic': logarithmic_func,
        'power': power_func
    }
    best_statistical_models = {}

    # Drop rows with NaN in any relevant column for statistical modeling
    statistical_data = merged_data.dropna(subset=lrc_parameters + all_features)

    if statistical_data.empty:
        print("Not enough data after dropping NaNs for statistical modeling. Exiting.")
        exit()

    print("--- Finding Best Statistical Relationships for LRC Parameters ---")
    for lrc_param in lrc_parameters:
        best_r2 = -np.inf
        best_model_details = None

        for feature in all_features:
            X = statistical_data[feature].values
            y = statistical_data[lrc_param].values

            for func_name, func in statistical_functions.items():
                try:
                    # Initial guesses for curve_fit
                    p0 = [1.0, 0.01] # Default initial guess
                    if func_name == 'logarithmic':
                        # Ensure x is positive for log
                        if (X <= 0).any(): continue
                        p0 = [1.0, 1.0]
                    elif func_name == 'power':
                        # Ensure x is positive for power
                        if (X <= 0).any(): continue
                        p0 = [1.0, 0.5]
                    
                    params, _ = curve_fit(func, X, y, p0=p0, maxfev=5000)
                    y_pred = func(X, *params)
                    r2 = r2_score(y, y_pred)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_details = {
                            'LRC_Parameter': lrc_param,
                            'Feature': feature,
                            'Function': func_name,
                            'R2': r2,
                            'Parameters': params.tolist() # Convert to list for storage
                        }
                except (RuntimeError, ValueError): # Handle cases where curve_fit fails
                    continue
        
        if best_model_details:
            best_statistical_models[lrc_param] = best_model_details
            print(f"  Best model for '{lrc_param}': {best_model_details['Function']} with {best_model_details['Feature']} (R²: {best_model_details['R2']:.3f})")
        else:
            print(f"  No suitable statistical model found for '{lrc_param}'.")

    # --- Apply Hybrid Statistical Model to Predict Weekly NEE ---
    print("\n--- Applying Hybrid Statistical Model to Predict Weekly NEE ---")
    weekly_hybrid_results = []
    for tower_name, tower_dir in base_directories.items():
        processed_ec_data_path = os.path.join(tower_dir, "processed_hesseflux_2024.csv")
        if not os.path.exists(processed_ec_data_path): continue
        df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
        df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)

        for week in range(1, 53):
            df_ec_target_week = df_ec[df_ec['Week'] == week].copy()
            
            # Get all features for the current week and tower from merged_data
            current_week_data_full = merged_data[
                (merged_data['Tower'] == tower_name) &
                (merged_data['Week'] == week)
            ]

            if df_ec_target_week.empty or current_week_data_full.empty: continue

            try:
                # Dynamically calculate LRC parameters using the best statistical models
                a_model = best_statistical_models.get('a')
                b_model = best_statistical_models.get('b')
                c_model = best_statistical_models.get('c')

                if not (a_model and b_model and c_model): continue

                # Get feature values for prediction
                a_feature_val = current_week_data_full[a_model['Feature']].iloc[0]
                b_feature_val = current_week_data_full[b_model['Feature']].iloc[0]
                c_feature_val = current_week_data_full[c_model['Feature']].iloc[0]

                # Calculate a, b, c using their respective best statistical functions
                a_val = statistical_functions[a_model['Function']](a_feature_val, *a_model['Parameters'])
                b_val = statistical_functions[b_model['Function']](b_feature_val, *b_model['Parameters'])
                c_val = statistical_functions[c_model['Function']](c_feature_val, *c_model['Parameters'])

                # Predict NEE using Mitscherlich function with dynamically calculated LRC params
                y_predicted = mitscherlich_lrc(df_ec_target_week['PPFD'].values, a_val, b_val, c_val)
                y_observed = df_ec_target_week['NEE'].values

                valid_indices = ~np.isnan(y_observed) & ~np.isnan(y_predicted)
                if np.sum(valid_indices) > 0:
                    weekly_hybrid_results.append({
                        'Tower': tower_name, 'Week': week,
                        'Measured_NEE': np.mean(y_observed[valid_indices]),
                        'Modeled_NEE': np.mean(y_predicted[valid_indices])
                    })
            except Exception as e:
                # print(f"Error predicting NEE for {tower_name} Week {week}: {e}")
                continue

    hybrid_results_df = pd.DataFrame(weekly_hybrid_results)

    # --- 4. Visualize and Report ---
    print("\n--- Generating Time Series Comparison Plots for Hybrid Statistical Model ---")
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})
    fig, axes = plt.subplots(len(base_directories), 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Hybrid Statistical Model: Measured vs. Modeled Weekly NEE (2024)', fontsize=16, y=0.95)

    for i, (tower_name, ax) in enumerate(zip(base_directories.keys(), axes.flatten())):
        tower_data = hybrid_results_df[hybrid_results_df['Tower'] == tower_name]
        if tower_data.empty: 
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        ax.plot(tower_data['Week'], tower_data['Measured_NEE'], 'o-', label='Measured NEE', markersize=4)
        ax.plot(tower_data['Week'], tower_data['Modeled_NEE'], 's--', label='Modeled NEE', markersize=4)
        
        if not tower_data.empty:
            overall_r2 = r2_score(tower_data['Measured_NEE'], tower_data['Modeled_NEE'])
            overall_rmse = np.sqrt(mean_squared_error(tower_data['Measured_NEE'], tower_data['Modeled_NEE']))
            ax.set_title(f'{tower_name} (R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f})')
        else:
            ax.set_title(tower_name)

        ax.set_ylabel('Mean NEE')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel('Week of Year')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, 'hybrid_statistical_nee_model_weekly_nee_comparison.png'))
    plt.close()
    print(f"Hybrid statistical model comparison plot saved to {combined_plots_dir}")
