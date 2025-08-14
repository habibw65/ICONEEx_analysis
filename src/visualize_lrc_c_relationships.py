

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Data Loading Functions (re-used from machine_learning_nee_model.py) ---
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
    lag_features = ['EVI', 'NDMI', 'NDVI', 'SAVI', 'TA', 'VPD', 'SW_IN', 'MODIS_PPFD']
    for feature in lag_features:
        merged_data[f'{feature}_lag1'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(1)
        merged_data[f'{feature}_lag2'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(2)

    # Seasonal indicators (sine and cosine of week number)
    merged_data['Week_sin'] = np.sin(2 * np.pi * merged_data['Week'] / 52)
    merged_data['Week_cos'] = np.cos(2 * np.pi * merged_data['Week'] / 52)

    # --- New Feature Engineering: Polynomial and Interaction Terms for TA ---
    print("--- Adding Polynomial and Interaction Terms for TA ---")
    # Polynomial terms for TA and its lags
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            merged_data[f'{ta_col}_sq'] = merged_data[ta_col]**2
            merged_data[f'{ta_col}_cub'] = merged_data[ta_col]**3

    # Interaction terms between TA (and its lags) and VIs
    vi_features = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            for vi_col in vi_features:
                if vi_col in merged_data.columns:
                    merged_data[f'{ta_col}_x_{vi_col}'] = merged_data[ta_col] * merged_data[vi_col]

    # Define features for plotting
    plot_features = ['TA', 'TA_lag1', 'TA_lag2', 'VPD', 'SW_IN', 'MODIS_PPFD',
                     'EVI', 'NDMI', 'NDVI', 'SAVI',
                     'Week_sin', 'Week_cos']
    
    # Add polynomial and interaction features to the plotting list if they exist
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            if f'{ta_col}_sq' in merged_data.columns: plot_features.append(f'{ta_col}_sq')
            if f'{ta_col}_cub' in merged_data.columns: plot_features.append(f'{ta_col}_cub')
            for vi_col in vi_features:
                if f'{ta_col}_x_{vi_col}' in merged_data.columns: plot_features.append(f'{ta_col}_x_{vi_col}')

    # Drop rows with NaN in 'c' or any of the plot_features
    plot_data = merged_data.dropna(subset=['c'] + plot_features)

    if plot_data.empty:
        print("Not enough data after dropping NaNs for plotting. Exiting.")
        exit()

    print("\n--- Generating Scatter Plots for LRC 'c' vs. Features ---")
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)

    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})

    for feature in plot_features:
        plt.figure(figsize=(8, 6))
        plt.scatter(plot_data[feature], plot_data['c'], alpha=0.6, s=10)
        plt.title(f"LRC 'c' vs. {feature} (2024)")
        plt.xlabel(feature)
        plt.ylabel("LRC 'c'")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_plots_dir, f'lrc_c_vs_{feature}_scatter.png'))
        plt.close()
        print(f"Scatter plot for LRC 'c' vs. {feature} saved.")

    print("Scatter plot generation complete.")
