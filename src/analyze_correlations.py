import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def load_and_prepare_satellite_data(data_dir, file_name, value_name):
    """
    Loads a satellite data CSV, converts time, melts it, and adds week/year.
    """
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, parse_dates=['system:time_start'], infer_datetime_format=True)
    df = df.rename(columns={'system:time_start': 'DateTime'})
    
    # Melt the DataFrame to long format
    df_melted = df.melt(id_vars=['DateTime'], var_name='Tower', value_name=value_name)
    
    df_melted['Year'] = df_melted['DateTime'].dt.year
    df_melted['Week'] = df_melted['DateTime'].dt.isocalendar().week.astype(int)
    
    # Calculate weekly average
    df_weekly = df_melted.groupby(['Tower', 'Year', 'Week'])[value_name].mean().reset_index()
    
    return df_weekly

def load_all_satellite_data(satellite_data_dir):
    """
    Loads and prepares all satellite data (PPFD and VIs).
    """
    ppfd_df = load_and_prepare_satellite_data(satellite_data_dir, 'daily_mean_modis_ppfd.csv', 'MODIS_PPFD')
    evi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_evi_daily.csv', 'EVI')
    ndmi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndmi_daily.csv', 'NDMI')
    ndvi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndvi_daily.csv', 'NDVI')
    savi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_savi_daily.csv', 'SAVI')

    # Merge all satellite dataframes
    merged_satellite_df = ppfd_df
    for df_vi in [evi_df, ndmi_df, ndvi_df, savi_df]:
        merged_satellite_df = pd.merge(merged_satellite_df, df_vi, on=['Tower', 'Year', 'Week'], how='outer')
        
    return merged_satellite_df

def load_lrc_parameters(gurteen_dir):
    """
    Loads and combines LRC parameters for 2023 and 2024.
    """
    lrc_2023_path = os.path.join(gurteen_dir, 'lrc_parameters_2023.csv')
    lrc_2024_path = os.path.join(gurteen_dir, 'lrc_parameters_2024.csv')
    
    lrc_dfs = []
    if os.path.exists(lrc_2023_path):
        lrc_dfs.append(pd.read_csv(lrc_2023_path))
    if os.path.exists(lrc_2024_path):
        lrc_dfs.append(pd.read_csv(lrc_2024_path))
        
    if lrc_dfs:
        return pd.concat(lrc_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# --- Correlation Functions ---
def linear_func(x, a, b):
    return a * x + b

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def logarithmic_func(x, a, b):
    # Add a small constant to avoid log(0)
    return a * np.log(x + 1e-9) + b

def power_func(x, a, b):
    return a * np.power(x, b)

# Main execution block
if __name__ == '__main__':
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    gurteen_dir = "/Users/habibw/Documents/Gurteen"

    print("Loading and preparing satellite data...")
    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    print("Satellite data loaded and aggregated to weekly averages.")

    print("Loading LRC parameters...")
    lrc_params = load_lrc_parameters(gurteen_dir)
    print("LRC parameters loaded.")

    lrc_params_successful = lrc_params[lrc_params['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)

    merged_data = pd.merge(lrc_params_successful, satellite_data_weekly, on=['Tower', 'Year', 'Week'], how='inner')

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func,
        'exponential': exponential_func,
        'logarithmic': logarithmic_func,
        'power': power_func
    }

    all_results = []

    for week in range(1, 53):
        weekly_data = merged_data[merged_data['Week'] == week]
        if len(weekly_data) < 2:
            continue

        for lrc_param in lrc_parameters:
            for vi in vegetation_indices:
                for func_name, func in correlation_functions.items():
                    subset_data = weekly_data.dropna(subset=[lrc_param, vi])
                    if len(subset_data) < 2:
                        continue

                    x_data = subset_data[vi].values
                    y_data = subset_data[lrc_param].values

                    try:
                        params, _ = curve_fit(func, x_data, y_data, maxfev=5000)
                        y_predicted = func(x_data, *params)
                        r2 = r2_score(y_data, y_predicted)
                        all_results.append({
                            'Week': week,
                            'LRC_Parameter': lrc_param,
                            'VI': vi,
                            'Correlation_Type': func_name,
                            'R2': r2,
                            'Param1': params[0],
                            'Param2': params[1]
                        })
                    except (RuntimeError, ValueError):
                        continue

    results_df = pd.DataFrame(all_results)

    # --- Find and Report Best Fits ---
    print("\n--- Best Correlation Results ---")
    for lrc_param in lrc_parameters:
        param_df = results_df[results_df['LRC_Parameter'] == lrc_param]
        if not param_df.empty:
            best_fit = param_df.loc[param_df['R2'].idxmax()]
            print(f"\nBest fit for LRC Parameter '{lrc_param}':")
            print(best_fit)
        else:
            print(f"\nNo successful correlations found for LRC Parameter '{lrc_param}'.")
