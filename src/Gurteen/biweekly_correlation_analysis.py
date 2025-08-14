
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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

# --- Statistical Functions for LRC Parameters ---
def linear_func(x, p1, p2): return p1 * x + p2
def exponential_func(x, p1, p2): return p1 * np.exp(p2 * x)
def logarithmic_func(x, p1, p2): return p1 * np.log(x + 1e-9) + p2
def power_func(x, p1, p2): return p1 * np.power(x, p2)

# --- Main Execution Block ---
if __name__ == '__main__':
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    gurteen_dir = "/Users/habibw/Documents/Gurteen"

    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    lrc_params = load_lrc_parameters(gurteen_dir)
    
    lrc_params_2024 = lrc_params[lrc_params['Year'] == 2024].copy()
    satellite_data_2024 = satellite_data_weekly[satellite_data_weekly['Year'] == 2024].copy()
    lrc_params_successful = lrc_params_2024[lrc_params_2024['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)

    # Merge all data sources
    merged_data = pd.merge(lrc_params_successful, satellite_data_2024, on=['Tower', 'Year', 'Week'], how='inner')

    # Calculate Bi_Week
    merged_data['Bi_Week'] = (merged_data['Week'] - 1) // 2 + 1

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func,
        'exponential': exponential_func,
        'logarithmic': logarithmic_func,
        'power': power_func
    }

    # Aggregate data by Bi_Week, Tower, and Year
    # We need to average LRC params and VIs for each bi-week for each tower
    # This creates the dataset for finding optimal combos across bi-weeks
    numeric_cols_for_biweekly_agg = lrc_parameters + vegetation_indices
    biweekly_aggregated_data = merged_data.groupby(['Tower', 'Year', 'Bi_Week'])[numeric_cols_for_biweekly_agg].mean().reset_index()

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func,
        'exponential': exponential_func,
        'logarithmic': logarithmic_func,
        'power': power_func
    }

    best_biweekly_combos = {}

    print("--- Finding Optimal VI-Correlation Combinations for Bi-Weekly Data (2024) ---")
    for lrc_param in lrc_parameters:
        best_r2 = -np.inf
        best_combo_details = None

        for vi in vegetation_indices:
            for func_name, func in correlation_functions.items():
                # Use the biweekly_aggregated_data for fitting
                subset_data = biweekly_aggregated_data.dropna(subset=[lrc_param, vi])
                
                if len(subset_data) < 2: continue

                X = subset_data[vi].values
                y = subset_data[lrc_param].values

                try:
                    p0 = [1.0, 0.01] # Default initial guess
                    if func_name == 'logarithmic':
                        if (X <= 0).any(): continue
                        p0 = [1.0, 1.0]
                    elif func_name == 'power':
                        if (X <= 0).any(): continue
                        p0 = [1.0, 0.5]
                    
                    params, _ = curve_fit(func, X, y, p0=p0, maxfev=5000)
                    y_pred = func(X, *params)
                    r2 = r2_score(y, y_pred)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_combo_details = {
                            'LRC_Parameter': lrc_param,
                            'VI': vi,
                            'Function': func_name,
                            'R2': r2,
                            'Parameters': params.tolist() # Convert to list for storage
                        }
                except (RuntimeError, ValueError): # Handle cases where curve_fit fails
                    continue
        
        if best_combo_details:
            best_biweekly_combos[lrc_param] = best_combo_details
            print(f"  Best combo for '{lrc_param}': {best_combo_details['Function']} with {best_combo_details['VI']} (R²: {best_combo_details['R2']:.3f})")
        else:
            print(f"  No suitable bi-weekly combo found for '{lrc_param}'.")
