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

    # Define the specific combinations to test (from Week 26 best fits)
    combinations_to_test = {
        'a': {'VI': 'EVI', 'Function': exponential_func, 'Name': 'exponential'},
        'b': {'VI': 'SAVI', 'Function': exponential_func, 'Name': 'exponential'},
        'c': {'VI': 'SAVI', 'Function': logarithmic_func, 'Name': 'logarithmic'}
    }

    consistency_results = []

    print("--- Checking Consistency of Best Week 26 Combinations Across All Bi-Weeks (2024) ---")
    # Iterate through bi-weeks (1 to 26, since there are 52 weeks)
    for bi_week in range(1, 27):
        bi_weekly_data = merged_data[merged_data['Bi_Week'] == bi_week]
        if len(bi_weekly_data) < 2: # Need at least 2 points for curve_fit
            for lrc_param in combinations_to_test.keys():
                consistency_results.append({
                    'Bi_Week': bi_week,
                    'LRC_Parameter': lrc_param,
                    'VI': combinations_to_test[lrc_param]['VI'],
                    'Function': combinations_to_test[lrc_param]['Name'],
                    'R2': np.nan,
                    'Status': 'Not enough data'
                })
            continue

        for lrc_param, combo_info in combinations_to_test.items():
            vi_col = combo_info['VI']
            func = combo_info['Function']
            func_name = combo_info['Name']

            subset_data = bi_weekly_data.dropna(subset=[lrc_param, vi_col])
            if len(subset_data) < 2: continue

            X = subset_data[vi_col].values
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

                consistency_results.append({
                    'Bi_Week': bi_week,
                    'LRC_Parameter': lrc_param,
                    'VI': vi_col,
                    'Function': func_name,
                    'R2': r2,
                    'Status': 'Success'
                })
            except (RuntimeError, ValueError):
                consistency_results.append({
                    'Bi_Week': bi_week,
                    'LRC_Parameter': lrc_param,
                    'VI': vi_col,
                    'Function': func_name,
                    'R2': np.nan,
                    'Status': 'Fit failed'
                })

    consistency_df = pd.DataFrame(consistency_results)
    print("\nConsistency Check Results (R² values for each bi-week):")
    print(consistency_df.pivot_table(index='Bi_Week', columns=['LRC_Parameter', 'Function', 'VI'], values='R2').round(3))

    print("\nSummary of Consistency Check (Mean R² across all bi-weeks):")
    print(consistency_df.groupby(['LRC_Parameter', 'Function', 'VI'])['R2'].mean().round(3))

    print("\nBi-Weeks with R² > 0.60 for each combination:")
    high_r2_weeks = consistency_df[consistency_df['R2'] > 0.60]
    if not high_r2_weeks.empty:
        print(high_r2_weeks.pivot_table(index='Bi_Week', columns=['LRC_Parameter', 'Function', 'VI'], values='R2').round(3))
    else:
        print("No bi-weeks found with R² > 0.60 for any combination.")
