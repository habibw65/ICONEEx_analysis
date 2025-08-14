

import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
    train_towers = ["Athenry", "JC1", "JC2"]
    test_towers = ["Gurteen", "Timoleague"]

    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    lrc_params = load_lrc_parameters(gurteen_dir)

    lrc_params_2024 = lrc_params[lrc_params['Year'] == 2024].copy()
    satellite_data_2024 = satellite_data_weekly[satellite_data_weekly['Year'] == 2024].copy()
    lrc_params_successful = lrc_params_2024[lrc_params_2024['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)
    merged_data = pd.merge(lrc_params_successful, satellite_data_2024, on=['Tower', 'Year', 'Week'], how='inner')

    # Split into training and testing data
    train_data = merged_data[merged_data['Tower'].isin(train_towers)]

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func, 'exponential': exponential_func,
        'logarithmic': logarithmic_func, 'power': power_func
    }

    weekly_scores = []
    for week in range(1, 53):
        weekly_train_data = train_data[train_data['Week'] == week]
        if len(weekly_train_data) < 3: continue
        best_fits_for_week = {}
        for lrc_param in lrc_parameters:
            best_r2 = -1
            best_model = None
            for vi in vegetation_indices:
                for func_name, func in correlation_functions.items():
                    subset_data = weekly_train_data.dropna(subset=[lrc_param, vi])
                    if len(subset_data) < 2: continue
                    try:
                        params, _ = curve_fit(func, subset_data[vi].values, subset_data[lrc_param].values, maxfev=5000)
                        r2 = r2_score(subset_data[lrc_param].values, func(subset_data[vi].values, *params))
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = {'VI': vi, 'Type': func_name, 'R2': r2, 'Params': params}
                    except (RuntimeError, ValueError): continue
            if best_model: best_fits_for_week[lrc_param] = best_model
        if len(best_fits_for_week) == 3:
            avg_r2 = np.mean([m['R2'] for m in best_fits_for_week.values()])
            weekly_scores.append({'Week': week, 'Avg_R2': avg_r2, 'Models': best_fits_for_week})

    if not weekly_scores:
        print("Could not find a suitable week in 2024 from training data.")
        exit()

    scores_df = pd.DataFrame(weekly_scores)
    best_week_details = scores_df.loc[scores_df['Avg_R2'].idxmax()]
    target_week = best_week_details['Week']
    best_models = best_week_details['Models']

    print("--- Leave-Two-Out Cross-Validation (2024) ---")
    print(f"Training Towers: {train_towers}")
    print(f"Testing Towers: {test_towers}")
    print(f"\nOptimal Week (from training data): {target_week} (Avg R²: {best_week_details['Avg_R2']:.3f})")
    print("\nBest Fit Models (from training data):")
    for param, model in best_models.items():
        print(f"  LRC '{param}': {model['Type']} model with {model['VI']} (R²: {model['R2']:.3f})")

    # --- Apply model to test towers ---
    all_towers_comparison_data = []
    for tower_name in test_towers:
        tower_dir = base_directories[tower_name]
        processed_ec_data_path = os.path.join(tower_dir, "processed_hesseflux_2024.csv")
        if not os.path.exists(processed_ec_data_path): continue

        df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
        df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)
        df_ec_target_week = df_ec[df_ec['Week'] == target_week].copy()

        tower_satellite_data_week = satellite_data_2024[
            (satellite_data_2024['Tower'] == tower_name) &
            (satellite_data_2024['Week'] == target_week)
        ].copy()

        if df_ec_target_week.empty or tower_satellite_data_week.empty: continue

        try:
            vi_a_val = tower_satellite_data_week[best_models['a']['VI']].iloc[0]
            a_val = correlation_functions[best_models['a']['Type']](vi_a_val, *best_models['a']['Params'])
            vi_b_val = tower_satellite_data_week[best_models['b']['VI']].iloc[0]
            b_val = correlation_functions[best_models['b']['Type']](vi_b_val, *best_models['b']['Params'])
            vi_c_val = tower_satellite_data_week[best_models['c']['VI']].iloc[0]
            c_val = correlation_functions[best_models['c']['Type']](vi_c_val, *best_models['c']['Params'])

            y_predicted = mitscherlich_lrc(df_ec_target_week['PPFD'].values, a_val, b_val, c_val)
            y_observed = df_ec_target_week['NEE'].values

            valid_indices = ~np.isnan(y_observed) & ~np.isnan(y_predicted)
            if np.sum(valid_indices) > 0:
                all_towers_comparison_data.append({
                    'Tower': tower_name,
                    'Measured NEE': np.mean(y_observed[valid_indices]),
                    'Modeled NEE': np.mean(y_predicted[valid_indices])
                })
        except Exception as e:
            print(f"Could not model NEE for {tower_name} Week {target_week}: {e}")

    if not all_towers_comparison_data:
        print("No data to plot for NEE comparison.")
        exit()

    plot_df = pd.DataFrame(all_towers_comparison_data).set_index('Tower')
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind='bar', ax=ax, width=0.6)
    ax.set_title(f'Leave-Two-Out Validation (Test Set): Week {target_week}, 2024')
    ax.set_xlabel('Tower')
    ax.set_ylabel('Mean NEE (µmol m⁻² s⁻¹)')
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, f'leave_two_out_validation_Week{target_week}.png'))
    plt.close()
    print(f"\nLeave-two-out validation plot saved to {combined_plots_dir}")
