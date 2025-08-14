import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

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
    lrc_2023_path = os.path.join(gurteen_dir, 'lrc_parameters_2023.csv')
    lrc_2024_path = os.path.join(gurteen_dir, 'lrc_parameters_2024.csv')
    lrc_dfs = []
    if os.path.exists(lrc_2023_path):
        lrc_dfs.append(pd.read_csv(lrc_2023_path))
    if os.path.exists(lrc_2024_path):
        lrc_dfs.append(pd.read_csv(lrc_2024_path))
    return pd.concat(lrc_dfs, ignore_index=True) if lrc_dfs else pd.DataFrame()

# --- Correlation Functions ---
def linear_func(x, a, b): return a * x + b
def exponential_func(x, a, b): return a * np.exp(b * x)
def logarithmic_func(x, a, b): return a * np.log(x + 1e-9) + b
def power_func(x, a, b): return a * np.power(x, b)

# --- NEE Modeling Functions ---
def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    if np.isclose(denom, 0): return np.full_like(ppfd, np.nan)
    exp_arg = np.clip((-c * ppfd) / denom, -700, 700)
    return 1 - denom * (1 - np.exp(exp_arg)) + b

def model_and_plot_nee_for_week(best_week_details, tower_base_dirs, satellite_data_weekly, gurteen_dir):
    target_week = best_week_details['Week']
    best_models = best_week_details['Models']
    print(f"\n--- Modeling NEE for Best Week: {target_week} ---")

    correlation_functions = {
        'linear': linear_func, 'exponential': exponential_func,
        'logarithmic': logarithmic_func, 'power': power_func
    }

    all_towers_comparison_data = []

    for tower_name, tower_dir in tower_base_dirs.items():
        for year in [2023, 2024]:
            processed_ec_data_path = os.path.join(tower_dir, f"processed_hesseflux_{year}.csv")
            if not os.path.exists(processed_ec_data_path): continue

            df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
            df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)
            df_ec_target_week = df_ec[df_ec['Week'] == target_week].copy()

            tower_satellite_data_week = satellite_data_weekly[
                (satellite_data_weekly['Tower'] == tower_name) &
                (satellite_data_weekly['Year'] == year) &
                (satellite_data_weekly['Week'] == target_week)
            ].copy()

            if df_ec_target_week.empty or tower_satellite_data_week.empty: continue

            # Calculate LRC params for this tower using the best models for the week
            try:
                param_a_model = best_models['a']
                vi_a_val = tower_satellite_data_week[param_a_model['VI']].iloc[0]
                func_a = correlation_functions[param_a_model['Type']]
                a_val = func_a(vi_a_val, param_a_model['Params'][0], param_a_model['Params'][1])

                param_b_model = best_models['b']
                vi_b_val = tower_satellite_data_week[param_b_model['VI']].iloc[0]
                func_b = correlation_functions[param_b_model['Type']]
                b_val = func_b(vi_b_val, param_b_model['Params'][0], param_b_model['Params'][1])

                param_c_model = best_models['c']
                vi_c_val = tower_satellite_data_week[param_c_model['VI']].iloc[0]
                func_c = correlation_functions[param_c_model['Type']]
                c_val = func_c(vi_c_val, param_c_model['Params'][0], param_c_model['Params'][1])

                y_predicted = mitscherlich_lrc(df_ec_target_week['PPFD'].values, a_val, b_val, c_val)
                y_observed = df_ec_target_week['NEE'].values

                valid_indices = ~np.isnan(y_observed) & ~np.isnan(y_predicted)
                if np.sum(valid_indices) == 0: continue

                all_towers_comparison_data.append({
                    'Tower_Year': f'{tower_name} - {year}',
                    'Measured NEE': np.mean(y_observed[valid_indices]),
                    'Modeled NEE': np.mean(y_predicted[valid_indices])
                })
            except Exception as e:
                print(f"Could not model NEE for {tower_name} {year} Week {target_week}: {e}")

    if not all_towers_comparison_data: 
        print("No data to plot for NEE comparison.")
        return

    plot_df = pd.DataFrame(all_towers_comparison_data).set_index('Tower_Year')
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'Weekly Mean NEE Comparison (Best Week: {target_week})')
    ax.set_xlabel('Tower - Year')
    ax.set_ylabel('Mean NEE (µmol m⁻² s⁻¹)')
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, f'best_week_nee_comparison_Week{target_week}.png'))
    plt.close()
    print(f"Best week NEE comparison bar chart saved to {combined_plots_dir}")

# Main execution block
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
    lrc_params_successful = lrc_params[lrc_params['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)
    merged_data = pd.merge(lrc_params_successful, satellite_data_weekly, on=['Tower', 'Year', 'Week'], how='inner')

    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_functions = {
        'linear': linear_func, 'exponential': exponential_func,
        'logarithmic': logarithmic_func, 'power': power_func
    }

    weekly_scores = []

    for week in range(1, 53):
        weekly_data = merged_data[merged_data['Week'] == week]
        if len(weekly_data) < 3: continue

        best_fits_for_week = {}
        for lrc_param in lrc_parameters:
            best_r2 = -1
            best_model = None
            for vi in vegetation_indices:
                for func_name, func in correlation_functions.items():
                    subset_data = weekly_data.dropna(subset=[lrc_param, vi])
                    if len(subset_data) < 2: continue
                    try:
                        params, _ = curve_fit(func, subset_data[vi].values, subset_data[lrc_param].values, maxfev=5000)
                        r2 = r2_score(subset_data[lrc_param].values, func(subset_data[vi].values, *params))
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = {'VI': vi, 'Type': func_name, 'R2': r2, 'Params': params}
                    except (RuntimeError, ValueError): continue
            if best_model: best_fits_for_week[lrc_param] = best_model

        if len(best_fits_for_week) == 3: # Ensure we have a best fit for all 3 params
            avg_r2 = np.mean([m['R2'] for m in best_fits_for_week.values()])
            weekly_scores.append({'Week': week, 'Avg_R2': avg_r2, 'Models': best_fits_for_week})

    if not weekly_scores:
        print("Could not find a suitable week with valid correlations for all LRC parameters.")
    else:
        scores_df = pd.DataFrame(weekly_scores)
        best_week_details = scores_df.loc[scores_df['Avg_R2'].idxmax()]
        print("--- Optimal Week Found ---")
        print(f"Best Week for Modeling: {best_week_details['Week']} (Avg R²: {best_week_details['Avg_R2']:.3f})")
        print("\nBest Fit Models for this Week:")
        for param, model in best_week_details['Models'].items():
            print(f"  LRC '{param}': {model['Type']} model with {model['VI']} (R²: {model['R2']:.3f})")
        
        model_and_plot_nee_for_week(best_week_details, base_directories, satellite_data_weekly, gurteen_dir)
