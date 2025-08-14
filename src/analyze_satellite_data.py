import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

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

def exponential_func(x, A, B):
    """
    Exponential function for correlation analysis: y = A * exp(B * x)
    """
    return A * np.exp(B * x)

# Mitscherlich function for Light Response Curve
def mitscherlich_lrc(ppfd, a, b, c):
    # NEE = 1 - (a + b) * (1 - exp((-c * PPFD) / (a + b))) + b
    # Note: This function is implemented as provided by the user.
    # It is an unusual form for a Mitscherlich LRC, especially the '1 - (a + b)' part.
    # If PPFD = 0, NEE = 1 + b, not b (respiration).
    # Given the user's equation, if a+b is zero, the equation breaks.
    # For practical fitting, we'll return NaN or a large value to indicate a bad fit.
    
    denom = a + b
    
    # Handle potential division by zero or very small (a+b) element-wise
    # Create a mask for problematic denominators
    problematic_denom_mask = np.isclose(denom, 0)
    
    # Initialize result array with NaNs
    result = np.full_like(ppfd, np.nan)

    # Calculate only where denominator is not problematic
    valid_indices = ~problematic_denom_mask
    if np.any(valid_indices):
        exp_arg = (-c[valid_indices] * ppfd[valid_indices]) / denom[valid_indices]
        exp_arg = np.clip(exp_arg, -700, 700) # Clip to prevent overflow/underflow
        result[valid_indices] = 1 - denom[valid_indices] * (1 - np.exp(exp_arg)) + b[valid_indices]

    return result

# Exponential functions for LRC parameters a, b, c
def a_func(evi, A_a, B_a):
    return A_a * np.exp(B_a * evi)

def b_func(evi, A_b, B_b):
    return A_b * np.exp(B_b * evi)

def c_func(evi, A_c, B_c):
    return A_c * np.exp(B_c * evi)

# Combined NEE model using satellite-derived VI and PPFD
def nee_model(data, A_a, B_a, A_b, B_b, A_c, B_c):
    ppfd = data[:, 0]
    evi = data[:, 1]

    a = a_func(evi, A_a, B_a)
    b = b_func(evi, A_b, B_b)
    c = c_func(evi, A_c, B_c)

    return mitscherlich_lrc(ppfd, a, b, c)

def plot_single_weekly_mean_nee_comparison_all_towers(tower_base_dirs, best_vi, correlation_df, gurteen_dir, target_week):
    print(f"\nGenerating single bar chart for weekly mean NEE comparison for Week {target_week} - All Towers...")

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    lrc_parameters = ['a', 'b', 'c']
    initial_guesses_combined_model = []
    for param in lrc_parameters:
        corr_row = correlation_df[(correlation_df['LRC_Parameter'] == param) & (correlation_df['VI'] == best_vi)]
        if not corr_row.empty and corr_row['Fit_Status'].iloc[0] == 'Success':
            initial_guesses_combined_model.extend([corr_row['A'].iloc[0], corr_row['B'].iloc[0]])
        else:
            initial_guesses_combined_model.extend([1.0, 0.01]) 

    satellite_data_weekly = load_all_satellite_data(os.path.join(gurteen_dir, "..", "satellite derived data"))

    all_towers_comparison_data = []

    for tower_name, tower_dir in tower_base_dirs.items():
        for year in [2023, 2024]: # Iterate through both years
            processed_ec_data_path = os.path.join(tower_dir, f"processed_hesseflux_{year}.csv")
            
            if not os.path.exists(processed_ec_data_path):
                continue
            
            df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
            df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)

            df_ec_target_week = df_ec[df_ec['Week'] == target_week].copy()

            tower_satellite_data_week = satellite_data_weekly[(satellite_data_weekly['Tower'] == tower_name) &
                                                              (satellite_data_weekly['Year'] == year) &
                                                              (satellite_data_weekly['Week'] == target_week)].copy()
            
            if df_ec_target_week.empty or tower_satellite_data_week.empty:
                continue

            current_evi = tower_satellite_data_week[best_vi].iloc[0]
            current_modis_ppfd = tower_satellite_data_week['MODIS_PPFD'].iloc[0]

            X_data_predict = df_ec_target_week[['PPFD']].copy()
            X_data_predict['EVI'] = current_evi
            X_data_predict = X_data_predict.values

            y_observed_target_week = df_ec_target_week['NEE'].values

            valid_indices = ~np.isnan(y_observed_target_week)
            X_data_predict = X_data_predict[valid_indices]
            y_observed_target_week = y_observed_target_week[valid_indices]

            if len(y_observed_target_week) == 0:
                continue

            try:
                y_predicted_target_week = nee_model(X_data_predict, *initial_guesses_combined_model)

                mean_measured_nee = np.mean(y_observed_target_week)
                mean_modeled_nee = np.mean(y_predicted_target_week)

                all_towers_comparison_data.append({
                    'Tower_Year': f'{tower_name} - {year}',
                    'Measured NEE': mean_measured_nee,
                    'Modeled NEE': mean_modeled_nee
                })

            except Exception as e:
                print(f"Error processing {tower_name} - {year} - Week {target_week} for single bar chart: {e}")

    if not all_towers_comparison_data:
        print(f"No valid data to generate single bar chart for Week {target_week} - All Towers.")
        return

    plot_df = pd.DataFrame(all_towers_comparison_data).set_index('Tower_Year')

    fig, ax = plt.subplots(figsize=(14, 8)) # Adjusted for multiple towers
    plot_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f'Weekly Mean NEE Comparison (Week {target_week}) - All Towers')
    ax.set_xlabel('Tower - Year')
    ax.set_ylabel('Mean NEE (µmol m⁻² s⁻¹)')
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to the new combined_plots folder in Documents
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, f'weekly_nee_comparison_all_towers_Week{target_week}.png'))
    plt.close()
    print(f"Single weekly mean NEE comparison bar chart for Week {target_week} saved to {combined_plots_dir}")


# Main execution block
if __name__ == '__main__':
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    gurteen_dir = "/Users/habibw/Documents/Gurteen" # Assuming LRC parameters are here
    base_directories = {
        "Gurteen": "/Users/habibw/Documents/Gurteen",
        "Athenry": "/Users/habibw/Documents/Athenry",
        "JC1": "/Users/habibw/Documents/JC1",
        "JC2": "/Users/habibw/Documents/JC2",
        "Timoleague": "/Users/habibw/Documents/Timoleague"
    }

    print("Loading and preparing satellite data...")
    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    print("Satellite data loaded and aggregated to weekly averages.")

    print("Loading LRC parameters...")
    lrc_params = load_lrc_parameters(gurteen_dir)
    print("LRC parameters loaded.")

    # Merge satellite data with LRC parameters
    # We only want successful LRC fits for correlation analysis
    lrc_params_successful = lrc_params[lrc_params['Fit_Status'] == 'Success'].copy()
    
    # Ensure 'Week' column in LRC parameters is integer for merging
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)

    merged_data = pd.merge(lrc_params_successful, satellite_data_weekly, on=['Tower', 'Year', 'Week'], how='inner')

    print(f"Merged data shape: {merged_data.shape}")
    print("Merged data head:")
    print(merged_data.head())

    # Save the merged data for inspection (optional)
    merged_data.to_csv(os.path.join(gurteen_dir, "merged_lrc_satellite_data.csv"), index=False)
    print(f"Merged data saved to {os.path.join(gurteen_dir, 'merged_lrc_satellite_data.csv')}")

    # --- Phase 2: Exponential Correlation Analysis (already done, but re-run for context) ---
    print("\nPerforming Exponential Correlation Analysis...")
    lrc_parameters = ['a', 'b', 'c']
    vegetation_indices = ['EVI', 'NDMI', 'NDVI', 'SAVI']

    correlation_results = []

    for lrc_param in lrc_parameters:
        for vi in vegetation_indices:
            # Drop rows with NaN in either the LRC parameter or the VI
            subset_data = merged_data.dropna(subset=[lrc_param, vi])

            if len(subset_data) < 2: # Need at least 2 points for curve_fit
                correlation_results.append({
                    'LRC_Parameter': lrc_param,
                    'VI': vi,
                    'A': np.nan, 'B': np.nan,
                    'R2': np.nan,
                    'Fit_Status': 'Not enough data'
                })
                continue

            try:
                x_data = subset_data[vi].values
                y_data = subset_data[lrc_param].values

                A_guess = np.mean(y_data) if np.mean(y_data) > 0 else 1.0
                B_guess = 0.01 

                p0 = [A_guess, B_guess]
                bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

                params, pcov = curve_fit(exponential_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
                
                y_predicted = exponential_func(x_data, *params)
                r2 = r2_score(y_data, y_predicted)

                correlation_results.append({
                    'LRC_Parameter': lrc_param,
                    'VI': vi,
                    'A': params[0], 'B': params[1],
                    'R2': r2,
                    'Fit_Status': 'Success'
                })

            except RuntimeError as e:
                correlation_results.append({
                    'LRC_Parameter': lrc_param,
                    'VI': vi,
                    'A': np.nan, 'B': np.nan,
                    'R2': np.nan,
                    'Fit_Status': f'Fit failed: {e}'
                })
            except ValueError as e:
                correlation_results.append({
                    'LRC_Parameter': lrc_param,
                    'VI': vi,
                    'A': np.nan, 'B': np.nan,
                    'R2': np.nan,
                    'Fit_Status': f'Fit failed (invalid input): {e}'
                })

    correlation_df = pd.DataFrame(correlation_results)
    print("\nExponential Correlation Results:")
    print(correlation_df)

    successful_correlations = correlation_df[correlation_df['Fit_Status'] == 'Success'].copy()
    successful_correlations = successful_correlations.dropna(subset=['R2'])

    if not successful_correlations.empty:
        avg_r2_by_vi = successful_correlations.groupby('VI')['R2'].mean().sort_values(ascending=False)
        best_vi = avg_r2_by_vi.index[0]
        print(f"\nBest Vegetation Index (highest average R² across LRC parameters): {best_vi} (Average R²: {avg_r2_by_vi.iloc[0]:.2f})")
    else:
        best_vi = None
        print("No successful exponential correlations found to determine the best VI.")

    # --- Phase 3: Develop and Model NEE (Focus on Week 43, 2024) ---
    print("\nModeling NEE for Week 43, 2024 using satellite-derived VI and PPFD...")

    if best_vi is None:
        print("Cannot model NEE without a best VI. Please check correlation analysis.")
    else:
        # Get the fitted parameters for a, b, c from the correlation analysis (Week 43, 2024)
        # These will be used as initial guesses for the combined NEE model
        initial_guesses_combined_model = []
        for param in lrc_parameters:
            corr_row = correlation_df[(correlation_df['LRC_Parameter'] == param) & (correlation_df['VI'] == best_vi)]
            if not corr_row.empty and corr_row['Fit_Status'].iloc[0] == 'Success':
                initial_guesses_combined_model.extend([corr_row['A'].iloc[0], corr_row['B'].iloc[0]])
            else:
                # Fallback to generic guesses if correlation fit failed
                initial_guesses_combined_model.extend([1.0, 0.01]) 

        # Iterate through each tower for Week 43, 2024
        for tower_name, tower_dir in base_directories.items():
            print(f"\nModeling NEE for {tower_name} (Week 43, 2024)...")
            
            # Load the original processed Eddy Covariance data for the specific tower and year
            processed_ec_data_path = os.path.join(tower_dir, f"processed_hesseflux_{year}.csv")
            if not os.path.exists(processed_ec_data_path):
                print(f"Processed Eddy Covariance data not found for {tower_name} - 2024. Skipping modeling.")
                continue
            
            df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
            
            # Filter for Week 43 data from the original processed EC data
            df_ec_week43 = df_ec[df_ec.index.isocalendar().week == 43].copy()
            
            # Get corresponding satellite data for this tower and week
            tower_satellite_data = satellite_data_weekly[(satellite_data_weekly['Tower'] == tower_name) &
                                                         (satellite_data_weekly['Year'] == 2024)].copy()

            # Ensure we have data for modeling
            if df_ec_week43.empty or tower_satellite_data.empty:
                print(f"No sufficient data for {tower_name} (Week 43, 2024) to model NEE.")
                continue

            # Get the single EVI and MODIS_PPFD value for this tower and week
            current_evi = tower_satellite_data[best_vi].iloc[0]
            current_modis_ppfd = tower_satellite_data['MODIS_PPFD'].iloc[0]

            # Create X_data for the model: [PPFD, EVI] for each 30-min interval
            # Use the original 30-min PPFD from the EC data, and the weekly EVI
            X_data_model = df_ec_week43[['PPFD']].copy()
            X_data_model['EVI'] = current_evi
            X_data_model = X_data_model.values

            y_observed_model = df_ec_week43['NEE'].values

            # Drop rows where y_observed_model is NaN (missing NEE data)
            valid_indices = ~np.isnan(y_observed_model)
            X_data_model = X_data_model[valid_indices]
            y_observed_model = y_observed_model[valid_indices]

            if len(y_observed_model) < len(initial_guesses_combined_model) / 2: # Need enough data points for fitting
                print(f"Not enough valid data points for {tower_name} (Week 43, 2024) to fit NEE model.")
                continue

            try:
                # Fit the combined NEE model
                params_nee_model, pcov_nee_model = curve_fit(nee_model, X_data_model, y_observed_model, p0=initial_guesses_combined_model, maxfev=10000)

                # Predict NEE using the fitted model
                y_predicted_model = nee_model(X_data_model, *params_nee_model)

                # Evaluate model performance
                r2_nee_model = r2_score(y_observed_model, y_predicted_model)
                rmse_nee_model = np.sqrt(mean_squared_error(y_observed_model, y_predicted_model))

                print(f"NEE Model Fitting Results for {tower_name} (Week 43, 2024):")
                print(f"  Fitted Parameters (A_a, B_a, A_b, B_b, A_c, B_c): {params_nee_model}")
                print(f"  R² of NEE Model: {r2_nee_model:.2f}")
                print(f"  RMSE of NEE Model: {rmse_nee_model:.2f}")

                # Visualize Modeled vs. Observed NEE
                plot_dir = os.path.join(tower_dir, "plot")
                os.makedirs(plot_dir, exist_ok=True)

                plt.figure(figsize=(8, 8))
                plt.scatter(y_observed_model, y_predicted_model, alpha=0.6)
                plt.plot([min(y_observed_model), max(y_observed_model)], [min(y_observed_model), max(y_observed_model)], '--r', label='1:1 line')
                plt.xlabel('Observed NEE (µmol m⁻² s⁻¹)')
                plt.ylabel('Modeled NEE (µmol m⁻² s⁻¹)')
                plt.title(f'Observed vs. Modeled NEE ({tower_name}, Week 43, 2024, {best_vi})')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'NEE_model_observed_vs_predicted_{tower_name}_Week43_{best_vi}.png'))
                plt.close()
                print(f"Observed vs. Modeled NEE plot saved to {plot_dir}")

            except RuntimeError as e:
                print(f"NEE model fitting failed for {tower_name} (Week 43, 2024): {e}")
            except ValueError as e:
                print(f"NEE model fitting failed for {tower_name} (Week 43, 2024) (invalid input): {e}")

    # --- Plot Weekly Mean NEE Comparison for Week 43 ---
    plot_single_weekly_mean_nee_comparison_all_towers(base_directories, best_vi, correlation_df, gurteen_dir, target_week=43)
