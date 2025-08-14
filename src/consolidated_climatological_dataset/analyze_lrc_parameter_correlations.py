import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Define correlation functions
def linear_func(x, m, c):
    return m * x + c

def exponential_func(x, A, B):
    return A * np.exp(B * x)

def logarithmic_func(x, A, B):
    # Ensure x is positive for logarithm
    x = np.where(x > 0, x, np.nan) # Replace non-positive x with NaN
    return A * np.log(x) + B

def power_func(x, A, B):
    # Ensure x is positive for power function
    x = np.where(x > 0, x, np.nan) # Replace non-positive x with NaN
    return A * (x**B)

def analyze_lrc_parameter_correlations(consolidated_data_path, lrc_params_path, output_dir):
    print(f"Loading consolidated data from {consolidated_data_path}...")
    df_climatology = pd.read_csv(consolidated_data_path)
    
    print(f"Loading LRC parameters from {lrc_params_path}...")
    df_lrc_params = pd.read_csv(lrc_params_path)

    all_best_lrc_param_fits = []

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.0
    })

    vi_columns = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    correlation_types = {
        'Linear': linear_func,
        'Exponential': exponential_func,
        'Logarithmic': logarithmic_func,
        'Power': power_func
    }

    # Prepare weekly average VI data
    # Group by Tower and Week, then average the VIs
    weekly_vi_data = df_climatology.groupby(['Tower', 'DayOfYear']).mean().reset_index()
    weekly_vi_data['Week'] = weekly_vi_data['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1)
    weekly_vi_data = weekly_vi_data.groupby(['Tower', 'Week'])[vi_columns].mean().reset_index()

    for tower_name in df_climatology['Tower'].unique():
        print(f"Analyzing LRC parameter correlations for {tower_name}...")
        
        # Filter LRC parameters for the current tower and successful fits
        tower_lrc_params = df_lrc_params[(df_lrc_params['Tower'] == tower_name) & 
                                         (df_lrc_params['Fit_Status'] == 'Success')].copy()
        
        # Merge LRC parameters with weekly VI data
        merged_data = pd.merge(tower_lrc_params, weekly_vi_data, on=['Tower', 'Week'], how='inner')

        if merged_data.empty:
            print(f"No merged data for LRC parameter correlation analysis for {tower_name}. Skipping.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False) # One row, three columns for a, b, c
        axes = axes.flatten()

        fig.suptitle(f'Best LRC Parameter Correlations for {tower_name}', fontsize=16, y=1.02)

        lrc_parameters = ['a', 'b', 'c']
        for param_idx, lrc_param in enumerate(lrc_parameters):
            ax = axes[param_idx]
            
            best_r2_param = -np.inf
            best_fit_info = None
            best_x_data = None
            best_y_data = None

            for vi_col in vi_columns:
                # Drop NaNs for correlation calculation
                plot_data = merged_data.dropna(subset=[lrc_param, vi_col])

                if len(plot_data) < 2: # Need at least 2 points for correlation
                    continue

                x_data = plot_data[vi_col].values
                y_data = plot_data[lrc_param].values

                for corr_type_name, corr_func in correlation_types.items():
                    try:
                        if corr_type_name == 'Linear':
                            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                            r_squared = r_value**2
                            fitted_params = (slope, intercept)
                        else:
                            # Initial guesses for non-linear fits
                            if corr_type_name == 'Exponential':
                                p0 = [np.mean(y_data), 0.01]
                                bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
                            elif corr_type_name == 'Logarithmic':
                                p0 = [1.0, np.mean(y_data)]
                                bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
                                if np.any(x_data <= 0): continue # Log requires positive x
                            elif corr_type_name == 'Power':
                                p0 = [1.0, 1.0]
                                bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
                                if np.any(x_data <= 0): continue # Power requires positive x

                            params, pcov = curve_fit(corr_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
                            y_predicted = corr_func(x_data, *params)
                            ss_res = np.sum((y_data - y_predicted)**2)
                            ss_tot = np.sum((y_data - np.mean(y_data))**2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                            fitted_params = params

                        if r_squared > best_r2_param:
                            best_r2_param = r_squared
                            best_fit_info = {
                                'Tower': tower_name,
                                'LRC_Parameter': lrc_param,
                                'VI': vi_col,
                                'Correlation_Type': corr_type_name,
                                'R2': r_squared,
                                'Parameters': fitted_params,
                                'Fit_Status': 'Success'
                            }
                            best_x_data = x_data
                            best_y_data = y_data

                    except (RuntimeError, ValueError, TypeError) as e:
                        pass # Skip failed fits
            
            if best_fit_info: # If a successful fit was found for this parameter
                all_best_lrc_param_fits.append(best_fit_info)

                # Plotting the best fit
                ax.scatter(best_x_data, best_y_data, s=10, alpha=0.6, label='Data')
                
                x_fit = np.linspace(np.nanmin(best_x_data), np.nanmax(best_x_data), 100)
                y_fit = correlation_types[best_fit_info['Correlation_Type']](x_fit, *best_fit_info['Parameters'])
                ax.plot(x_fit, y_fit, color='red', linewidth=1, label='Best Fit')

                corr_text = f"VI: {best_fit_info['VI']}\nType: {best_fit_info['Correlation_Type']}\nR²={best_fit_info['R2']:.2f}"
                ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', 
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=0.2, alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No Best Fit', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=8, color='gray')

            ax.set_title(f'LRC Parameter {lrc_param}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('VI Value', fontsize=8)
            ax.set_ylabel(f'LRC {lrc_param}', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'Best_LRC_Parameter_Correlations_{tower_name}.png'))
        plt.close()
        print(f"Saved Best_LRC_Parameter_Correlations_{tower_name}.png to {output_dir}")

    # Save all best fit correlation results to a single CSV
    best_fits_df = pd.DataFrame(all_best_lrc_param_fits)
    best_fits_output_path = os.path.join(output_dir, "all_towers_best_lrc_parameter_correlations.csv")
    best_fits_df.to_csv(best_fits_output_path, index=False)
    print(f"All towers best LRC parameter correlation results saved to: {best_fits_output_path}")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    consolidated_data_file = os.path.join(consolidated_dataset_dir, "consolidated_half_hourly_climatology_data.csv")
    lrc_params_file = os.path.join(consolidated_dataset_dir, "all_towers_weekly_lrc_parameters.csv")
    
    analyze_lrc_parameter_correlations(consolidated_data_file, lrc_params_file, consolidated_dataset_dir)
