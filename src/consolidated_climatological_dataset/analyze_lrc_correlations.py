import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, linregress

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

# Mitscherlich function (from previous scripts, for reference if needed)
def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    if np.isclose(denom, 0):
        return np.full_like(ppfd, np.nan)
    exp_arg = (-c * ppfd) / denom
    exp_arg = np.clip(exp_arg, -700, 700)
    return 1 - denom * (1 - np.exp(exp_arg)) + b

def analyze_lrc_correlations(consolidated_data_path, output_dir, selected_weeks):
    print(f"Loading consolidated data from {consolidated_data_path}...")
    df_climatology = pd.read_csv(consolidated_data_path)

    all_best_fits = []

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

    for tower_name in df_climatology['Tower'].unique():
        print(f"Analyzing LRC correlations for {tower_name}...")
        tower_df = df_climatology[df_climatology['Tower'] == tower_name].copy()

        # Create a figure for each tower with subplots for each selected week
        n_cols = 3 # Max 3 columns
        n_rows = int(np.ceil(len(selected_weeks) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f'NEE vs. VIs - Best Fit Correlations for {tower_name} (Selected Weeks)', fontsize=16, y=0.99)

        for i, week_num in enumerate(selected_weeks):
            ax = axes[i]
            weekly_data = tower_df[tower_df['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1) == week_num].copy()
            
            best_r2_week = -np.inf
            best_fit_info = None

            for vi_col in vi_columns:
                # Drop NaNs for correlation calculation and plotting
                plot_data = weekly_data.dropna(subset=['NEE', vi_col])

                if len(plot_data) < 2: # Need at least 2 points for correlation
                    continue

                x_data = plot_data[vi_col].values
                y_data = plot_data['NEE'].values

                for corr_type_name, corr_func in correlation_types.items():
                    try:
                        if corr_type_name == 'Linear':
                            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                            r_squared = r_value**2
                            fitted_params = (slope, intercept)
                            y_predicted = corr_func(x_data, *fitted_params)
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

                        if r_squared > best_r2_week:
                            best_r2_week = r_squared
                            best_fit_info = {
                                'Tower': tower_name,
                                'Week': week_num,
                                'VI': vi_col,
                                'Correlation_Type': corr_type_name,
                                'R2': r_squared,
                                'Parameters': fitted_params,
                                'Fit_Status': 'Success'
                            }

                    except (RuntimeError, ValueError, TypeError) as e:
                        # print(f"Fit failed for {tower_name} Week {week_num} {vi_col} {corr_type_name}: {e}")
                        pass # Skip failed fits
            
            if best_fit_info: # If a successful fit was found for this week
                all_best_fits.append(best_fit_info)

                # Plotting the best fit
                ax.scatter(plot_data[best_fit_info['VI']], plot_data['NEE'], s=5, alpha=0.6, label='Data')
                
                x_fit = np.linspace(plot_data[best_fit_info['VI']].min(), plot_data[best_fit_info['VI']].max(), 100)
                y_fit = correlation_types[best_fit_info['Correlation_Type']](x_fit, *best_fit_info['Parameters'])
                ax.plot(x_fit, y_fit, color='red', linewidth=1, label='Best Fit')

                corr_text = f"VI: {best_fit_info['VI']}\nType: {best_fit_info['Correlation_Type']}\nR²={best_fit_info['R2']:.2f}"
                ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', 
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=0.2, alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No Best Fit', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=8, color='gray')

            ax.set_title(f'Week {week_num}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('VI Value', fontsize=8)
            ax.set_ylabel('NEE', fontsize=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'Best_Correlations_{tower_name}_SelectedWeeks.png'))
        plt.close()
        print(f"Saved Best_Correlations_{tower_name}_SelectedWeeks.png to {output_dir}")

    # Save all best fit correlation results to a single CSV
    best_fits_df = pd.DataFrame(all_best_fits)
    best_fits_output_path = os.path.join(output_dir, "all_towers_best_lrc_correlations.csv")
    best_fits_df.to_csv(best_fits_output_path, index=False)
    print(f"All towers best LRC correlation results saved to: {best_fits_output_path}")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    consolidated_data_file = os.path.join(consolidated_dataset_dir, "consolidated_half_hourly_climatology_data.csv")
    
    selected_weeks_to_plot = [14, 37, 38, 40, 43]

    analyze_lrc_correlations(consolidated_data_file, consolidated_dataset_dir, selected_weeks_to_plot)
