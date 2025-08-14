import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Mitscherlich function for Light Response Curve
def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    if np.isclose(denom, 0):
        return np.full_like(ppfd, np.nan)

    exp_arg = (-c * ppfd) / denom
    exp_arg = np.clip(exp_arg, -700, 700)

    return 1 - denom * (1 - np.exp(exp_arg)) + b

def generate_weekly_lrcs(consolidated_data_path, output_dir):
    print(f"Loading consolidated data from {consolidated_data_path}...")
    df_climatology = pd.read_csv(consolidated_data_path)

    all_lrc_params = []

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5
    })

    for tower_name in df_climatology['Tower'].unique():
        print(f"Generating weekly LRCs for {tower_name}...")
        tower_output_dir = os.path.join(output_dir, f'{tower_name}_LRC_Plots')
        os.makedirs(tower_output_dir, exist_ok=True)

        tower_df = df_climatology[df_climatology['Tower'] == tower_name].copy()

        # Ensure DayOfYear, Hour, Minute are sorted for consistent weekly grouping
        tower_df = tower_df.sort_values(by=['DayOfYear', 'Hour', 'Minute'])

        # Calculate week number based on DayOfYear (approximate, for grouping)
        # This assumes a consistent mapping of DayOfYear to Week across all years
        tower_df['Week'] = tower_df['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1)

        for week_num in sorted(tower_df['Week'].unique()):
            weekly_df = tower_df[tower_df['Week'] == week_num].copy()

            # Filter for daytime data and drop NaNs for fitting
            lrc_data = weekly_df[weekly_df['PPFD'] > 10].dropna(subset=['NEE', 'PPFD'])

            # Skip if not enough data points for fitting
            if len(lrc_data) < 3: # Need at least 3 points for 3 parameters
                all_lrc_params.append({
                    'Tower': tower_name,
                    'Week': week_num,
                    'a': np.nan, 'b': np.nan, 'c': np.nan,
                    'R2': np.nan,
                    'Fit_Status': 'Not enough data'
                })
                continue

            try:
                # Initial guesses for a, b, c
                b_guess = lrc_data[lrc_data['PPFD'] < 20]['NEE'].mean()
                if np.isnan(b_guess): b_guess = 5.0
                a_guess = 20.0
                c_guess = 0.05

                p0 = [a_guess, b_guess, c_guess]
                bounds = ([0.1, -50, 0.001], [100, 50, 1.0]) # [a_min, b_min, c_min], [a_max, b_max, c_max]

                params, pcov = curve_fit(mitscherlich_lrc, lrc_data['PPFD'], lrc_data['NEE'], p0=p0, bounds=bounds, maxfev=5000)
                
                # Calculate R-squared
                y_observed = lrc_data['NEE'].values
                y_predicted = mitscherlich_lrc(lrc_data['PPFD'].values, *params)
                ss_res = np.sum((y_observed - y_predicted)**2)
                ss_tot = np.sum((y_observed - np.mean(y_observed))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                all_lrc_params.append({
                    'Tower': tower_name,
                    'Week': week_num,
                    'a': params[0], 'b': params[1], 'c': params[2],
                    'R2': r_squared,
                    'Fit_Status': 'Success'
                })

                # Generate plot
                plt.figure(figsize=(10, 6))
                plt.scatter(lrc_data['PPFD'], lrc_data['NEE'], label='Data', alpha=0.6, s=10)
                
                ppfd_fit = np.linspace(0, lrc_data['PPFD'].max() * 1.1, 100)
                nee_fit = mitscherlich_lrc(ppfd_fit, *params)
                plt.plot(ppfd_fit, nee_fit, color='red', linewidth=2, label='Fitted Curve')

                param_text = f'a={params[0]:.2f}\nb={params[1]:.2f}\nc={params[2]:.3f}\nR²={r_squared:.2f}'
                plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8), fontsize=10)

                plt.title(f'LRC for {tower_name} - Week {week_num}')
                plt.xlabel('PPFD (µmol m⁻² s⁻¹)')
                plt.ylabel('NEE (µmol m⁻² s⁻¹)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(tower_output_dir, f'LRC_{tower_name}_Week{week_num}.png'))
                plt.close()

            except (RuntimeError, ValueError) as e:
                all_lrc_params.append({
                    'Tower': tower_name,
                    'Week': week_num,
                    'a': np.nan, 'b': np.nan, 'c': np.nan,
                    'R2': np.nan,
                    'Fit_Status': f'Fit failed: {e}'
                })
                print(f"LRC fit failed for {tower_name} - Week {week_num}: {e}")

    # Save all LRC parameters to a single CSV
    lrc_params_df = pd.DataFrame(all_lrc_params)
    lrc_params_output_path = os.path.join(output_dir, "all_towers_weekly_lrc_parameters.csv")
    lrc_params_df.to_csv(lrc_params_output_path, index=False)
    print(f"All towers weekly LRC parameters saved to: {lrc_params_output_path}")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    consolidated_data_file = os.path.join(consolidated_dataset_dir, "consolidated_half_hourly_climatology_data.csv")
    
    generate_weekly_lrcs(consolidated_data_file, consolidated_dataset_dir)
