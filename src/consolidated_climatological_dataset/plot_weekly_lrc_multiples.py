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

def plot_weekly_lrc_multiples(consolidated_data_path, lrc_params_path, output_base_dir):
    print(f"Loading consolidated data from {consolidated_data_path}...")
    df_climatology = pd.read_csv(consolidated_data_path)
    
    print(f"Loading LRC parameters from {lrc_params_path}...")
    df_lrc_params = pd.read_csv(lrc_params_path)

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

    for tower_name in df_climatology['Tower'].unique():
        print(f"Generating weekly LRC multiple plots for {tower_name}...")
        tower_output_dir = os.path.join(output_base_dir, f'{tower_name}_LRC_Plots')
        os.makedirs(tower_output_dir, exist_ok=True)

        tower_df = df_climatology[df_climatology['Tower'] == tower_name].copy()
        
        # Ensure DayOfYear, Hour, Minute are sorted for consistent weekly grouping
        tower_df = tower_df.sort_values(by=['DayOfYear', 'Hour', 'Minute'])

        # Calculate week number based on DayOfYear (approximate, for grouping)
        tower_df['Week'] = tower_df['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1)

        unique_weeks = sorted(tower_df['Week'].unique())
        n_weeks = len(unique_weeks)
        
        # Determine grid size for subplots (e.g., 8 rows, 7 columns for 52 weeks)
        n_cols = 7
        n_rows = int(np.ceil(n_weeks / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharex=True, sharey=True)
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

        fig.suptitle(f'Weekly Light Response Curves - {tower_name}', fontsize=16, y=0.99)

        for i, week_num in enumerate(unique_weeks):
            ax = axes[i]
            weekly_data = tower_df[tower_df['Week'] == week_num].copy()

            # Filter for daytime data and drop NaNs for fitting
            lrc_data = weekly_data[weekly_data['PPFD'] > 10].dropna(subset=['NEE', 'PPFD'])

            ax.scatter(lrc_data['PPFD'], lrc_data['NEE'], s=5, alpha=0.6, label='Data')

            # Retrieve fitted parameters for this week and tower
            params_row = df_lrc_params[(df_lrc_params['Tower'] == tower_name) &
                                       (df_lrc_params['Week'] == week_num) &
                                       (df_lrc_params['Fit_Status'] == 'Success')]
            
            if not params_row.empty:
                a, b, c = params_row['a'].iloc[0], params_row['b'].iloc[0], params_row['c'].iloc[0]
                r_squared = params_row['R2'].iloc[0]

                ppfd_fit = np.linspace(0, lrc_data['PPFD'].max() * 1.1, 50) # Fewer points for faster plotting
                nee_fit = mitscherlich_lrc(ppfd_fit, a, b, c)
                ax.plot(ppfd_fit, nee_fit, color='red', linewidth=1, label='Fitted Curve')

                param_text = f'a={a:.2f}\nb={b:.2f}\nc={c:.3f}\nR²={r_squared:.2f}'
                ax.text(0.98, 0.02, param_text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', 
                        fontsize=6, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=0.2, alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=8, color='red')

            ax.set_title(f'Week {week_num}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('PPFD', fontsize=8)
            ax.set_ylabel('NEE', fontsize=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
        plt.savefig(os.path.join(tower_output_dir, f'Weekly_LRC_Multiples_{tower_name}.png'))
        plt.close()
        print(f"Saved Weekly_LRC_Multiples_{tower_name}.png to {tower_output_dir}")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    consolidated_data_file = os.path.join(consolidated_dataset_dir, "consolidated_half_hourly_climatology_data.csv")
    lrc_params_file = os.path.join(consolidated_dataset_dir, "all_towers_weekly_lrc_parameters.csv")
    
    plot_weekly_lrc_multiples(consolidated_data_file, lrc_params_file, consolidated_dataset_dir)
