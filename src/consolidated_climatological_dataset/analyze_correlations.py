import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def analyze_correlations(consolidated_data_path, output_dir, selected_weeks):
    print(f"Loading consolidated data from {consolidated_data_path}...")
    df_climatology = pd.read_csv(consolidated_data_path)

    all_correlation_results = []

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

    for tower_name in df_climatology['Tower'].unique():
        print(f"Analyzing correlations for {tower_name}...")
        tower_df = df_climatology[df_climatology['Tower'] == tower_name].copy()

        # Filter for selected weeks
        tower_df_selected_weeks = tower_df[tower_df['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1).isin(selected_weeks)].copy()

        if tower_df_selected_weeks.empty:
            print(f"No data for selected weeks for {tower_name}. Skipping correlation analysis.")
            continue

        # Create a figure for each tower with subplots for each VI
        fig, axes = plt.subplots(len(selected_weeks), len(vi_columns), figsize=(len(vi_columns) * 3, len(selected_weeks) * 3), squeeze=False)
        fig.suptitle(f'NEE vs. Vegetation Indices - {tower_name} (Selected Weeks)', fontsize=16, y=0.99)

        for i, week_num in enumerate(selected_weeks):
            weekly_data = tower_df_selected_weeks[tower_df_selected_weeks['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1) == week_num].copy()
            
            for j, vi_col in enumerate(vi_columns):
                ax = axes[i, j]

                # Drop NaNs for correlation calculation and plotting
                plot_data = weekly_data.dropna(subset=['NEE', vi_col])

                if len(plot_data) < 2: # Need at least 2 points for correlation
                    ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=8, color='gray')
                    ax.set_title(f'Week {week_num} - {vi_col}', fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    continue

                x = plot_data[vi_col]
                y = plot_data['NEE']

                # Calculate Pearson correlation
                pearson_r, pearson_p = pearsonr(x, y)
                
                # Calculate Spearman correlation
                spearman_rho, spearman_p = spearmanr(x, y)

                all_correlation_results.append({
                    'Tower': tower_name,
                    'Week': week_num,
                    'VI': vi_col,
                    'Pearson_R': pearson_r,
                    'Pearson_P': pearson_p,
                    'Spearman_Rho': spearman_rho,
                    'Spearman_P': spearman_p
                })

                ax.scatter(x, y, s=5, alpha=0.6)
                ax.set_title(f'Week {week_num} - {vi_col}', fontsize=10)
                ax.set_xlabel(vi_col, fontsize=8)
                ax.set_ylabel('NEE', fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.6)

                # Display correlation coefficients
                corr_text = f'P_R={pearson_r:.2f}\nS_R={spearman_rho:.2f}'
                ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', 
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=0.2, alpha=0.7))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
        plt.savefig(os.path.join(output_dir, f'Correlations_{tower_name}_SelectedWeeks.png'))
        plt.close()
        print(f"Saved Correlations_{tower_name}_SelectedWeeks.png to {output_dir}")

    # Save all correlation results to a single CSV
    correlation_df = pd.DataFrame(all_correlation_results)
    correlation_output_path = os.path.join(output_dir, "all_towers_weekly_correlations.csv")
    correlation_df.to_csv(correlation_output_path, index=False)
    print(f"All towers weekly correlation results saved to: {correlation_output_path}")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    consolidated_data_file = os.path.join(consolidated_dataset_dir, "consolidated_half_hourly_climatology_data.csv")
    
    selected_weeks_to_plot = [14, 37, 38, 40, 43]

    analyze_correlations(consolidated_data_file, consolidated_dataset_dir, selected_weeks_to_plot)
