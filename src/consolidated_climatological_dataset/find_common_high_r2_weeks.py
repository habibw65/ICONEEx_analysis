import pandas as pd
import os

def find_common_high_r2_weeks(lrc_params_path):
    print(f"Loading LRC parameters from {lrc_params_path}...")
    df_lrc_params = pd.read_csv(lrc_params_path)

    # Filter for successful fits and R2 >= 0.70
    high_r2_fits = df_lrc_params[(df_lrc_params['Fit_Status'] == 'Success') & 
                                 (df_lrc_params['R2'] >= 0.70)].copy()

    if high_r2_fits.empty:
        print("No weeks found with R2 >= 0.70 for any tower.")
        return

    # Get the total number of unique towers
    all_towers = df_lrc_params['Tower'].unique()
    num_towers = len(all_towers)
    print(f"Total unique towers: {num_towers}")

    # Group by week and count how many towers meet the criteria for each week
    weekly_tower_counts = high_r2_fits.groupby('Week')['Tower'].nunique()

    # Find weeks where at least 5 towers meet the criteria
    common_high_r2_weeks = weekly_tower_counts[weekly_tower_counts >= 5].index.tolist()

    if common_high_r2_weeks:
        print(f"Weeks with R2 >= 0.70 for at least 5 towers: {common_high_r2_weeks}")
    else:
        print("No weeks found where R2 >= 0.70 for at least 5 towers.")

if __name__ == '__main__':
    consolidated_dataset_dir = "/Users/habibw/Documents/consolidated climatological dataset"
    lrc_params_file = os.path.join(consolidated_dataset_dir, "all_towers_weekly_lrc_parameters.csv")
    
    find_common_high_r2_weeks(lrc_params_file)