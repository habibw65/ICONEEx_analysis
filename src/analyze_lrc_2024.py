import pandas as pd
import numpy as np
import os

output_gurteen_dir = "/Users/habibw/Documents/Gurteen"
lrc_df_2024_path = os.path.join(output_gurteen_dir, "lrc_parameters_2024.csv")

if os.path.exists(lrc_df_2024_path):
    lrc_df_2024 = pd.read_csv(lrc_df_2024_path)

    # Filter for successful fits and valid R2 values for 2024
    successful_fits_2024 = lrc_df_2024[lrc_df_2024['Fit_Status'] == 'Success'].copy()
    successful_fits_2024 = successful_fits_2024.dropna(subset=['R2'])

    if not successful_fits_2024.empty:
        # Group by Week and calculate the mean R2
        weekly_avg_r2_2024 = successful_fits_2024.groupby('Week')['R2'].mean()
        
        # Find the week with the highest average R2
        most_usable_week_2024 = weekly_avg_r2_2024.idxmax()
        max_avg_r2_2024 = weekly_avg_r2_2024.max()

        print(f"For 2024, the week with the most usable tower data (highest average R²): Week {most_usable_week_2024} (Average R²: {max_avg_r2_2024:.2f})")
    else:
        print("No successful LRC fits with valid R² values found for 2024.")
else:
    print(f"LRC parameters CSV for 2024 not found at {lrc_df_2024_path}")
