import pandas as pd
import os

output_gurteen_dir = "/Users/habibw/Documents/Gurteen"
lrc_df_2024_path = os.path.join(output_gurteen_dir, "lrc_parameters_2024.csv")

if os.path.exists(lrc_df_2024_path):
    lrc_df_2024 = pd.read_csv(lrc_df_2024_path)

    # Filter for Week 43 in 2024 and successful fits
    week_43_data = lrc_df_2024[(lrc_df_2024['Week'] == 43) & (lrc_df_2024['Fit_Status'] == 'Success')].copy()
    week_43_data = week_43_data.dropna(subset=['R2'])

    if not week_43_data.empty:
        print(f"LRC Parameters and R² for Week 43, 2024 (Successful Fits):")
        print(week_43_data[['Tower', 'DataType', 'a', 'b', 'c', 'R2', 'Fit_Status']])
    else:
        print("No successful LRC fits found for Week 43, 2024.")
else:
    print(f"LRC parameters CSV for 2024 not found at {lrc_df_2024_path}")
