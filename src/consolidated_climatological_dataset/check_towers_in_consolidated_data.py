import pandas as pd
import os

file_path = "/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    unique_towers = df['Tower'].unique()
    print("Unique towers in consolidated_half_hourly_climatology_data.csv:")
    for tower in unique_towers:
        print(f"- {tower}")
else:
    print(f"Error: File not found at {file_path}")
