import pandas as pd

file_path = "/Users/habibw/Documents/Gurteen/consolidated_climatology_data.csv"
df = pd.read_csv(file_path)

nan_counts = df.isnull().sum()

print("NaN counts per column in consolidated_climatology_data.csv:")
print(nan_counts[nan_counts > 0])

if nan_counts.sum() == 0:
    print("No NaN values found in the entire DataFrame.")
else:
    print(f"Total NaN values in DataFrame: {nan_counts.sum()}")
