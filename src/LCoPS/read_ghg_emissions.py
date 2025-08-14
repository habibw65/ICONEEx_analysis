import pandas as pd

excel_file_path = '/Users/habibw/Documents/LCoPS/GHG_emissions.xlsx'

try:
    df = pd.read_excel(excel_file_path)
    print("DataFrame Head:")
    print(df.head())
    print("\nDataFrame Columns:")
    print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
