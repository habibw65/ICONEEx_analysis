import pandas as pd

excel_file_path = '/Users/habibw/Documents/LCoPS/GHG_emissions.xlsx'

try:
    df = pd.read_excel(excel_file_path)
    
    print("Unique values in 'Nutrient status scenario (Estimate 1)':")
    print(df['Nutrient status scenario (Estimate 1)'].unique())
    
    print("\nUnique values in 'Management intensity scenario (Estimate 2)':")
    print(df['Management intensity scenario (Estimate 2)'].unique())
    
    print("\nUnique values in 'Drainage scenario':")
    print(df['Drainage scenario'].unique())

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
