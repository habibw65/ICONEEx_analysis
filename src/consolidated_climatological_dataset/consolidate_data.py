import pandas as pd
import numpy as np
import os

def consolidate_data(): # Removed gurteen_dir parameter as paths are absolute
    # Define absolute base directories for all towers
    tower_base_dirs = {
        "Gurteen": "/Users/habibw/Documents/Gurteen",
        "Athenry": "/Users/habibw/Documents/Athenry",
        "JC1": "/Users/habibw/Documents/JC1",
        "JC2": "/Users/habibw/Documents/JC2",
        "Timoleague": "/Users/habibw/Documents/Timoleague",
        "Lullymore": "/Users/habibw/Documents/Lullymore",
        "Clarabog": "/Users/habibw/Documents/Clarabog"
    }

    all_tower_data = []
    # Load and process tower data
    for tower_name, base_dir in tower_base_dirs.items():
        print(f"Loading processed data for {tower_name}...")
        for year in range(2018, 2025): # Assuming data from 2018 to 2024
            processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
            if os.path.exists(processed_file_path):
                try:
                    # Read CSV without setting index_col initially
                    df = pd.read_csv(processed_file_path, parse_dates=['DateTime'])
                    df['Tower'] = tower_name
                    df['DayOfYear'] = df['DateTime'].dt.dayofyear
                    df['Hour'] = df['DateTime'].dt.hour
                    df['Minute'] = df['DateTime'].dt.minute
                    all_tower_data.append(df)
                except Exception as e:
                    print(f"Error loading {processed_file_path}: {e}")
            else:
                print(f"Processed file not found for {tower_name} - {year}: {processed_file_path}")

    if not all_tower_data:
        print("No tower data found for consolidation.")
        return
    
    combined_tower_df = pd.concat(all_tower_data, ignore_index=True)
    print(f"Combined tower data shape: {combined_tower_df.shape}")
    print(f"Combined tower data columns: {combined_tower_df.columns.tolist()}")
    print(f"Combined tower data head:\n{combined_tower_df.head()}")

    # Load and process satellite VI data
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    vi_types = ['evi', 'ndmi', 'ndvi', 'savi']
    all_satellite_data = []

    for vi_type in vi_types:
        vi_file = os.path.join(satellite_data_dir, f'modis_{vi_type}_daily.csv')
        if os.path.exists(vi_file):
            print(f"Loading satellite VI data: {os.path.basename(vi_file)}...")
            df_vi = pd.read_csv(vi_file)
            df_vi['DateTime'] = pd.to_datetime(df_vi['DateTime'])
            df_vi['DayOfYear'] = df_vi['DateTime'].dt.dayofyear
            all_satellite_data.append(df_vi)
        else:
            print(f"Warning: Satellite VI file not found: {vi_file}. Skipping.")

    if not all_satellite_data:
        print("No satellite VI data found for consolidation.")
        # If no satellite data, proceed with only tower data
        merged_full_df = combined_tower_df
    else:
        # Merge all satellite dataframes into a single long format dataframe
        combined_satellite_df = pd.concat(all_satellite_data, ignore_index=True)
        combined_satellite_df = combined_satellite_df[combined_satellite_df['Tower'].isin(tower_base_dirs.keys())]
        print(f"Combined satellite data shape (filtered): {combined_satellite_df.shape}")
        print(f"Combined satellite data columns: {combined_satellite_df.columns.tolist()}")
        print(f"Combined satellite data head:\n{combined_satellite_df.head()}")

        # Merge combined_tower_df and combined_satellite_df
        merged_full_df = pd.merge(combined_tower_df, combined_satellite_df, on=['DateTime', 'Tower', 'DayOfYear'], how='outer', suffixes=('_tower', '_satellite'))
    
    print(f"Merged full data shape: {merged_full_df.shape}")

    # Create half-hourly climatology (average for each DayOfYear, Hour, Minute for each Tower)
    # Exclude non-numeric columns before calculating mean
    numeric_cols = merged_full_df.select_dtypes(include=np.number).columns.tolist()
    
    # Ensure 'DayOfYear', 'Hour', 'Minute' are in numeric_cols if they are not already
    for col in ['DayOfYear', 'Hour', 'Minute']:
        if col not in numeric_cols:
            numeric_cols.append(col)

    climatology_df = merged_full_df.groupby(['Tower', 'DayOfYear', 'Hour', 'Minute'], as_index=False)[numeric_cols].mean()
    print(f"Climatology data shape (before imputation): {climatology_df.shape}")

    # Impute missing values in the climatology using interpolation
    print("Imputing missing values in climatology data...")
    climatology_df_imputed = climatology_df.copy()
    
    # Sort by Tower, DayOfYear, Hour, Minute for correct interpolation
    climatology_df_imputed = climatology_df_imputed.sort_values(by=['Tower', 'DayOfYear', 'Hour', 'Minute'])

    for tower in climatology_df_imputed['Tower'].unique():
        # Select only numeric columns for interpolation
        # Exclude 'DayOfYear', 'Hour', 'Minute' from interpolation as they are keys
        cols_to_interpolate = [col for col in numeric_cols if col not in ['DayOfYear', 'Hour', 'Minute']]
        
        subset = climatology_df_imputed[climatology_df_imputed['Tower'] == tower].set_index(['DayOfYear', 'Hour', 'Minute'])
        
        # Apply interpolation
        # Use .copy() to avoid SettingWithCopyWarning
        interpolated_values = subset[cols_to_interpolate].interpolate(method='linear', limit_direction='both').copy()
        
        # Update the original DataFrame with interpolated values
        climatology_df_imputed.loc[climatology_df_imputed['Tower'] == tower, cols_to_interpolate] = interpolated_values.values

    # Final fillna for any remaining NaNs (e.g., if a whole column was NaN for a tower for all years)
    climatology_df_imputed = climatology_df_imputed.fillna(0) # Filling with 0 as a general placeholder
    
    print(f"Imputed climatology data shape: {climatology_df_imputed.shape}")
    print(f"Imputed climatology data NaNs: {climatology_df_imputed.isnull().sum().sum()}")

    # Save the consolidated climatology dataset
    output_file = "/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv"
    climatology_df_imputed.to_csv(output_file, index=False)
    print(f"Consolidated half-hourly climatology data saved to: {output_file}")

if __name__ == '__main__':
    consolidate_data()