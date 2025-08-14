import pandas as pd
import os

def merge_and_clean_satellite_vis(satellite_data_dir):
    vi_types = ['evi', 'ndmi', 'ndvi', 'savi']

    for vi_type in vi_types:
        existing_file = os.path.join(satellite_data_dir, f'modis_{vi_type}_daily.csv')
        new_file = os.path.join(satellite_data_dir, f'modis_{vi_type}_daily_clarabog_lullymore.csv')

        if os.path.exists(existing_file) and os.path.exists(new_file):
            print(f"Merging {os.path.basename(new_file)} into {os.path.basename(existing_file)}...")
            
            df_existing = pd.read_csv(existing_file)
            df_new = pd.read_csv(new_file)

            # Rename timestamp column to 'DateTime' for both existing and new DataFrames
            if 'system:time_start' in df_existing.columns:
                df_existing = df_existing.rename(columns={'system:time_start': 'DateTime'})
            if 'system:time_start' in df_new.columns:
                df_new = df_new.rename(columns={'system:time_start': 'DateTime'})
            
            # Convert DateTime to datetime objects for proper merging
            df_existing['DateTime'] = pd.to_datetime(df_existing['DateTime'])
            df_new['DateTime'] = pd.to_datetime(df_new['DateTime'])

            # Melt DataFrames to long format before concatenating
            df_existing_melted = df_existing.melt(id_vars=['DateTime'], var_name='Tower', value_name=vi_type.upper())
            df_new_melted = df_new.melt(id_vars=['DateTime'], var_name='Tower', value_name=vi_type.upper())

            # Concatenate and drop duplicates based on 'DateTime' and 'Tower'
            df_merged = pd.concat([df_existing_melted, df_new_melted]).drop_duplicates(subset=['DateTime', 'Tower']).sort_values(by=['Tower', 'DateTime'])
            
            # Pivot back to wide format for saving, if desired, or keep long format
            # For now, let's keep it in long format as it's more flexible for analysis
            df_merged.to_csv(existing_file, index=False)
            print(f"Successfully merged and updated {os.path.basename(existing_file)}.")

            # Delete the new file
            os.remove(new_file)
            print(f"Deleted {os.path.basename(new_file)}.")
        elif not os.path.exists(existing_file):
            print(f"Warning: Existing file {os.path.basename(existing_file)} not found. Skipping merge for {vi_type}.")
        elif not os.path.exists(new_file):
            print(f"Warning: New file {os.path.basename(new_file)} not found. Skipping merge for {vi_type}.")

if __name__ == '__main__':
    satellite_data_directory = "/Users/habibw/Documents/satellite derived data"
    merge_and_clean_satellite_vis(satellite_data_directory)
