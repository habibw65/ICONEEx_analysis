"""
ERA5 Data Processing for NEE Upscaling

This script processes downloaded ERA5 GRIB files and prepares them for integration
with Light Response Curve parameters and satellite vegetation indices for CO2 flux upscaling.

Key Features:
- Converts ERA5 GRIB files to structured CSV format
- Extracts meteorological variables for tower locations
- Calculates derived variables (VPD, wind speed, etc.)
- Resamples data to match flux tower temporal resolution
- Saves processed data for ML model training

Author: ICONeEx Analysis Team
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Tower coordinates (latitude, longitude)
TOWER_COORDINATES = {
    'Gurteen': (53.2914, -8.2347),
    'Athenry': (53.3031, -8.7389),
    'JC1': (53.2819, -7.9472),
    'JC2': (53.2819, -7.9472),
    'Timoleague': (51.6464, -8.7361),
    'Lullymore': (53.3506, -6.9194),
    'Clarabog': (54.2667, -7.8333)
}

# ERA5 variables and their units
ERA5_VARIABLES = {
    '2m_temperature': {'unit': 'K', 'description': '2-meter temperature'},
    '2m_dewpoint_temperature': {'unit': 'K', 'description': '2-meter dewpoint temperature'},
    'surface_pressure': {'unit': 'Pa', 'description': 'Surface pressure'},
    'total_precipitation': {'unit': 'm', 'description': 'Total precipitation'},
    'surface_solar_radiation_downwards': {'unit': 'J/m²', 'description': 'Surface solar radiation downwards'},
    '10m_u_component_of_wind': {'unit': 'm/s', 'description': '10-meter U wind component'},
    '10m_v_component_of_wind': {'unit': 'm/s', 'description': '10-meter V wind component'},
    'evaporation': {'unit': 'm', 'description': 'Evaporation'},
    'boundary_layer_height': {'unit': 'm', 'description': 'Boundary layer height'}
}

def calculate_derived_variables(df):
    """
    Calculate derived meteorological variables from ERA5 data.
    
    Args:
        df: DataFrame with ERA5 variables
    
    Returns:
        df: DataFrame with additional derived variables
    """
    # Convert temperature from Kelvin to Celsius
    if '2m_temperature' in df.columns:
        df['air_temp_c'] = df['2m_temperature'] - 273.15
    
    if '2m_dewpoint_temperature' in df.columns:
        df['dewpoint_temp_c'] = df['2m_dewpoint_temperature'] - 273.15
    
    # Calculate relative humidity
    if 'air_temp_c' in df.columns and 'dewpoint_temp_c' in df.columns:
        # Using Magnus formula
        def saturation_vapor_pressure(temp_c):
            return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
        
        es_air = saturation_vapor_pressure(df['air_temp_c'])
        es_dew = saturation_vapor_pressure(df['dewpoint_temp_c'])
        df['relative_humidity'] = (es_dew / es_air) * 100
        
        # Calculate Vapor Pressure Deficit (VPD)
        df['vpd'] = es_air - es_dew  # hPa
    
    # Calculate wind speed and direction
    if '10m_u_component_of_wind' in df.columns and '10m_v_component_of_wind' in df.columns:
        df['wind_speed'] = np.sqrt(df['10m_u_component_of_wind']**2 + df['10m_v_component_of_wind']**2)
        df['wind_direction'] = np.arctan2(df['10m_v_component_of_wind'], df['10m_u_component_of_wind']) * 180 / np.pi
        df['wind_direction'] = (df['wind_direction'] + 360) % 360  # Convert to 0-360 degrees
    
    # Convert surface pressure from Pa to hPa
    if 'surface_pressure' in df.columns:
        df['surface_pressure_hpa'] = df['surface_pressure'] / 100
    
    # Convert solar radiation from J/m² to W/m² (assuming hourly data)
    if 'surface_solar_radiation_downwards' in df.columns:
        df['solar_radiation_wm2'] = df['surface_solar_radiation_downwards'] / 3600
    
    # Convert precipitation from m to mm
    if 'total_precipitation' in df.columns:
        df['precipitation_mm'] = df['total_precipitation'] * 1000
    
    # Convert evaporation from m to mm
    if 'evaporation' in df.columns:
        df['evaporation_mm'] = df['evaporation'] * 1000
    
    return df

def find_nearest_grid_point(lat, lon, lats, lons):
    """
    Find the nearest grid point to the given coordinates.
    
    Args:
        lat, lon: Target coordinates
        lats, lons: Grid coordinates
    
    Returns:
        lat_idx, lon_idx: Indices of nearest grid point
    """
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_era5_grib_files(grib_dir, output_dir, start_year=2022, end_year=2024):
    """
    Process ERA5 GRIB files and extract data for tower locations.
    
    Args:
        grib_dir: Directory containing ERA5 GRIB files
        output_dir: Directory to save processed data
        start_year, end_year: Year range to process
    """
    print("Processing ERA5 GRIB files...")
    print(f"Input directory: {grib_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data storage for each tower
    tower_data = {tower: [] for tower in TOWER_COORDINATES.keys()}
    
    # Process each year
    for year in range(start_year, end_year + 1):
        print(f"\\nProcessing year {year}...")
        
        # Find all GRIB files for this year
        grib_pattern = os.path.join(grib_dir, f"era5_*_{year}_*.grib")
        grib_files = glob.glob(grib_pattern)
        
        if not grib_files:
            print(f"No GRIB files found for year {year}")
            continue
        
        print(f"Found {len(grib_files)} GRIB files for {year}")
        
        # Process each GRIB file
        for grib_file in sorted(grib_files):
            try:
                # Extract metadata from filename
                filename = os.path.basename(grib_file)
                parts = filename.split('_')
                if len(parts) >= 5:
                    variable = parts[1]
                    year_str = parts[2]
                    month_str = parts[3]
                    day_str = parts[4].split('.')[0]
                    
                    print(f"Processing {variable} for {year_str}-{month_str}-{day_str}")
                    
                    # Open GRIB file with xarray
                    ds = xr.open_dataset(grib_file, engine='cfgrib')
                    
                    # Get coordinate arrays
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    
                    # Extract data for each tower location
                    for tower, (lat, lon) in TOWER_COORDINATES.items():
                        # Find nearest grid point
                        lat_idx, lon_idx = find_nearest_grid_point(lat, lon, lats, lons)
                        
                        # Extract data for this location
                        tower_ds = ds.isel(latitude=lat_idx, longitude=lon_idx)
                        
                        # Convert to DataFrame
                        tower_df = tower_ds.to_dataframe().reset_index()
                        
                        # Add metadata
                        tower_df['tower'] = tower
                        tower_df['variable'] = variable
                        tower_df['actual_lat'] = lats[lat_idx]
                        tower_df['actual_lon'] = lons[lon_idx]
                        tower_df['target_lat'] = lat
                        tower_df['target_lon'] = lon
                        
                        tower_data[tower].append(tower_df)
                    
                    ds.close()
                    
            except Exception as e:
                print(f"Error processing {grib_file}: {e}")
                continue
    
    # Combine and save data for each tower
    for tower in TOWER_COORDINATES.keys():
        if tower_data[tower]:
            print(f"\\nCombining data for {tower}...")
            
            # Concatenate all data for this tower
            combined_df = pd.concat(tower_data[tower], ignore_index=True)
            
            # Pivot to get variables as columns
            pivot_df = combined_df.pivot_table(
                index=['time', 'tower', 'actual_lat', 'actual_lon', 'target_lat', 'target_lon'],
                columns='variable',
                values=[col for col in combined_df.columns if col not in ['time', 'tower', 'variable', 'actual_lat', 'actual_lon', 'target_lat', 'target_lon']],
                aggfunc='first'
            ).reset_index()
            
            # Flatten column names
            pivot_df.columns = [f"{col[1]}_{col[0]}" if col[0] else col[1] for col in pivot_df.columns]
            
            # Rename time column
            pivot_df = pivot_df.rename(columns={'time_': 'DateTime'})
            
            # Calculate derived variables
            pivot_df = calculate_derived_variables(pivot_df)
            
            # Sort by time
            pivot_df = pivot_df.sort_values('DateTime')
            
            # Save processed data
            output_file = os.path.join(output_dir, f"era5_{tower.lower()}_processed.csv")
            pivot_df.to_csv(output_file, index=False)
            print(f"Saved processed ERA5 data for {tower}: {output_file}")
            print(f"Data shape: {pivot_df.shape}")
            print(f"Date range: {pivot_df['DateTime'].min()} to {pivot_df['DateTime'].max()}")

def resample_era5_to_half_hourly(processed_dir, output_dir):
    """
    Resample ERA5 hourly data to half-hourly to match flux tower data.
    
    Args:
        processed_dir: Directory with processed ERA5 CSV files
        output_dir: Directory to save resampled data
    """
    print("\\nResampling ERA5 data to half-hourly resolution...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for tower in TOWER_COORDINATES.keys():
        input_file = os.path.join(processed_dir, f"era5_{tower.lower()}_processed.csv")
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue
        
        print(f"Processing {tower}...")
        
        # Load data
        df = pd.read_csv(input_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
        
        # Resample to 30-minute intervals using linear interpolation
        df_resampled = df.resample('30T').interpolate(method='linear')
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        # Save resampled data
        output_file = os.path.join(output_dir, f"era5_{tower.lower()}_half_hourly.csv")
        df_resampled.to_csv(output_file, index=False)
        print(f"Saved half-hourly ERA5 data for {tower}: {output_file}")
        print(f"Resampled shape: {df_resampled.shape}")

if __name__ == "__main__":
    # Configuration
    era5_grib_dir = "/Users/habibw/Documents/ERA5_Data"
    processed_output_dir = "/Users/habibw/Documents/ERA5_Data/processed"
    half_hourly_output_dir = "/Users/habibw/Documents/ERA5_Data/half_hourly"
    
    # Check if GRIB directory exists
    if not os.path.exists(era5_grib_dir):
        print(f"ERA5 GRIB directory not found: {era5_grib_dir}")
        print("Please run download_era5_sample.py first to download ERA5 data.")
        sys.exit(1)
    
    # Process GRIB files
    process_era5_grib_files(era5_grib_dir, processed_output_dir, start_year=2022, end_year=2024)
    
    # Resample to half-hourly
    resample_era5_to_half_hourly(processed_output_dir, half_hourly_output_dir)
    
    print("\\n✅ ERA5 data processing completed!")
    print(f"Processed files available in: {processed_output_dir}")
    print(f"Half-hourly files available in: {half_hourly_output_dir}")
