import cdsapi
import os
import calendar
import time

def download_era5_data(output_dir, year, month, day, variable, area, time_steps):
    c = cdsapi.Client()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file name
    output_file = os.path.join(output_dir, f'era5_{variable}_{year}_{month:02d}_{day:02d}.grib')

    print(f"Attempting to download ERA5 data for {variable} on {year}-{month:02d}-{day:02d}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': variable,
                'year': str(year),
                'month': f'{month:02d}',
                'day': f'{day:02d}',
                'time': time_steps,
                'area': area, # North, West, South, East. Example: [55, -11, 51, -5] for Ireland
            },
            output_file)
        print(f"Successfully downloaded {output_file}")
    except Exception as e:
        print(f"Error downloading ERA5 data: {e}")

if __name__ == '__main__':
    # --- Configuration for download ---
    output_directory = '/Users/habibw/Documents/ERA5_Data' # Directory to save downloaded data
    
    # Define the period for which to download data
    start_year = 2023
    end_year = 2024
    
    era5_variables = [
        '2m_temperature',
        '2m_dewpoint_temperature',
        'surface_pressure',
        'total_precipitation',
        'surface_solar_radiation_downwards',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'evaporation',
        'boundary_layer_height',
    ]
    # Area for Ireland (North, West, South, East)
    # You might need to adjust these coordinates slightly for your exact area of interest
    ireland_area = [55.5, -10.5, 51.0, -5.0]
    # Time steps (hourly data for the day)
    hourly_time_steps = [
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
    ]

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                print(f"Downloading data for {year}-{month:02d}-{day:02d}")
                for variable in era5_variables:
                    download_era5_data(output_directory, year, month, day, 
                                       variable, ireland_area, hourly_time_steps)
                    time.sleep(1) # Be nice to the CDS API
