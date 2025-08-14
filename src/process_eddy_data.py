import pandas as pd
import numpy as np
import os
import hesseflux
import pyjams as pj # For esat function in VPD calculation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import calendar # For month names
from scipy.optimize import curve_fit

def calculate_vpd(df, rh_col='RH', temp_col='air_temp_c'):
    """
    Calculates Vapor Pressure Depicit (VPD) from Relative Humidity (RH) and Air Temperature.
    RH should be in percentage, Temperature in Celsius. VPD will be in kPa.
    """
    # Convert RH from % to fraction
    df['RH_fraction'] = df[rh_col] / 100.0
    # Calculate Saturation Vapor Pressure (SVP) in kPa
    df['SVP'] = 0.6108 * np.exp((17.27 * df[temp_col]) / (df[temp_col] + 237.3))
    # Calculate VPD in kPa
    df['VPD'] = (1 - df['RH_fraction']) * df['SVP']
    return df

def _findfirststart(starts, names):
    """
    Helper function to find the first occurrence in 'names' that starts with any string in 'starts'.
    Used by hesseflux examples.
    """
    found_names = []
    for s in starts:
        for name in names:
            if name.startswith(s):
                found_names.append(name)
                break # Found the first one for this 's', move to next 's'
    return found_names

def process_year_data_with_hesseflux(year, base_dir):
    """
    Processes Eddy Covariance data for a given year using hesseflux.
    """
    file_path = os.path.join(base_dir, f"{year}.csv")
    output_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")

    print(f"Processing data for {os.path.basename(base_dir)} - year: {year}")

    # Read the data
    df = pd.read_csv(file_path, na_values=['NA', -9999, -10272.15])

    # Convert DateTime to datetime objects and set as index
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('DateTime')

    # Calculate VPD if not present
    if 'VPD' not in df.columns:
        print("VPD column not found. Calculating VPD from RH and air_temp_c.")
        df = calculate_vpd(df, rh_col='RH', temp_col='air_temp_c')

    # Rename columns to hesseflux-like conventions for easier mapping
    # hesseflux expects specific names, so we map our data to those.
    # If your original data already matches, this step can be simplified.
    df = df.rename(columns={
        'SWIN_1_1_1': 'SW_IN', # Use SWIN_1_1_1 for SW_IN
        'air_temp_c': 'TA',
        'u*': 'USTAR',
        'co2_qc_flag': 'NEE_QC', # Assuming these are the QC flags
        'qc_LE': 'LE_QC',
        'qc_H': 'H_QC'
    })

    # Ensure all required columns are present after renaming
    required_hesseflux_cols = ['NEE', 'LE', 'H', 'PPFD', 'SW_IN', 'TA', 'VPD', 'USTAR', 'NEE_QC', 'LE_QC', 'H_QC']
    if not all(col in df.columns for col in required_hesseflux_cols):
        missing_cols = [col for col in required_hesseflux_cols if col not in df.columns]
        print(f"Error: Missing required columns for hesseflux processing: {missing_cols}")
        return

    # Create the flag dataframe (dff)
    # All NaN values will be set to undef and will be ignored in the following.
    # This happens via a second dataframe (dff), having the same columns and index as the input dataframe df,
    # representing quality flags. All cells that have a value other than 0 in the flag dataframe dff
    # will be ignored in the dataframe df.
    undef_val = -9999.0 # hesseflux uses -9999. for undef
    df.fillna(undef_val, inplace=True) # Fill NaNs with undef_val for hesseflux processing

    dff = df.copy(deep=True).astype(int)
    dff[:] = 0
    dff[df == undef_val] = 2 # Flag missing values as 2

    # Day / night determination
    # hesseflux uses SW_IN > 10 W m-2 as daytime
    isday = df['SW_IN'] > 10.0

    # --- hesseflux Processing Steps ---

    # 1. Spike / outlier flagging (using hesseflux.madspikes)
    # Parameters from hesseflux_example.cfg defaults
    nscan = 15 * 48 # 15 days * 48 half-hours/day
    nfill = 1 * 48  # 1 day * 48 half-hours/day
    z = 7.0
    deriv = 2 # mad on second derivatives

    # Columns to apply spike detection to (NEE, LE, H)
    # hesseflux expects 'FC' for CO2 flux, so we'll use 'NEE' and map it if needed
    # The user guide shows 'FC' and 'NEE' as options.
    # We'll use the actual column names from our df.
    flux_cols_for_spikes = ['NEE', 'LE', 'H']
    
    # Ensure these columns exist before passing to madspikes
    if not all(col in df.columns for col in flux_cols_for_spikes):
        print(f"Error: Missing flux columns for spike detection: {[col for col in flux_cols_for_spikes if col not in df.columns]}")
        return

    try:
        sflag = hesseflux.madspikes(df[flux_cols_for_spikes], flag=dff[flux_cols_for_spikes], isday=isday,
                                     undef=undef_val, nscan=nscan, nfill=nfill, z=z, deriv=deriv, plot=False)
        for col in flux_cols_for_spikes:
            dff.loc[sflag[col] == 2, col] = 3 # Flag spikes as 3
        print("Spike detection completed.")
    except Exception as e:
        print(f"Error during spike detection: {e}")
        # Continue processing even if spike detection fails, but log the error
        pass

    # 2. u* filtering (using hesseflux.ustarfilter)
    # Parameters from hesseflux_example.cfg defaults
    ustarmin = 0.01 # Default for non-forest
    nboot_ustar = 1 # No bootstrap for uncertainty
    plateaucrit = 0.95
    seasonout = True
    applyustarflag = True

    # hesseflux.ustarfilter expects NEE, USTAR, TA
    ustar_filter_cols = ['NEE', 'USTAR', 'TA']
    if not all(col in df.columns for col in ustar_filter_cols):
        print(f"Error: Missing columns for u* filtering: {[col for col in ustar_filter_cols if col not in df.columns]}")
        return

    try:
        ustars, flag_ustar = hesseflux.ustarfilter(df[ustar_filter_cols], flag=dff[ustar_filter_cols],
                                                    isday=isday, undef=undef_val,
                                                    ustarmin=ustarmin, nboot=nboot_ustar,
                                                    plateaucrit=plateaucrit, seasonout=seasonout, plot=False)
        if applyustarflag:
            for col in flux_cols_for_spikes: # Apply u* flag to NEE, LE, H
                dff.loc[flag_ustar == 2, col] = 5 # Flag u* filtered data as 5
        print("u* filtering completed.")
    except Exception as e:
        print(f"Error during u* filtering: {e}")
        pass

    # 3. Flux Partitioning (using hesseflux.nee2gpp)
    # Parameters from hesseflux_example.cfg defaults
    nogppnight = False

    # hesseflux.nee2gpp expects NEE, SW_IN, TA, VPD
    partition_cols = ['NEE', 'SW_IN', 'TA', 'VPD']
    if not all(col in df.columns for col in partition_cols):
        print(f"Error: Missing columns for flux partitioning: {[col for col in partition_cols if col not in df.columns]}")
        return

    try:
        # Nighttime method (Reichstein et al.)
        df_partn = hesseflux.nee2gpp(df[partition_cols], flag=dff[partition_cols], isday=isday,
                                     undef=undef_val, method='reichstein', nogppnight=nogppnight)
        df_partn = df_partn.rename(columns=lambda c: c + '_reichstein')

        # Daytime method (Lasslop et al.)
        df_partd = hesseflux.nee2gpp(df[partition_cols], flag=dff[partition_cols], isday=isday,
                                     undef=undef_val, method='lasslop', nogppnight=nogppnight)
        df_partd = df_partd.rename(columns=lambda c: c + '_lasslop')

        # Concatenate partitioned fluxes back to the main DataFrame
        df = pd.concat([df, df_partn, df_partd], axis=1)
        
        # Update dff with the new partitioned flux columns
        for col in df_partn.columns:
            dff[col] = 0 # Initialize with 0 (good quality) for new columns
            dff.loc[df[col] == undef_val, col] = 2 # Flag missing values
        for col in df_partd.columns:
            dff[col] = 0 # Initialize with 0 (good quality) for new columns
            dff.loc[df[col] == undef_val, col] = 2 # Flag missing values

        print("Flux partitioning completed.")
    except Exception as e:
        print(f"Error during flux partitioning: {e}")
        pass

    # 4. Gap-filling / Imputation (using hesseflux.gapfill)
    # Parameters from hesseflux_example.cfg defaults
    sw_dev = 50.0
    ta_dev = 2.5
    vpd_dev = 5.0
    longgap = 60 # days

    # hesseflux.gapfill expects SW_IN, TA, VPD for environmental conditions
    # And the fluxes to be filled (NEE, LE, H, GPP, RECO)
    fill_env_cols = ['SW_IN', 'TA', 'VPD']
    fill_flux_cols = ['NEE', 'LE', 'H'] # Add GPP and RECO after partitioning

    # Add partitioned fluxes to fill_flux_cols if they exist
    # Ensure these columns are present in df before extending
    if 'GPP_reichstein' in df.columns:
        fill_flux_cols.extend(['GPP_reichstein', 'RECO_reichstein', 'GPP_lasslop', 'RECO_lasslop'])

    # Now, all_fill_cols will correctly include the newly added partitioned flux columns
    all_fill_cols = list(set(fill_env_cols + fill_flux_cols)) # Use set to avoid duplicates

    if not all(col in df.columns for col in all_fill_cols):
        print(f"Error: Missing columns for gap-filling. Missing: {[col for col in all_fill_cols if col not in df.columns]}")
        return

    try:
        # Combine environmental and flux columns for gapfill input
        df_filled, dff_filled = hesseflux.gapfill(df[all_fill_cols], flag=dff[all_fill_cols],
                                                  sw_dev=sw_dev, ta_dev=ta_dev, vpd_dev=vpd_dev,
                                                  longgap=longgap, undef=undef_val, err=False, verbose=0)
        
        # Rename filled columns with '_f' suffix
        df_filled = df_filled.rename(columns=lambda c: c + '_f')
        dff_filled = dff_filled.rename(columns=lambda c: c + '_f')

        df = pd.concat([df, df_filled], axis=1)
        dff = pd.concat([dff, dff_filled], axis=1) # Keep track of flags for filled data
        print("Gap-filling completed.")
    except Exception as e:
        print(f"Error during gap-filling: {e}")
        pass

    # Final output preparation (optional, based on hesseflux output conventions)
    # You might want to select specific columns for the final output CSV
    
    # Replace undef_val with NaN for final output
    df.replace(undef_val, np.nan, inplace=True)

    # Save the processed data
    df.to_csv(output_file_path, index=True)
    print(f"Processed data saved to: {output_file_path}")

    # Generate plots
    generate_plots(df, base_dir, year)
    return df # Return the processed DataFrame

def generate_plots(df, base_dir, year):
    print(f"Generating plots for {os.path.basename(base_dir)} - {year}...")
    plot_dir = os.path.join(base_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    # Time Series Plots
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['NEE'], label='NEE (Raw)', alpha=0.7, color='#1f77b4')
    if 'NEE_f' in df.columns:
        plt.plot(df.index, df['NEE_f'], label='NEE (Filled)', alpha=0.7, color='#ff7f0e')
    plt.title(f'NEE Time Series - {os.path.basename(base_dir)} - {year}')
    plt.xlabel('Date')
    plt.ylabel('NEE (µmol m⁻² s⁻¹)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'NEE_timeseries_{year}.png'))
    plt.close()

    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['LE'], label='LE (Raw)', alpha=0.7, color='#1f77b4')
    if 'LE_f' in df.columns:
        plt.plot(df.index, df['LE_f'], label='LE (Filled)', alpha=0.7, color='#ff7f0e')
    plt.title(f'LE Time Series - {os.path.basename(base_dir)} - {year}')
    plt.xlabel('Date')
    plt.ylabel('LE (W m⁻²)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'LE_timeseries_{year}.png'))
    plt.close()

    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['H'], label='H (Raw)', alpha=0.7, color='#1f77b4')
    if 'H_f' in df.columns:
        plt.plot(df.index, df['H_f'], label='H (Filled)', alpha=0.7, color='#ff7f0e')
    plt.title(f'H Time Series - {os.path.basename(base_dir)} - {year}')
    plt.xlabel('Date')
    plt.ylabel('H (W m⁻²)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'H_timeseries_{year}.png'))
    plt.close()

    # Diurnal Cycle Plots
    df['Hour'] = df.index.hour + df.index.minute / 60

    plt.figure(figsize=(10, 6))
    plt.plot(df.groupby('Hour')['NEE'].mean(), label='NEE (Raw)', color='#1f77b4')
    if 'NEE_f' in df.columns:
        plt.plot(df.groupby('Hour')['NEE_f'].mean(), label='NEE (Filled)', color='#ff7f0e')
    plt.title(f'Average Diurnal Cycle of NEE - {os.path.basename(base_dir)} - {year}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average NEE (µmol m⁻² s⁻¹)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'NEE_diurnal_cycle_{year}.png'))
    plt.close()

    if 'GPP_reichstein' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.groupby('Hour')['GPP_reichstein'].mean(), label='GPP (Reichstein)', color='#2ca02c')
        if 'GPP_reichstein_f' in df.columns:
            plt.plot(df.groupby('Hour')['GPP_reichstein_f'].mean(), label='GPP (Reichstein, Filled)', color='#d62728')
        plt.title(f'Average Diurnal Cycle of GPP (Reichstein) - {os.path.basename(base_dir)} - {year}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average GPP (µmol m⁻² s⁻¹)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'GPP_reichstein_diurnal_cycle_{year}.png'))
        plt.close()

    if 'RECO_reichstein' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.groupby('Hour')['RECO_reichstein'].mean(), label='RECO (Reichstein)', color='#9467bd')
        if 'RECO_reichstein_f' in df.columns:
            plt.plot(df.groupby('Hour')['RECO_reichstein_f'].mean(), label='RECO (Reichstein, Filled)', color='#8c564b')
        plt.title(f'Average Diurnal Cycle of RECO (Reichstein) - {os.path.basename(base_dir)} - {year}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average RECO (µmol m⁻² s⁻¹)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'RECO_reichstein_diurnal_cycle_{year}.png'))
        plt.close()

    print(f"Plots generated and saved to {plot_dir}")

def generate_small_multiples_fingerprint(base_dir):
    print(f"Generating small multiples fingerprint plots for {os.path.basename(base_dir)}...")
    plot_dir = os.path.join(base_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 16), sharex=True, sharey=True) # Adjusted figsize for vertical orientation
    fig.suptitle(f'NEE Diurnal-Seasonal Fingerprint ({os.path.basename(base_dir)}) (Raw vs. Filled, 2023 vs. 2024)', fontsize=18, y=0.98) # Adjusted y for suptitle

    years = [2023, 2024]
    data_types = {'Raw': 'NEE', 'Filled': 'NEE_f'}
    
    # Determine a common colorbar range
    all_nee_data = []
    for year in years:
        processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
        if not os.path.exists(processed_file_path): # Skip if file doesn't exist
            continue
        df = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)
        df['Hour'] = df.index.hour + df.index.minute / 60
        df['DayOfYear'] = df.index.dayofyear
        if 'NEE' in df.columns:
            all_nee_data.append(df.pivot_table(values='NEE', index='DayOfYear', columns='Hour', aggfunc='mean').values)
        if 'NEE_f' in df.columns:
            all_nee_data.append(df.pivot_table(values='NEE_f', index='DayOfYear', columns='Hour', aggfunc='mean').values)
    
    # Flatten the list of arrays and remove NaNs to get min/max for colorbar
    all_nee_values = np.concatenate([arr.flatten() for arr in all_nee_data if arr is not None])
    # Use symmetric vmin/vmax around 0 for NEE
    abs_max = np.nanpercentile(np.abs(all_nee_values), 99) # Use 99th percentile to avoid extreme outliers
    vmin = -abs_max
    vmax = abs_max

    # Prepare month ticks for X-axis
    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335] # Day of year for 1st of each month (approx)
    month_labels = [calendar.month_abbr[i] for i in range(1, 13)]

    for i, year in enumerate(years):
        processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
        if not os.path.exists(processed_file_path): # Skip if file doesn't exist
            ax = axes[0, i] # Raw subplot
            ax.text(0.5, 0.5, f'No data for {year}', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=12)
            ax.set_title(f'{year} - Raw')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Hour of Day')
            ax.set_xticks(month_days)
            ax.set_xticklabels(month_labels, rotation=45, ha='right')
            ax.set_yticks(np.arange(0, 25, 6))
            ax.grid(True, linestyle='--', alpha=0.7)

            ax = axes[1, i] # Filled subplot
            ax.text(0.5, 0.5, f'No data for {year}', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=12)
            ax.set_title(f'{year} - Filled')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Hour of Day')
            ax.set_xticks(month_days)
            ax.set_xticklabels(month_labels, rotation=45, ha='right')
            ax.set_yticks(np.arange(0, 25, 6))
            ax.grid(True, linestyle='--', alpha=0.7)
            continue

        df = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)
        df['Hour'] = df.index.hour + df.index.minute / 60
        df['DayOfYear'] = df.index.dayofyear

        for j, (data_type_label, col_name) in enumerate(data_types.items()):
            ax = axes[j, i] # Row for data type, Column for year
            if col_name in df.columns:
                nee_fingerprint_data = df.pivot_table(values=col_name, index='DayOfYear', columns='Hour', aggfunc='mean')
                
                # Plotting with Day of Year on X-axis and Hour of Day on Y-axis
                im = ax.imshow(nee_fingerprint_data, aspect='auto', origin='lower', cmap='RdYlGn',
                               extent=[1, 366, 0, 24], vmin=vmin, vmax=vmax) # Corrected extent
                
                ax.set_title(f'{year} - {data_type_label}')
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('Hour of Day')
                ax.set_xticks(month_days)
                ax.set_xticklabels(month_labels, rotation=45, ha='right')
                ax.set_yticks(np.arange(0, 25, 6)) # Every 6 hours
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f'No {col_name} data', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=12)
                ax.set_title(f'{year} - {data_type_label}')
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('Hour of Day')
                ax.set_xticks(month_days)
                ax.set_xticklabels(month_labels, rotation=45, ha='right')
                ax.set_yticks(np.arange(0, 25, 6))
                ax.grid(True, linestyle='--', alpha=0.7)

    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height] - Adjusted position further right
    fig.colorbar(im, cax=cbar_ax, label='Average NEE (µmol m⁻² s⁻¹)')

    plt.tight_layout(rect=[0.05, 0.03, 0.90, 0.95]) # Adjusted rect for more space on the right
    plt.savefig(os.path.join(plot_dir, f'NEE_fingerprint_small_multiples_{os.path.basename(base_dir)}.png'))
    plt.close()
    print(f"Small multiples fingerprint plots generated and saved to {plot_dir}")

# Mitscherlich function for Light Response Curve
def mitscherlich_lrc(ppfd, a, b, c):
    # NEE = 1 - (a + b) * (1 - exp((-c * PPFD) / (a + b))) + b
    # Note: This function is implemented as provided by the user.
    # It is an unusual form for a Mitscherlich LRC, especially the '1 - (a + b)' part.
    # If PPFD = 0, NEE = 1 + b, not b (respiration).
    # Given the user's equation, if a+b is zero, the equation breaks.
    # For practical fitting, we'll return NaN or a large value to indicate a bad fit.
    
    denom = a + b
    if np.isclose(denom, 0):
        return np.full_like(ppfd, np.nan)

    # Ensure the exponent argument is not too large or too small to avoid overflow/underflow
    exp_arg = (-c * ppfd) / denom
    # Clip exp_arg to prevent overflow/underflow issues with np.exp
    exp_arg = np.clip(exp_arg, -700, 700) # Approx. log(np.finfo(float).max) and log(np.finfo(float).min)

    return 1 - denom * (1 - np.exp(exp_arg)) + b

def get_weekly_lrc_parameters(df, base_dir, year):
    print(f"Calculating weekly Light Response Curve parameters for {os.path.basename(base_dir)} - {year}...")
    lrc_plot_dir = os.path.join(base_dir, "plot", "lrc_plots")
    os.makedirs(lrc_plot_dir, exist_ok=True)

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    lrc_params_list = []

    # Define seasons (month ranges)
    seasons = {
        'Winter': [12, 1, 2],  # Dec, Jan, Feb
        'Spring': [3, 4, 5],   # Mar, Apr, May
        'Summer': [6, 7, 8],   # Jun, Jul, Aug
        'Autumn': [9, 10, 11]  # Sep, Oct, Nov
    }

    # Ensure PPFD column exists
    if 'PPFD' not in df.columns:
        print(f"Skipping LRC parameter calculation for {os.path.basename(base_dir)} - {year}: PPFD column not found.")
        return lrc_params_list

    # Iterate through weeks
    df['Week'] = df.index.isocalendar().week.astype(int)
    for week_num in sorted(df['Week'].unique()):
        weekly_df = df[df['Week'] == week_num].copy()

        for data_type_label, col_name in {'Raw': 'NEE', 'Filled': 'NEE_f'}.items():
            # Filter for daytime data for fitting
            lrc_data = weekly_df[weekly_df['PPFD'] > 10].dropna(subset=[col_name, 'PPFD'])

            # Skip if not enough data points for fitting
            if len(lrc_data) < 3: # Need at least 3 points for 3 parameters
                lrc_params_list.append({
                    'Tower': os.path.basename(base_dir),
                    'Year': year,
                    'Week': week_num,
                    'DataType': data_type_label,
                    'a': np.nan, 'b': np.nan, 'c': np.nan,
                    'a_lower': np.nan, 'a_upper': np.nan,
                    'b_lower': np.nan, 'b_upper': np.nan,
                    'c_lower': np.nan, 'c_upper': np.nan,
                    'R2': np.nan,
                    'Fit_Status': 'Not enough data'
                })
                continue

            try:
                # Initial guesses for a, b, c
                b_guess = lrc_data[lrc_data['PPFD'] < 20][col_name].mean()
                if np.isnan(b_guess): b_guess = 5.0
                a_guess = 20.0
                c_guess = 0.05

                p0 = [a_guess, b_guess, c_guess]
                bounds = ([0.1, -50, 0.001], [100, 50, 1.0]) # [a_min, b_min, c_min], [a_max, b_max, c_max]

                params, pcov = curve_fit(mitscherlich_lrc, lrc_data['PPFD'], lrc_data[col_name], p0=p0, bounds=bounds, maxfev=5000)
                
                # Calculate R-squared
                y_observed = lrc_data[col_name].values
                y_predicted = mitscherlich_lrc(lrc_data['PPFD'].values, *params)
                ss_res = np.sum((y_observed - y_predicted)**2)
                ss_tot = np.sum((y_observed - np.mean(y_observed))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                # Calculate 95% confidence intervals
                perr = np.sqrt(np.diag(pcov)) # Standard errors
                a_lower, a_upper = params[0] - 1.96 * perr[0], params[0] + 1.96 * perr[0]
                b_lower, b_upper = params[1] - 1.96 * perr[1], params[1] + 1.96 * perr[1]
                c_lower, c_upper = params[2] - 1.96 * perr[2], params[2] + 1.96 * perr[2]

                lrc_params_list.append({
                    'Tower': os.path.basename(base_dir),
                    'Year': year,
                    'Week': week_num,
                    'DataType': data_type_label,
                    'a': params[0], 'b': params[1], 'c': params[2],
                    'a_lower': a_lower, 'a_upper': a_upper,
                    'b_lower': b_lower, 'b_upper': b_upper,
                    'c_lower': c_lower, 'c_upper': c_upper,
                    'R2': r_squared,
                    'Fit_Status': 'Success'
                })

            except RuntimeError as e:
                lrc_params_list.append({
                    'Tower': os.path.basename(base_dir),
                    'Year': year,
                    'Week': week_num,
                    'DataType': data_type_label,
                    'a': np.nan, 'b': np.nan, 'c': np.nan,
                    'a_lower': np.nan, 'a_upper': np.nan,
                    'b_lower': np.nan, 'b_upper': np.nan,
                    'c_lower': np.nan, 'c_upper': np.nan,
                    'R2': np.nan,
                    'Fit_Status': f'Fit failed: {e}'
                })
                print(f"LRC fit failed for {os.path.basename(base_dir)} - {year} Week {week_num} {data_type_label}: {e}")
            except ValueError as e:
                lrc_params_list.append({
                    'Tower': os.path.basename(base_dir),
                    'Year': year,
                    'Week': week_num,
                    'DataType': data_type_label,
                    'a': np.nan, 'b': np.nan, 'c': np.nan,
                    'a_lower': np.nan, 'a_upper': np.nan,
                    'b_lower': np.nan, 'b_upper': np.nan,
                    'c_lower': np.nan, 'c_upper': np.nan,
                    'R2': np.nan,
                    'Fit_Status': f'Fit failed (invalid input): {e}'
                })
                print(f"LRC fit failed for {os.path.basename(base_dir)} - {year} Week {week_num} {data_type_label} (invalid input): {e}")

    print(f"Weekly LRC parameter calculation completed for {os.path.basename(base_dir)} - {year}.")
    return lrc_params_list

def plot_seasonal_lrc_multiples(lrc_params_df, base_dir, year):
    print(f"Generating seasonal Light Response Curve plots for {os.path.basename(base_dir)} - {year}...")
    lrc_plot_dir = os.path.join(base_dir, "plot", "lrc_plots")
    os.makedirs(lrc_plot_dir, exist_ok=True)

    # Set global matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    # Define seasons (month ranges)
    seasons = {
        'Winter': [12, 1, 2],  # Dec, Jan, Feb
        'Spring': [3, 4, 5],   # Mar, Apr, May
        'Summer': [6, 7, 8],   # Jun, Jul, Aug
        'Autumn': [9, 10, 11]  # Sep, Oct, Nov
    }

    processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
    df = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)

    # Ensure PPFD column exists
    if 'PPFD' not in df.columns:
        print(f"Skipping LRC plots for {os.path.basename(base_dir)} - {year}: PPFD column not found.")
        return

    fig, axes = plt.subplots(len(seasons), 2, figsize=(16, 20), sharex=True, sharey=True)
    fig.suptitle(f'Light Response Curves - {os.path.basename(base_dir)} - {year}', fontsize=20, y=0.98)

    for row_idx, (season_name, months) in enumerate(seasons.items()):
        seasonal_df = df[df.index.month.isin(months)].copy()

        for col_idx, (data_type_label, col_name) in enumerate({'Raw': 'NEE', 'Filled': 'NEE_f'}.items()):
            ax = axes[row_idx, col_idx]
            
            # Filter for daytime data for fitting
            lrc_data = seasonal_df[seasonal_df['PPFD'] > 10].dropna(subset=[col_name, 'PPFD'])

            # Skip if not enough data points for fitting
            if len(lrc_data) < 10:
                ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=10)
                ax.set_title(f'{season_name} - {data_type_label}')
                ax.grid(True, linestyle='--', alpha=0.7)
                continue

            ax.scatter(lrc_data['PPFD'], lrc_data[col_name], label=f'{data_type_label} Data', alpha=0.6, s=10, color='#1f77b4' if data_type_label == 'Raw' else '#ff7f0e')

            # Retrieve parameters and R-squared from the lrc_params_df
            params_row = lrc_params_df[(lrc_params_df['Tower'] == os.path.basename(base_dir)) &
                                       (lrc_params_df['Year'] == year) &
                                       (lrc_params_df['Week'] == lrc_data.index.isocalendar().week.iloc[0]) &
                                       (lrc_params_df['DataType'] == data_type_label)]
            
            if not params_row.empty and params_row['Fit_Status'].iloc[0] == 'Success':
                a, b, c = params_row['a'].iloc[0], params_row['b'].iloc[0], params_row['c'].iloc[0]
                r_squared = params_row['R2'].iloc[0]

                ppfd_fit = np.linspace(0, lrc_data['PPFD'].max() * 1.1, 100)
                nee_fit = mitscherlich_lrc(ppfd_fit, a, b, c)
                ax.plot(ppfd_fit, nee_fit, color='red', linewidth=2, label='Fitted Curve')

                # Display parameters and R-squared
                param_text = f'a={a:.2f}\nb={b:.2f}\nc={c:.3f}\nR²={r_squared:.2f}'
                ax.text(0.95, 0.95, param_text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8), fontsize=10)

            else:
                ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)

            ax.set_title(f'{season_name} - {data_type_label}')
            ax.set_xlabel('PPFD (µmol m⁻² s⁻¹)')
            ax.set_ylabel('NEE (µmol m⁻² s⁻¹)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left') # Place legend to avoid parameter text

    # Set common X and Y labels for the entire figure
    fig.text(0.5, 0.04, 'PPFD (µmol m⁻² s⁻¹)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'NEE (µmol m⁻² s⁻¹)', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.08, 0.06, 1, 0.95]) # Adjust rect to make space for common labels
    plt.savefig(os.path.join(lrc_plot_dir, f'LRC_seasonal_small_multiples_{os.path.basename(base_dir)}_{year}.png'))
    plt.close()

    print(f"Seasonal LRC plots generated and saved to {lrc_plot_dir}")


# Define the base directories for each tower
base_directories = [
    "/Users/habibw/Documents/Gurteen",
    "/Users/habibw/Documents/Athenry",
    "/Users/habibw/Documents/JC1",
    "/Users/habibw/Documents/JC2",
    "/Users/habibw/Documents/Timoleague",
    "/Users/habibw/Documents/Clarabog",
    "/Users/habibw/Documents/Lullymore"
]

all_lrc_params_2023 = []
all_lrc_params_2024 = []

for base_dir in base_directories:
    # Process data for 2023 and 2024 if files exist
    for year in [2023, 2024]:
        if os.path.exists(os.path.join(base_dir, f"{year}.csv")):
            processed_df = process_year_data_with_hesseflux(year, base_dir)
            if processed_df is not None: # Only proceed if processing was successful
                weekly_lrc_data = get_weekly_lrc_parameters(processed_df, base_dir, year)
                if year == 2023:
                    all_lrc_params_2023.extend(weekly_lrc_data)
                elif year == 2024:
                    all_lrc_params_2024.extend(weekly_lrc_data)
        else:
            print(f"Skipping processing for {os.path.basename(base_dir)} - {year}: CSV file not found.")

    # Generate small multiples fingerprint plots after individual year processing
    # Check if processed files exist before generating plots that rely on them
    processed_2023_exists = os.path.exists(os.path.join(base_dir, "processed_hesseflux_2023.csv"))
    processed_2024_exists = os.path.exists(os.path.join(base_dir, "processed_hesseflux_2024.csv"))

    if processed_2023_exists or processed_2024_exists:
        generate_small_multiples_fingerprint(base_dir)
    else:
        print(f"Skipping fingerprint plot generation for {os.path.basename(base_dir)}: No processed data found.")

# Convert collected LRC parameters to DataFrames and save to CSV
output_gurteen_dir = "/Users/habibw/Documents/Gurteen"

if all_lrc_params_2023:
    lrc_df_2023 = pd.DataFrame(all_lrc_params_2023)
    lrc_df_2023.to_csv(os.path.join(output_gurteen_dir, "lrc_parameters_2023.csv"), index=False)
    print(f"LRC parameters for 2023 saved to {os.path.join(output_gurteen_dir, 'lrc_parameters_2023.csv')}")
else:
    print("No LRC parameters collected for 2023.")

if all_lrc_params_2024:
    lrc_df_2024 = pd.DataFrame(all_lrc_params_2024)
    lrc_df_2024.to_csv(os.path.join(output_gurteen_dir, "lrc_parameters_2024.csv"), index=False)
    print(f"LRC parameters for 2024 saved to {os.path.join(output_gurteen_dir, 'lrc_parameters_2024.csv')}")
else:
    print("No LRC parameters collected for 2024.")

# Identify the week with the most usable tower data based on optimal R2 value
print("Identifying the week with the most usable tower data...")

combined_lrc_df = pd.DataFrame()
if all_lrc_params_2023:
    combined_lrc_df = pd.concat([combined_lrc_df, lrc_df_2023])
if all_lrc_params_2024:
    combined_lrc_df = pd.concat([combined_lrc_df, lrc_df_2024])

if not combined_lrc_df.empty:
    # Filter for successful fits and valid R2 values
    successful_fits = combined_lrc_df[combined_lrc_df['Fit_Status'] == 'Success'].copy()
    successful_fits = successful_fits.dropna(subset=['R2'])

    if not successful_fits.empty:
        # Group by Week and calculate the mean R2
        weekly_avg_r2 = successful_fits.groupby('Week')['R2'].mean()
        
        # Find the week with the highest average R2
        most_usable_week = weekly_avg_r2.idxmax()
        max_avg_r2 = weekly_avg_r2.max()

        print(f"The week with the most usable tower data (highest average R²): Week {most_usable_week} (Average R²: {max_avg_r2:.2f})")

        # Plot seasonal LRC multiples using the collected parameters
        # This part needs to be called after all parameters are collected and saved
        for base_dir in base_directories:
            for year in [2023, 2024]:
                processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
                if os.path.exists(processed_file_path):
                    plot_seasonal_lrc_multiples(combined_lrc_df, base_dir, year)
                else:
                    print(f"Skipping seasonal LRC plot generation for {os.path.basename(base_dir)} - {year}: Processed data not found.")
    else:
        print("No successful LRC fits with valid R² values found across all data.")
else:
    print("No LRC parameters collected for any tower or year.")