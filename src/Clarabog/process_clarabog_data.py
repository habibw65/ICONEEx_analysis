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
    Processes Eddy Covariance data for a given year.
    Assumes data is in clara_YYYY.csv format.
    """
    file_path = os.path.join(base_dir, f"clara_{year}.csv")
    output_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")

    print(f"Processing data for {os.path.basename(base_dir)} - year: {year}")

    if not os.path.exists(file_path):
        print(f"Error: Data file for {year} not found in {base_dir} (expected clara_{year}.csv).")
        return None

    df = pd.read_csv(file_path, na_values=['NA', -9999, -10272.15], encoding='utf-8-sig')

    if df.empty:
        print(f"No data for {year} in {os.path.basename(base_dir)}. Skipping processing.")
        return None

    # Rename columns to hesseflux-like conventions for easier mapping
    column_renames = {
        'Net Ecosystem Exchange': 'NEE',
        'Latent heat Flux': 'LE',
        'Sensible Heat Flux': 'H',
        'Photosynthetic Photon Flux Density': 'PPFD',
        'Net Radiation': 'SW_IN',
        'Air Temperature': 'TA',
        'Relative Humidity': 'RH',
        'U star': 'USTAR',
        ' ': 'NEE' # Handle empty column name for NEE in some files
    }
    df = df.rename(columns={k: v for k, v in column_renames.items() if k in df.columns})

    # Convert DateTime to datetime objects and set as index
    # Drop rows where DateTime is NaT (Not a Time) after conversion
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)
    df = df.set_index('DateTime')

    # Initialize missing quality control columns with 0 (good quality)
    qc_cols = ['NEE_QC', 'LE_QC', 'H_QC']
    for col in qc_cols:
        if col not in df.columns:
            df[col] = 0
            print(f"Warning: Quality control column '{col}' not found. Initializing with 0 (good quality).")

    # Calculate VPD if not present and if RH and air_temp_c are available
    if 'VPD' not in df.columns:
        if 'RH' in df.columns and 'TA' in df.columns: # Use TA for air_temp_c
            print("VPD column not found. Calculating VPD from RH and TA.")
            df = calculate_vpd(df, rh_col='RH', temp_col='TA')
        else:
            print("Warning: 'RH' or 'TA' columns not found. Skipping VPD calculation.")
            df['VPD'] = np.nan # Assign NaN to VPD if it cannot be calculated

    # Ensure all required columns are present after renaming
    required_hesseflux_cols = ['NEE', 'LE', 'H', 'PPFD', 'SW_IN', 'TA', 'VPD', 'USTAR', 'NEE_QC', 'LE_QC', 'H_QC']
    if not all(col in df.columns for col in required_hesseflux_cols):
        missing_cols = [col for col in required_hesseflux_cols if col not in df.columns]
        print(f"Error: Missing required columns for hesseflux processing: {missing_cols}")
        return None

    # Create the flag dataframe (dff)
    undef_val = -9999.0 # hesseflux uses -9999. for undef
    df.fillna(undef_val, inplace=True) # Fill NaNs with undef_val for hesseflux processing

    dff = pd.DataFrame(0, index=df.index, columns=df.columns) # Initialize with zeros
    dff[df == undef_val] = 2 # Flag missing values as 2

    # Day / night determination
    isday = df['SW_IN'] > 10.0

    # --- hesseflux Processing Steps ---

    # 1. Spike / outlier flagging (using hesseflux.madspikes)
    nscan = 15 * 48 # 15 days * 48 half-hours/day
    nfill = 1 * 48  # 1 day * 48 half-hours/day
    z = 7.0
    deriv = 2 # mad on second derivatives

    flux_cols_for_spikes = ['NEE', 'LE', 'H']
    if not all(col in df.columns for col in flux_cols_for_spikes):
        print(f"Error: Missing flux columns for spike detection: {[col for col in flux_cols_for_spikes if col not in df.columns]}")
        return None

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
    ustarmin = 0.01 # Default for non-forest
    nboot_ustar = 1 # No bootstrap for uncertainty
    plateaucrit = 0.95
    seasonout = True
    applyustarflag = True

    ustar_filter_cols = ['NEE', 'USTAR', 'TA']
    if not all(col in df.columns for col in ustar_filter_cols):
        print(f"Error: Missing columns for u* filtering: {[col for col in ustar_filter_cols if col not in df.columns]}")
        return None

    # Check if enough data points for u* filtering (approx. 365 days * 48 half-hours/day)
    # hesseflux ustarfilter requires full years of data. If not, skip.
    if len(df) < 365 * 48 * 0.8: # Check for at least 80% of a full year's data
        print(f"Warning: Not enough data for full year u* filtering for {year}. Skipping u* filtering.")
        # Initialize ustars and flag_ustar to avoid errors later
        ustars = pd.DataFrame(index=df.index, columns=['USTAR'])
        flag_ustar = pd.DataFrame(0, index=df.index, columns=['USTAR'])
    else:
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
    nogppnight = False

    partition_cols = ['NEE', 'SW_IN', 'TA', 'VPD']
    if not all(col in df.columns for col in partition_cols):
        print(f"Error: Missing columns for flux partitioning: {[col for col in partition_cols if col not in df.columns]}")
        return None

    try:
        df_partn = hesseflux.nee2gpp(df[partition_cols], flag=dff[partition_cols], isday=isday,
                                     undef=undef_val, method='reichstein', nogppnight=nogppnight)
        df_partn = df_partn.rename(columns=lambda c: c + '_reichstein')

        df_partd = hesseflux.nee2gpp(df[partition_cols], flag=dff[partition_cols], isday=isday,
                                     undef=undef_val, method='lasslop', nogppnight=nogppnight)
        df_partd = df_partd.rename(columns=lambda c: c + '_lasslop')

        df = pd.concat([df, df_partn, df_partd], axis=1)
        
        for col in df_partn.columns:
            dff[col] = 0
            dff.loc[df[col] == undef_val, col] = 2
        for col in df_partd.columns:
            dff[col] = 0
            dff.loc[df[col] == undef_val, col] = 2

        print("Flux partitioning completed.")
    except Exception as e:
        print(f"Error during flux partitioning: {e}")
        pass

    # 4. Gap-filling / Imputation (using hesseflux.gapfill)
    sw_dev = 50.0
    ta_dev = 2.5
    vpd_dev = 5.0
    longgap = 60 # days

    fill_env_cols = ['SW_IN', 'TA', 'VPD']
    fill_flux_cols = ['NEE', 'LE', 'H']
    if 'GPP_reichstein' in df.columns:
        fill_flux_cols.extend(['GPP_reichstein', 'RECO_reichstein', 'GPP_lasslop', 'RECO_lasslop'])

    all_fill_cols = list(set(fill_env_cols + fill_flux_cols))

    if not all(col in df.columns for col in all_fill_cols):
        print(f"Error: Missing columns for gap-filling. Missing: {[col for col in all_fill_cols if col not in df.columns]}")
        return None

    try:
        df_filled, dff_filled = hesseflux.gapfill(df[all_fill_cols], flag=dff[all_fill_cols],
                                                  sw_dev=sw_dev, ta_dev=ta_dev, vpd_dev=vpd_dev,
                                                  longgap=longgap, undef=undef_val, err=False, verbose=0)
        
        df_filled = df_filled.rename(columns=lambda c: c + '_f')
        dff_filled = dff_filled.rename(columns=lambda c: c + '_f')

        df = pd.concat([df, df_filled], axis=1)
        dff = pd.concat([dff, dff_filled], axis=1)
        print("Gap-filling completed.")
    except Exception as e:
        print(f"Error during gap-filling: {e}")
        pass

    df.replace(undef_val, np.nan, inplace=True)

    df.to_csv(output_file_path, index=True)
    print(f"Processed data saved to: {output_file_path}")

    generate_plots(df, base_dir, year)
    return df

def generate_plots(df, base_dir, year):
    print(f"Generating plots for {os.path.basename(base_dir)} - {year}...")
    plot_dir = os.path.join(base_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5
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

def generate_small_multiples_fingerprint(base_dir, years):
    print(f"Generating small multiples fingerprint plots for {os.path.basename(base_dir)}...")
    plot_dir = os.path.join(base_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5
    })

    fig, axes = plt.subplots(2, len(years), figsize=(6 * len(years), 16), sharex=True, sharey=True)
    if len(years) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(f'NEE Diurnal-Seasonal Fingerprint ({os.path.basename(base_dir)}) (Raw vs. Filled, {", ".join(map(str, years))})', fontsize=18, y=0.98)

    data_types = {'Raw': 'NEE', 'Filled': 'NEE_f'}
    
    all_nee_data = []
    for year in years:
        processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
        if not os.path.exists(processed_file_path):
            continue
        df = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)
        df['Hour'] = df.index.hour + df.index.minute / 60
        df['DayOfYear'] = df.index.dayofyear
        if 'NEE' in df.columns:
            all_nee_data.append(df.pivot_table(values='NEE', index='DayOfYear', columns='Hour', aggfunc='mean').values)
        if 'NEE_f' in df.columns:
            all_nee_data.append(df.pivot_table(values='NEE_f', index='DayOfYear', columns='Hour', aggfunc='mean').values)
    
    all_nee_values = np.concatenate([arr.flatten() for arr in all_nee_data if arr is not None])
    abs_max = np.nanpercentile(np.abs(all_nee_values), 99)
    vmin = -abs_max
    vmax = abs_max

    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = [calendar.month_abbr[i] for i in range(1, 13)]

    for i, year in enumerate(years):
        processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
        if not os.path.exists(processed_file_path):
            for j in range(2):
                ax = axes[j, i]
                ax.text(0.5, 0.5, f'No data for {year}', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=12)
                ax.set_title(f'{year} - {list(data_types.keys())[j]}')
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
            ax = axes[j, i]
            if col_name in df.columns:
                nee_fingerprint_data = df.pivot_table(values=col_name, index='DayOfYear', columns='Hour', aggfunc='mean')
                
                im = ax.imshow(nee_fingerprint_data, aspect='auto', origin='lower', cmap='RdYlGn',
                               extent=[1, 366, 0, 24], vmin=vmin, vmax=vmax)
                
                ax.set_title(f'{year} - {data_type_label}')
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('Hour of Day')
                ax.set_xticks(month_days)
                ax.set_xticklabels(month_labels, rotation=45, ha='right')
                ax.set_yticks(np.arange(0, 25, 6))
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

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Average NEE (µmol m⁻² s⁻¹)')

    plt.tight_layout(rect=[0.05, 0.03, 0.90, 0.95])
    plt.savefig(os.path.join(plot_dir, f'NEE_fingerprint_small_multiples_{os.path.basename(base_dir)}.png'))
    plt.close()
    print(f"Small multiples fingerprint plots generated and saved to {plot_dir}")

# Mitscherlich function for Light Response Curve
def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    if np.isclose(denom, 0):
        return np.full_like(ppfd, np.nan)

    exp_arg = (-c * ppfd) / denom
    exp_arg = np.clip(exp_arg, -700, 700)

    return 1 - denom * (1 - np.exp(exp_arg)) + b

def get_weekly_lrc_parameters(df, base_dir, year):
    print(f"Calculating weekly Light Response Curve parameters for {os.path.basename(base_dir)} - {year}...")
    lrc_plot_dir = os.path.join(base_dir, "plot", "lrc_plots")
    os.makedirs(lrc_plot_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18, # Adjusted for vertical orientation
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    lrc_params_list = []

    if 'PPFD' not in df.columns:
        print(f"Skipping LRC parameter calculation for {os.path.basename(base_dir)} - {year}: PPFD column not found.")
        return lrc_params_list

    df['Week'] = df.index.isocalendar().week.astype(int)
    for week_num in sorted(df['Week'].unique()):
        weekly_df = df[df['Week'] == week_num].copy()

        for data_type_label, col_name in {'Raw': 'NEE', 'Filled': 'NEE_f'}.items(): # Only 'Raw' as we are not gap-filling
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
        'figure.titlesize': 18, # Adjusted for vertical orientation
        'figure.dpi': 300, # High resolution
        'savefig.dpi': 300,
        'axes.linewidth': 1.0, # Thicker axis lines
        'grid.linewidth': 0.5, # Thinner grid lines
        'lines.linewidth': 1.5 # Thicker plot lines
    })

    # Define seasons (month ranges)
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }

    processed_file_path = os.path.join(base_dir, f"processed_hesseflux_{year}.csv")
    df = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)

    # Ensure PPFD column exists
    if 'PPFD' not in df.columns:
        print(f"Skipping LRC plots for {os.path.basename(base_dir)} - {year}: PPFD column not found.")
        return

    fig, axes = plt.subplots(len(seasons), 1, figsize=(8, 6 * len(seasons)), sharex=True, sharey=True) # Adjusted for single column
    fig.suptitle(f'Light Response Curves - {os.path.basename(base_dir)} - {year}', fontsize=20, y=0.98)

    for row_idx, (season_name, months) in enumerate(seasons.items()):
        ax = axes[row_idx] # Single column of plots
        seasonal_df = df[df.index.month.isin(months)].copy()

        for data_type_label, col_name in {'Raw': 'NEE', 'Filled': 'NEE_f'}.items(): # Only 'Raw' as we are not gap-filling
            # Filter for daytime data for fitting
            lrc_data = seasonal_df[seasonal_df['PPFD'] > 10].dropna(subset=[col_name, 'PPFD'])

            # Skip if not enough data points for fitting
            if len(lrc_data) < 10:
                ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='gray', fontsize=10)
                ax.set_title(f'{season_name} - {data_type_label}')
                ax.grid(True, linestyle='--', alpha=0.7)
                continue

            ax.scatter(lrc_data['PPFD'], lrc_data[col_name], label=f'{data_type_label} Data', alpha=0.6, s=10, color='#1f77b4')

            # Retrieve parameters and R-squared from the lrc_params_df
            # Find the closest week's parameters for the current season's data
            # This is a simplification; ideally, you'd have parameters for each week or a seasonal fit.
            # For now, we'll just pick the first available successful fit for the tower and year.
            params_row = lrc_params_df[(lrc_params_df['Tower'] == os.path.basename(base_dir)) &
                                       (lrc_params_df['Year'] == year) &
                                       (lrc_params_df['DataType'] == data_type_label) &
                                       (lrc_params_df['Fit_Status'] == 'Success')].iloc[0] if not lrc_params_df[(lrc_params_df['Tower'] == os.path.basename(base_dir)) &
                                       (lrc_params_df['Year'] == year) &
                                       (lrc_params_df['DataType'] == data_type_label) &
                                       (lrc_params_df['Fit_Status'] == 'Success')].empty else None
            
            if params_row is not None:
                a, b, c = params_row['a'], params_row['b'], params_row['c']
                r_squared = params_row['R2']

                ppfd_fit = np.linspace(0, lrc_data['PPFD'].max() * 1.1, 100)
                nee_fit = mitscherlich_lrc(ppfd_fit, a, b, c)
                ax.plot(ppfd_fit, nee_fit, color='red', linewidth=2, label='Fitted Curve')

                param_text = f'a={a:.2f}\nb={b:.2f}\nc={c:.3f}\nR²={r_squared:.2f}'
                ax.text(0.95, 0.95, param_text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8), fontsize=10)

            else:
                ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes, horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)

            ax.set_title(f'{season_name} - {data_type_label}')
            ax.set_xlabel('PPFD (µmol m⁻² s⁻¹)')
            ax.set_ylabel('NEE (µmol m⁻² s⁻¹)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left')

    # Set common X and Y labels for the entire figure
    fig.text(0.5, 0.04, 'PPFD (µmol m⁻² s⁻¹)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'NEE (µmol m⁻² s⁻¹)', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.08, 0.06, 1, 0.95]) # Adjust rect to make space for common labels
    plt.savefig(os.path.join(lrc_plot_dir, f'LRC_seasonal_small_multiples_{os.path.basename(base_dir)}_{year}.png'))
    plt.close()

    print(f"Seasonal LRC plots generated and saved to {lrc_plot_dir}")


if __name__ == '__main__':
    clarabog_dir = "/Users/habibw/Documents/Clarabog"

    all_lrc_params = []
    years_to_plot = []

    # Process each year separately from the CSV files
    for year in [2018, 2019, 2020, 2021, 2022]: # Assuming these are the years for Clarabog
        processed_df = process_year_data_with_hesseflux(year, clarabog_dir)
        if processed_df is not None:
            weekly_lrc_data = get_weekly_lrc_parameters(processed_df, clarabog_dir, year)
            all_lrc_params.extend(weekly_lrc_data)
            years_to_plot.append(year)

    # Generate small multiples fingerprint plots for Clarabog
    if years_to_plot:
        generate_small_multiples_fingerprint(clarabog_dir, sorted(list(set(years_to_plot))))
    else:
        print(f"Skipping fingerprint plot generation for Clarabog: No processed data found.")

    # Save LRC parameters for Clarabog
    if all_lrc_params:
        lrc_df = pd.DataFrame(all_lrc_params)
        lrc_df.to_csv(os.path.join(clarabog_dir, "lrc_parameters_clarabog.csv"), index=False)
        print(f"LRC parameters for Clarabog saved to {os.path.join(clarabog_dir, 'lrc_parameters_clarabog.csv')}")

        # Plot seasonal LRC multiples for Clarabog
        for year in sorted(list(set(years_to_plot))):
            processed_file_path = os.path.join(clarabog_dir, f"processed_hesseflux_{year}.csv")
            if os.path.exists(processed_file_path):
                plot_seasonal_lrc_multiples(lrc_df, clarabog_dir, year)
            else:
                print(f"Skipping seasonal LRC plot generation for Clarabog - {year}: Processed data not found.")
    else:
        print("No LRC parameters collected for Clarabog.")