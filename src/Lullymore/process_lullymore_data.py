import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import calendar # For month names

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


if __name__ == '__main__':
    lullymore_dir = "/Users/habibw/Documents/Lullymore"

    # Generate small multiples fingerprint plots for Lullymore
    # Assuming processed_hesseflux_YYYY.csv files already exist
    processed_2022_exists = os.path.exists(os.path.join(lullymore_dir, "processed_hesseflux_2022.csv"))
    processed_2023_exists = os.path.exists(os.path.join(lullymore_dir, "processed_hesseflux_2023.csv"))

    years_to_plot = []
    if processed_2022_exists:
        years_to_plot.append(2022)
    if processed_2023_exists:
        years_to_plot.append(2023)

    if years_to_plot:
        generate_small_multiples_fingerprint(lullymore_dir, years_to_plot)
    else:
        print(f"Skipping fingerprint plot generation for Lullymore: No processed data found.")
