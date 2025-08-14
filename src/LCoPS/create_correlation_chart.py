import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os

# Define the output directory
output_dir = '/Users/habibw/Documents/LCoPS/'
output_plot_path = os.path.join(output_dir, 'landcover_correlation_chart.png')

try:
    # Define the data for the correlation chart
    data = {
        'Land cover': ['Grasslands', 'Forests', 'Near-natural peatlands', 'Heath', 'Cutover', 'Others', 'Bare peat', 'Scrub', 'Hedgerows', 'Bracken'],
        'Proportion on all peat soils': [30.6, 20.1, 16.2, 15.0, 6.4, 4.9, 3.1, 2.2, 1.4, 0.0],
        'Proportion on shallow peat-peaty soils': [37.1, 18.7, 4.6, 26.3, 0.0, 4.9, 0.0, 4.0, 1.5, 2.9]
    }
    df_corr = pd.DataFrame(data)

    # Define the category mapping and their colors (consistent with previous charts)
    category_color_map = {
        'Grasslands': '#f7ea48',
        'Near-natural peatlands': '#3b73b5',
        'Forests': '#1a9c78',
        'Heath': '#d984c0',
        'Cutover': '#f0a13c',
        'Bare peat': '#cc572f',
        'Scrub': '#e3d580',
        'Hedgerows': '#712b6b',
        'Others': '#bfbfbf',
        'Bracken': '#8B4513' # Added Bracken color
    }

    # Map colors to the land cover types
    colors = [category_color_map.get(lc, '#000000') for lc in df_corr['Land cover']] # Default to black if not found

    # Set the SciencePlots style for Nature publication
    plt.style.use(['science', 'nature', 'no-latex'])

    fig, ax = plt.subplots(figsize=(16, 9)) # 16:9 aspect ratio

    # Create the scatter plot
    ax.scatter(df_corr['Proportion on all peat soils'], df_corr['Proportion on shallow peat-peaty soils'],
               color=colors, s=200) # Increased marker size

    # Add land cover class names as labels next to each point
    for i, row in df_corr.iterrows():
        x_offset, y_offset = 8, 8 # Default offset
        ha_align = 'left' # Default horizontal alignment

        if row['Land cover'] == 'Bracken':
            x_offset = 3 # Move 2 points to the left
            y_offset = 14 # Move 4 points above
        elif row['Land cover'] == 'Bare peat':
            x_offset = 4 # Move 4 points left
            y_offset = 8 # Keep default y_offset
        elif row['Land cover'] == 'Forests':
            x_offset = 8 # Keep default x_offset
            y_offset = 6 # Move 2 points down

        ax.annotate(row['Land cover'], (row['Proportion on all peat soils'], row['Proportion on shallow peat-peaty soils']),
                    textcoords="offset points", xytext=(x_offset, y_offset), ha=ha_align, va='bottom', fontsize=16) # Increased fontsize

    # Set axis limits
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    # Add y=x dotted correlation line
    ax.plot([0, 50], [0, 50], 'k--', lw=1)
    ax.text(0.85, 0.95, 'y=x', transform=ax.transAxes, ha='right', va='top', fontsize=16) # Increased fontsize

    # Add labels
    ax.set_xlabel('Proportion on all peat soils (%)', fontsize=18)
    ax.set_ylabel('Proportion on shallow peat-peaty soils (%)', fontsize=18)

    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"Correlation chart saved to: {output_plot_path}")

except Exception as e:
    print(f"An error occurred: {e}")