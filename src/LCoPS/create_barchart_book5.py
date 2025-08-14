
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np
from matplotlib.patches import Patch

# Define the path to the Excel file (not used for data reading in this version)
excel_file_path = '/Users/habibw/Documents/LCoPS/Book5.xlsx' # Just for reference
output_dir = '/Users/habibw/Documents/LCoPS'
output_plot_path = os.path.join(output_dir, 'landcover_barchart_book5.png')

try:
    # Define the data for the bar chart directly
    data = {
        'Land cover': ['Amenity Grassland', 'Artificial Waterbodies', 'Bare Peat', 'Bare Soil and Disturbed Ground',
                       'Blanket Bog', 'Bracken', 'Broadleaved Forest and Woodland', 'Buildings', 'Burnt Areas',
                       'Coastal Sediments', 'Coniferous Forest', 'Cultivated Land', 'Cutover Bog', 'Dry Grassland',
                       'Dry Heath', 'Exposed Rock and Sediments', 'Fens', 'Hedgerows', 'Improved Grassland',
                       'Lakes and Ponds', 'Marine Water', 'Mixed Forest', 'Mudflats', 'Other Artificial Surfaces',
                       'Raised Bog', 'Rivers and Streams', 'Saltmarsh', 'Sand Dunes', 'Scrub', 'Swamp',
                       'Transitional Forest', 'Transitional Waterbodies', 'Treelines', 'Ways', 'Wet Grassland',
                       'Wet Heath'],
        'Area (ha)': [204.89, 242.52, 1281.40, 206.92, 20921.54, 5517.93, 6851.09, -258.76, 219.01, 18.21,
                      8357.85, -1510.02, 6512.46, 6240.26, 25781.56, 484.13, 669.68, 2404.14, 14078.42, 1154.48,
                      -865.66, 2618.10, -46.61, -403.32, 2430.01, 1434.62, 85.30, 197.35, 8999.62, 294.79,
                      5850.74, -6.39, 82.92, 951.16, 54128.37, 32495.96]
    }
    df = pd.DataFrame(data)

    land_cover_col = 'Land cover'
    area_col = 'Area (ha)'

    # Sort the DataFrame by area in descending order
    df = df.sort_values(by=area_col, ascending=False)

    # Define the category mapping and their colors
    category_color_map = {
        'Grasslands': '#f7ea48',
        'Near-natural peatlands': '#3b73b5',
        'Forests': '#1a9c78',
        'Heath': '#d984c0',
        'Cutover bogs': '#f0a13c',
        'Bare peat': '#cc572f',
        'Scrub': '#e3d580',
        'Hedgerows': '#712b6b',
        'Others': '#bfbfbf' # This will be the default for unmapped categories
    }

    # Define a mapping from land cover class to its category
    land_cover_to_category = {
        'Wet Grassland': 'Grasslands',
        'Improved Grassland': 'Grasslands',
        'Blanket Bog': 'Near-natural peatlands',
        'Transitional Forest': 'Forests',
        'Wet Heath': 'Heath',
        'Dry Heath': 'Heath',
        'Cutover Bog': 'Cutover bogs',
        'Coniferous Forest': 'Forests',
        'Bare Peat': 'Bare peat',
        'Raised Bog': 'Near-natural peatlands',
        'Broadleaved Forest and Woodland': 'Forests',
        'Scrub': 'Scrub',
        'Woodland': 'Forests',
        'Hedgerows': 'Hedgerows',
        'Dry Grassland': 'Grasslands',
        'Ways': 'Others',
        'Bracken': 'Others',
        'Lakes and Ponds': 'Others',
        'Mixed Forest': 'Others',
        'Rivers and Streams': 'Others',
        'Cultivated Land': 'Others',
        'Exposed Rock and Sediments': 'Others',
        'Other Artificial Surfaces': 'Others',
        'Treelines': 'Others',
        'Amenity Grassland': 'Others',
        'Bare Soil and Disturbed Ground': 'Others',
        'Artificial Waterbodies': 'Others',
        'Buildings': 'Others',
        'Fens': 'Others',
        'Burnt Areas': 'Others',
        'Swamp': 'Others',
        'Sand Dunes': 'Others',
        'Saltmarsh': 'Others',
        'Marine Water': 'Others',
        'Coastal Sediments': 'Others',
        'Mudflats': 'Others',
        'Transitional Waterbodies': 'Others',
        'nan': 'Others' # Explicitly map string 'nan' to 'Others'
    }

    # Map land cover classes to their category colors
    df['category'] = df[land_cover_col].apply(lambda x: land_cover_to_category.get(str(x), 'Others'))
    colors = df['category'].map(category_color_map)

    # Set the SciencePlots style for Nature publication
    plt.style.use(['science', 'nature', 'no-latex'])

    # --- Bar Chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(20, 11.25)) # Increased figure width and height proportionally
    bars = ax_bar.bar(df[land_cover_col], df[area_col], color=colors)

    # Add labels for bar chart
    ax_bar.set_xlabel('Land Cover Type', fontsize=18)
    ax_bar.set_ylabel('Area (ha)', fontsize=18)

    # Rotate x-axis labels and increase font size
    ax_bar.tick_params(axis='x', labelsize=18) # Adjusted labelsize
    ax_bar.tick_params(axis='y', labelsize=18) # Adjusted labelsize
    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')

    # Add grid for better readability
    ax_bar.grid(True, linestyle='--', alpha=0.7)

    # Create a legend
    legend_handles = []
    for category, color in category_color_map.items():
        legend_handles.append(Patch(color=color, label=category))

    ax_bar.legend(handles=legend_handles, loc='upper right', facecolor='white', edgecolor='black', fontsize=16, title_fontsize=18, frameon=True)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"Bar chart saved to: {output_plot_path}")

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
