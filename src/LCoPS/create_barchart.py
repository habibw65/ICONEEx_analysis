import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np
from matplotlib.patches import Patch

# Define the path to the Excel file
excel_file_path = '/Users/habibw/Documents/LCoPS/Book2.xlsx'
output_dir = '/Users/habibw/Documents/LCoPS'
output_plot_path = os.path.join(output_dir, 'landcover_barchart.png')

try:
    # Read the Excel file
    df = pd.read_excel(excel_file_path)

    # Assuming the first column is land cover and the second is area
    land_cover_col = df.columns[0]
    area_col = df.columns[1]

    # Exclude the row where land_cover_col contains 'Total'
    # Fill NaN values with empty string before converting to string and checking for 'Total'
    df = df[df[land_cover_col].fillna('').astype(str).str.contains('Total') == False]

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
        'Broadleaved Forest': 'Forests',
        'Scrub': 'Scrub',
        'Woodland': 'Forests',
        'Hedgerows': 'Hedgerows',
        'Dry Grasslands': 'Grasslands',
        'Broadleaved Forest and Woodland': 'Forests',
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

    print(f"Unique land cover values after initial filter: {df[land_cover_col].unique()}")
    print(f"Keys in land_cover_to_category: {land_cover_to_category.keys()}")

    # Map land cover classes to their category colors
    df['category'] = df[land_cover_col].apply(lambda x: land_cover_to_category.get(str(x), 'Others'))

    print(f"Unique categories after mapping: {df['category'].unique()}")

    colors = df['category'].map(category_color_map)

    print(f"Colors after final mapping: {colors.unique()}")

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