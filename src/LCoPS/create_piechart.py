import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np # Import numpy for calculations
from matplotlib.patches import Patch

# Define the path to the Excel file (still needed for category_color_map)
excel_file_path = '/Users/habibw/Documents/LCoPS/Book2.xlsx'
output_dir = '/Users/habibw/Documents/LCoPS'
output_pie_chart_path = os.path.join(output_dir, 'landcover_piechart.png')

try:
    # Define the aggregated data for the pie chart directly
    pie_data_raw = {
        'Grasslands': 30.6,
        'Forests': 20.1,
        'Near-natural peatlands': 16.2,
        'Heath': 15.0,
        'Cutover': 6.4,
        'Others': 4.9, 
        'Bare peat': 3.1,
        'Scrub': 2.2,
        'Hedgerows': 1.4
    }
    pie_data = pd.Series(pie_data_raw)

    # Define the category mapping and their colors (ensure consistency)
    category_color_map = {
        'Grasslands': '#f7ea48',
        'Near-natural peatlands': '#3b73b5',
        'Forests': '#1a9c78',
        'Heath': '#d984c0',
        'Cutover': '#f0a13c',
        'Bare peat': '#cc572f',
        'Scrub': '#e3d580',
        'Hedgerows': '#712b6b',
        'Others': '#bfbfbf' # Renamed from 'Other land covers'
    }

    # Create a list of categories in the desired legend order
    legend_order = ['Grasslands', 'Forests', 'Near-natural peatlands', 'Heath', 'Cutover', 'Bare peat', 'Scrub', 'Hedgerows', 'Others']

    # Filter pie_data to include only categories present in legend_order and reorder
    pie_data = pie_data.reindex(legend_order).dropna()

    # Ensure all categories in pie_data have a color defined
    pie_colors = [category_color_map.get(cat, '#bfbfbf') for cat in pie_data.index] # Default to grey if not found

    # Set the SciencePlots style for Nature publication
    plt.style.use(['science', 'nature', 'no-latex'])

    fig_pie, ax_pie = plt.subplots(figsize=(17.5, 15)) # Increased width and height for legend space

    # Custom autopct function to adjust label position for small slices
    def autopct_custom(pct):
        return ('%1.1f%%' % pct) if pct > 5 else '' # Only show label if > 5%

    wedges, texts, autotexts = ax_pie.pie(pie_data, labels=None, autopct=autopct_custom,
                                          startangle=90, colors=pie_colors,
                                          textprops={'fontsize': 32}) # Increased fontsize
    ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    # Manually place labels for small slices outside
    for i, autotext in enumerate(autotexts):
        if autotext.get_text() == '': # This means the percentage was <= 5%
            pct = pie_data.iloc[i] # Use the percentage directly
            ang = (wedges[i].theta2 + wedges[i].theta1) / 2. # Angle of the wedge
            x = np.cos(np.deg2rad(ang)) # X-coordinate of the label
            y = np.sin(np.deg2rad(ang)) # Y-coordinate of the label

            # Determine horizontal alignment based on angle
            horizontalalignment = "left" if x > 0 else "right"
            connectionstyle = "arc3,rad=0.1"

            # Adjust xytext for label distance, making it closer
            xytext_x = 1.1 * x # Bring labels closer
            xytext_y = 1.1 * y # Bring labels closer

            # Specific adjustment for 2.2% (assuming it's the one causing overlap)
            if abs(pct - 2.2) < 0.1: # Check if it's approximately 2.2%
                xytext_y += 0.05 # Move it slightly up to avoid overlap
            elif abs(pct - 1.4) < 0.1: # Specific adjustment for 1.4% (Hedgerows)
                xytext_x -= 0.03 # Move it slightly to the left
                horizontalalignment = "right" if x < 0 else "left"

            ax_pie.annotate('{:.1f}%'.format(pct), xy=(x, y), xytext=(xytext_x, xytext_y),
                         textcoords="data", fontsize=24, color='black',
                         arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='black', lw=0.5),
                         horizontalalignment=horizontalalignment, verticalalignment='center')

    # Create a legend
    legend_handles = []
    for category in legend_order:
        if category in pie_data.index: # Only add if the category is present in the data
            legend_handles.append(Patch(color=category_color_map[category], label=category))

    ax_pie.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), facecolor='white', edgecolor='black', fontsize=24, title_fontsize=28, frameon=True)

    plt.tight_layout()
    plt.savefig(output_pie_chart_path, dpi=300)
    print(f"Pie chart saved to: {output_pie_chart_path}")

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")