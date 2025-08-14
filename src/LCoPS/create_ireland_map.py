
import geopandas as gpd
import matplotlib.pyplot as plt
import scienceplots
import os
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

# Define file paths
ireland_shp_path = '/Users/habibw/Documents/LCoPS/IPSM_NLCM/Ireland.shp'
output_dir = '/Users/habibw/Documents/LCoPS/'
output_map_path = os.path.join(output_dir, 'ireland_map.png')

try:
    # Load Ireland shapefile
    ireland_gdf = gpd.read_file(ireland_shp_path)

    # Reproject to Irish Transverse Mercator (ITM) for accurate representation
    ireland_gdf = ireland_gdf.to_crs(epsg=2157)

    # Set the SciencePlots style for Nature publication
    plt.style.use(['science', 'nature', 'no-latex'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 12)) # Adjust figure size for map

    # Plot Ireland
    ireland_gdf.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

    # Set aspect ratio to equal for accurate map representation
    ax.set_aspect('equal')

    # Remove title and axis
    ax.set_axis_off()

    # Add Scale Bar
    ax.add_artist(ScaleBar(dx=1, units="m", location='lower left', 
                           frameon=False, scale_loc='bottom', 
                           font_properties={'size': '12'}, 
                           color='black', box_alpha=0, 
                           border_pad=0.5, sep=5))

    # Add North Arrow manually
    x_arrow, y_arrow = 0.95, 0.95 # Position in axes fraction (further into corner)
    arrow_length = 0.05 # Length of the arrow in axes fraction (smaller)
    arrow_width = 0.01 # Width of the arrow in axes fraction (smaller)

    arrow = FancyArrowPatch((x_arrow, y_arrow - arrow_length/2), (x_arrow, y_arrow + arrow_length/2),
                            mutation_scale=200, fc="black", ec="black",
                            arrowstyle='-|>', lw=1.5, transform=ax.transAxes)
    ax.add_patch(arrow)
    ax.text(x_arrow, y_arrow + arrow_length/2 + 0.01, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=14, color='black')

    # Create a dummy legend handle for the Ireland boundary if needed
    from matplotlib.patches import Patch
    legend_handles = [Patch(color='lightgray', label='Ireland')]
    ax.legend(handles=legend_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=12)

    # Add grid with markers and labels on edges
    # Define transformer for ITM to Lat/Lon conversion
    transformer = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    def format_lon(x, pos):
        lon, lat = transformer.transform(x, ax.get_ylim()[0]) # Use a fixed y for transformation
        degrees = int(lon)
        minutes = int(abs(lon - degrees) * 60)
        seconds = int(abs(lon - degrees) * 3600 % 60)
        return f'{degrees}°{minutes}′{seconds}″' + ('E' if degrees >= 0 else 'W')

    def format_lat(y, pos):
        lon, lat = transformer.transform(ax.get_xlim()[0], y) # Use a fixed x for transformation
        degrees = int(lat)
        minutes = int(abs(lat - degrees) * 60)
        seconds = int(abs(lat - degrees) * 3600 % 60)
        return f'{degrees}°{minutes}′{seconds}″' + ('N' if degrees >= 0 else 'S')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.tick_params(axis='both', which='both', 
                   bottom=True, top=True, left=True, right=True, 
                   labelbottom=True, labelleft=True, labeltop=False, labelright=False, 
                   direction='out', labelsize=10) # Show ticks and labels on outside
    
    # Ensure labels are horizontal
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), rotation=0, va='center')

    ax.grid(True, linestyle='-', alpha=0.5, color='gray', lw=0.5) # Explicitly set grid properties

    ax.set_facecolor('white') # Ensure background is white

    plt.tight_layout()
    plt.savefig(output_map_path, dpi=300)
    print(f"Map saved to: {output_map_path}")

except FileNotFoundError:
    print(f"Error: The shapefile '{ireland_shp_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
