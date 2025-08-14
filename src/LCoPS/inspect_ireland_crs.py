
import geopandas as gpd

ireland_shp_path = '/Users/habibw/Documents/LCoPS/IPSM_NLCM/Ireland.shp'

try:
    ireland_gdf = gpd.read_file(ireland_shp_path)
    print("CRS of Ireland.shp:")
    print(ireland_gdf.crs)

except FileNotFoundError:
    print(f"Error: The shapefile '{ireland_shp_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
