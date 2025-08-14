
import geopandas as gpd

shapefile_path = '/Users/habibw/Documents/LCoPS/IPSM_NLCM/IPSM_NLCM.shp'

try:
    gdf = gpd.read_file(shapefile_path)
    print("Columns in IPSM_NLCM.shp:")
    print(gdf.columns.tolist())

except Exception as e:
    print(f"An error occurred: {e}")
