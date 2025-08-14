
import geopandas as gpd

geojason_path = '/Users/habibw/Documents/LCoPS/IPSM_NLCM/ipsm_nlcm.geojson'

try:
    gdf = gpd.read_file(geojason_path)
    print("Columns in ipsm_nlcm.geojson:")
    print(gdf.columns.tolist())

except Exception as e:
    print(f"An error occurred: {e}")
