import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from rasterio.warp import transform, calculate_default_transform

# Charger le fichier de polygones de forêt 
forest_polygons = gpd.read_file("data_projet/vecteurs/FORMATION_VEGETALE.shp")  

# Vérifier si les polygones sont en EPSG:2154
if forest_polygons.crs != "EPSG:2154":
    forest_polygons = forest_polygons.to_crs("EPSG:2154")

# Charger l'image Sentinel-2 pour obtenir l'emprise spatiale et la résolution
image_s2_path = "data_projet/rasters/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"  
with rasterio.open(image_s2_path) as src:
    transform_s2 = src.transform
    width = src.width
    height = src.height
    crs_s2 = src.crs  
    dtype = src.dtypes[0]

# Calculer la transformation nécessaire pour reprojeter l'image Sentinel-2 vers EPSG:2154
transform_2154, width_2154, height_2154 = calculate_default_transform(
    crs_s2, "EPSG:2154", width, height, *src.bounds
)

# Reprojeter les polygones de forêt pour qu'ils soient alignés avec le raster 
forest_polygons = forest_polygons.to_crs("EPSG:2154")

# Créer un masque pour la couche de forêt (invert=True pour que la forêt soit en 1 et le reste en 0)
mask = geometry_mask(forest_polygons.geometry, transform=transform_2154, invert=True, out_shape=(height_2154, width_2154))

# Créer un raster pour les zones de forêt (1) et hors forêt (0)
raster_mask = np.zeros((height_2154, width_2154), dtype=np.uint8)
raster_mask[mask] = 1  # Zone de forêt = 1, hors forêt = 0

# Sauvegarder le résultat en GeoTIFF
output_path = "groupe_9/results/data/img_pretraitees/masque_foret.tif"
with rasterio.open(
    output_path, 
    'w', 
    driver='GTiff', 
    count=1, 
    dtype=np.uint8, 
    crs="EPSG:2154", 
    transform=transform_2154, 
    width=width_2154, 
    height=height_2154
) as dst:
    dst.write(raster_mask, 1)
 
print(f"Le masque raster a été sauvegardé sous {output_path}")
