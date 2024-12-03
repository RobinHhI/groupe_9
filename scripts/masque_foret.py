from osgeo import gdal, ogr
import os

# Chemins d'accès relatifs
in_vector = "data/vecteurs/FORMATION_VEGETALE.shp"
ref_image = "data/rasters/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"
out_image = "groupe_9/results/data/img_pretraitees/mask_forest.tif"

def rasterize_with_exact_grid(vector_path, ref_raster_path, output_raster_path):
    """Rasterise un vecteur en suivant exactement les dimensions et alignements du raster de référence."""
    # Charger le raster de référence pour obtenir les dimensions et la grille
    ref_ds = gdal.Open(ref_raster_path)
    if not ref_ds:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster de référence : {ref_raster_path}")
    
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize

    # Charger le fichier vecteur
    vector_ds = ogr.Open(vector_path)
    if not vector_ds:
        raise FileNotFoundError(f"Impossible d'ouvrir le fichier vecteur : {vector_path}")
    
    vector_layer = vector_ds.GetLayer()

    # Créer un raster vide avec les dimensions et la grille du raster de référence
    driver = gdal.GetDriverByName("GTiff")
    target_ds = driver.Create(output_raster_path, xsize, ysize, 1, gdal.GDT_Byte)
    target_ds.SetProjection(ref_proj)
    target_ds.SetGeoTransform(ref_geotrans)

    # Forets = 1
    gdal.RasterizeLayer(
        target_ds, [1], vector_layer,
        burn_values=[1],
        options=["COMPRESS=LZW"]  # Option de compression
    )

    # Nettoyage
    target_ds = None
    ref_ds = None
    vector_ds = None
    print(f"Masque forêt compressé et enregistré dans {output_raster_path}")

# Appeler la fonction
rasterize_with_exact_grid(in_vector, ref_image, out_image)
