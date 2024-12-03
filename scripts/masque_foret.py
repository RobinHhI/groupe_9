# Importation des librairies
from my_function import reproject_raster_to_memory, get_raster_properties
import sys
import os
from osgeo import gdal
gdal.UseExceptions()
sys.path.append('libsigma')
# Importation des fonctions personnalisées

# Chemins d'accès relatifs
in_vector = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
ref_image = "data/rasters/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"
out_image = "groupe_9/results/data/img_pretraitees/mask_forest.tif"

# Reprojetage du raster de référence en mémoire
print("Reprojection du raster de référence en mémoire...")
reprojected_raster = reproject_raster_to_memory(ref_image, "EPSG:2154")

# Extraire les propriétés du raster reprojeté
pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs = get_raster_properties(
    reprojected_raster)

# Aligner les valeurs d'emprise à la résolution du raster
xmin_aligned = xmin - (xmin % pixel_width)
ymin_aligned = ymin - (ymin % pixel_width)
xmax_aligned = xmax + (pixel_width - (xmax % pixel_width))
ymax_aligned = ymax + (pixel_width - (ymax % pixel_width))

# Construire la commande pour rasteriser avec gdal_rasterize
cmd_pattern = (
    "gdal_rasterize -burn 1 "  # Valeur pour la zone de forêt
    "-init 0 "  # Valeur pour la zone hors forêt
    "-a_nodata 99 "  # NoData défini
    "-te {xmin} {ymin} {xmax} {ymax} "  # Emprise alignée
    "-tr {pixel_width} {pixel_width} "  # Résolution (carrée)
    "-ot Byte -of GTiff "  # Type de données 8 bits et format GTiff
    "-co COMPRESS=LZW "  # Compression pour économiser de l'espace
    "{in_vector} {out_image}"
)

# Remplir les paramètres de la commande
cmd = cmd_pattern.format(
    xmin=xmin_aligned,
    ymin=ymin_aligned,
    xmax=xmax_aligned,
    ymax=ymax_aligned,
    pixel_width=pixel_width,
    in_vector=in_vector,
    out_image=out_image,
)

# Exécuter la commande
print("Création de la masque forêt avec gdal_rasterize...")
os.system(cmd)

# Confirmer la fin de l'exécution
print("Masque forêt compressé et enregistré dans :", out_image)
