# Importation des librairies
from my_function import create_forest_mask

# Chemins d'accès relatifs
mask_vector = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
reference_image = "data/images/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"
clip_vector = "data/vecteurs/emprise_etude.shp"
output_image = "groupe_9/results/data/img_pretraitees/mask_forest.tif"

# Créer le masque
create_forest_mask(mask_vector, reference_image, clip_vector, output_image)
