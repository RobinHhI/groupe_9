#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a forest mask using the first available FRE_B2 image as reference.

Created on Dec 03, 2024
"""

# Importation des librairies
from my_function import create_forest_mask, find_raster_bands, log_error_and_raise


# Chemins d'accès relatifs
mask_vector = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
<< << << < HEAD
input_folder = "data/images/"
band_prefix = "FRE_B2"

# Trouver l'image de réference
reference_images = find_raster_bands(input_folder, [band_prefix])

if not reference_images:
    log_error_and_raise(
        f"No images with prefix {band_prefix} found in {input_folder}")
else:
    reference_image = reference_images[0]

clip_vector = "data/project/emprise_etude.shp"
== == == =
reference_image = "data/images/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"
clip_vector = "data/vecteurs/emprise_etude.shp"
>>>>>> > 484ef419a5aa257a1b010550f25fcb8bb3840990
output_image = "groupe_9/results/data/img_pretraitees/mask_forest.tif"

# Créer le masque
create_forest_mask(mask_vector, reference_image, clip_vector, output_image)
