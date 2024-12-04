#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prétraitement des données Sentinel-2

Ce script effectue les étapes nécessaires pour traiter les bandes Sentinel-2 :
1. Ajuste de la masque forêt en fonction des propriétés des rasters Sentinel.
2. Traitement des bandes Sentinel-2 (reprojection, recadrage, rechantillonnage).
3. Fusion des bandes pour produire une image multitemporelle à 60 bandes.

Créé le 3 décembre 2024
"""

# Importation des librairies
from my_function import find_raster_bands, reproject_raster, clip_raster_to_extent, resample_raster, apply_mask, merge_rasters

# Parameters
clip_vector = "data/vecteurs/emprise_etude.shp"
mask_raster = "groupe_9/results/data/img_pretraitees/mask_forest.tif"
input_folder = "data/rasters/"
band_prefixes = ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B8"]
output_path = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_allbands.tif"

# Process bands
band_files = find_raster_bands(input_folder, band_prefixes)
processed_rasters = []

for band in band_files:
    reprojected = reproject_raster(band, "EPSG:2154")
    clipped = clip_raster_to_extent(reprojected, clip_vector)
    resampled = resample_raster(clipped, 10)
    masked = apply_mask(resampled, mask_raster)
    processed_rasters.append(masked)

# Merge all bands
merge_rasters(processed_rasters, output_path)
