#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des échantillons de la BD forêt

Ce script permet de réaliser une analyse des échantillons de la BD forêt filtrée.

1. Produit un diagramme bâton du nombre de polygones par classe.
2. Produit un violin plot de la distribution du nombre de pixels par classe de polygone.

Créé le 5 décembre 2024
"""

from my_function_robin import sample_data_analysis

shapefile_path = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes_a_conserver = [
    "Autres feuillus", "Chêne", "Robinier", "Peupleraie", 
    "Autres conifères autre que pin", "Autres pin", 
    "Douglas", "Pin laricio ou pin noir", "Pin maritime"
]
output_dir = "groupe_9/results/figure"

sample_data_analysis(shapefile_path, raster_path, classes_a_conserver, output_dir)
