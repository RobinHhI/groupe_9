#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de la variabilité spectrale des classes BD forêt

Ce script réalise deux analyses :
1. Calcul de la distance moyenne au centroïde global pour chaque classe.
2. Calcul de la distance moyenne au centroïde local (par polygone) pour chaque classe.

Les graphiques sont produits en bâtons pour la variabilité globale
et en "violin plots" pour la variabilité locale (pour chaque polygon)

"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from my_function_lucas import calculate_global_centroid_distance, calculate_local_centroid_distance

# Paramètres
shapefile_path = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
output_folder = "results/figure"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Charger les données vectorielles
logging.info("Chargement du shapefile...")
gdf = gpd.read_file(shapefile_path)

# Filtrer les classes rouges (essences simples) et bleues (essences mélangées)
classes_rouges = ["Autres feuillus", "Chêne", "Robinier", "Peupleraie",
                  "Douglas", "Pin laricio ou pin noir", "Pin maritime"]
classes_bleues = ["Mélange de feuillus", "Mélange conifères",
                  "Mélange de conifères prépondérants et feuillus",
                  "Mélange de feuillus préponderants conifères"]

gdf = gdf[gdf['Nom'].isin(classes_rouges + classes_bleues)]
