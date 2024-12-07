#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de la phénologie des peuplements purs

Ce script permet de réaliser une analyse de NDVI de peuplements purs

Il produit des subplots de séries temporelles de NDVI pour les classes indiquées

Créé le 6 décembre 2024
"""

from my_function_robin import analyse_ndvi_par_classe


raster_file = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
shapefile = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
classes_selection = ["Chêne", "Robinier", "Peupleraie", "Douglas", "Pin laricio ou pin noir", "Pin maritime"]
stats = analyse_ndvi_par_classe(raster_file, shapefile, classes_selection, nom_champ="Nom", output_plot="groupe_9/results/figure/temp_mean_ndvi.png")