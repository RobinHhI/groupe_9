#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production d'une carte à l'échelle des peuplements

Ce script attribuera pour chaque polygone de la BD forêt une classe de peuplement 
(voir nomenclature de la colonne "classif objet" de la figure 2) à partir des pourcentages des classes 
présentes (de carte_essences_echelle_pixel.tif) dans les polygones :

Créé le 31 décembre 2024
"""
import os
import sys
sys.path.append('libsigma')
import re  # pour utiliser fonction regex
import time  # Pour calculer le temp de traitement
import logging
import numpy as np
import geopandas as gpd

# personal librairies
from my_function import get_raster_properties
import classification as cla
import read_and_write as rw

from rasterstats import zonal_stats
import geopandas as gpd

# 1 --- define parameters
sample_folder = "groupe_9/results/data/sample"
classif_folder = 'groupe_9/results/data/classif'
sample_filename = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.shp")
classif_filename = os.path.join(classif_folder, 'carte_essences_echelle_pixel.tif')

#pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs = get_raster_properties(classif_filename)

# pixel_surf = pixel_width * pixel_height
pixel_surf = 100
pourcentage_categorie = 75.0
surf_mini = 20000
feuillus_ilots = 16
melange_feuillus = 15
coniferes_ilots = 27
melange_coniferes = 26
melange_conif_prep_feuil = 28
melange_feuil_prep_conif = 29

np_pourcentage_categorie = np.zeros((26,2), dtype=float)

gdf_sample = gpd.read_file(sample_filename)

# Calcul des statistiques catégoriques
stats = zonal_stats(
    gdf_sample,
    classif_filename,
    all_touched = True,
    categorical = True,  # Active la table de fréquences par catégorie
    nodata = 0
)

code_predict = [] 
code_surface = []
code_percent_conif = []
code_percent_feuil = []
for i, zone_stats in enumerate(stats):
    total_pixels = sum(zone_stats.values())
    
    pourcentage_feuillus = 0
    pourcentage_coniferes = 0
    peuplement = 0
    np_pourcentage_categorie[np_pourcentage_categorie > 0.0] = 0.0
    
    for category, count in zone_stats.items():
        percentage = (count / total_pixels) * 100.0
        if category > 10 and category < 15 :
            np_pourcentage_categorie[int(category), 0] += percentage
            pourcentage_feuillus += percentage
        elif category > 20 and category < 26 :
            np_pourcentage_categorie[int(category), 1] += percentage
            pourcentage_coniferes += percentage
    
    surface = total_pixels * pixel_surf
    if (total_pixels * pixel_surf) < surf_mini :
        if pourcentage_feuillus > pourcentage_categorie :
            peuplement = feuillus_ilots
        elif pourcentage_coniferes > pourcentage_categorie :
            peuplement = coniferes_ilots
        elif pourcentage_coniferes > pourcentage_feuillus :
                peuplement = melange_conif_prep_feuil
        elif (pourcentage_coniferes < pourcentage_feuillus) :
                peuplement = melange_feuil_prep_conif   
    else : 
        indice_feuil = np_pourcentage_categorie[0].argmax()
        indice_conif = np_pourcentage_categorie[1].argmax()
        
        if  np_pourcentage_categorie[indice_feuil, 0] > pourcentage_categorie :            
            peuplement = indice_feuil
        elif np_pourcentage_categorie[indice_conif, 0] > pourcentage_categorie :
            peuplement = indice_conif
        elif pourcentage_feuillus > pourcentage_categorie :
                peuplement = melange_feuillus
        elif pourcentage_coniferes > pourcentage_categorie :
                peuplement = melange_coniferes
        elif pourcentage_coniferes > pourcentage_feuillus :
                peuplement = melange_conif_prep_feuil
        elif (pourcentage_coniferes < pourcentage_feuillus) :
                peuplement = melange_feuil_prep_conif   
                  
    code_predict.append(peuplement)
    code_surface.append(surface)
    code_percent_conif.append(pourcentage_coniferes)
    code_percent_feuil.append(pourcentage_feuillus)

column_exist = gdf_sample.get(["code_predi"])

if column_exist is not None : 
    gdf_sample = gdf_sample.drop(column_exist, axis=1)

gdf_sample.insert(7,"code_predit", code_predict)   
                        
#gdf_sample.insert(8,"surface", code_surface)
#gdf_sample.insert(9, "taux_feuil", code_percent_feuil)
#gdf_sample.insert(10, "taux_conif", code_percent_conif)
#gdf_sample.to_file("groupe_9/results/data/sample/test.shp")    
gdf_sample.to_file(sample_filename)
    