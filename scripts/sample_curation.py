#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Sélection des Échantillons

Ce script filtre et reclassifie les données de la BD Forêt selon des instructions spécifiques,
puis sauvegarde le résultat sous forme de fichier shapefile.
"""

# Importation des bibliothèques nécessaires
import geopandas as gpd
import os  # Pour gérer les chemins et la création des répertoires

# Importation des fonctions personnalisées
from my_function import filter_and_reclassify, save_vector_file

# Définition des chemins des fichiers
shp_path = "data/vecteurs/FORMATION_VEGETALE.shp"
output_path = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"

# Création du répertoire de sortie si nécessaire
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crée le répertoire si celui-ci n'existe pas déjà
    print(f"Répertoire de sortie créé : {output_dir}")

# Chargement des données de la BD Forêt
print("Chargement des données depuis :", shp_path)
bd_foret = gpd.read_file(shp_path)

# Filtrage et reclassification des données
print("Filtrage et reclassification des données...")
bd_foret_filtered = filter_and_reclassify(bd_foret)

# Sauvegarde du fichier final
print("Sauvegarde des données filtrées et reclassifiées dans :", output_path)
save_vector_file(bd_foret_filtered, output_path)

# Confirmation de la fin de l'exécution
print("Traitement terminé avec succès. Le fichier a été sauvegardé dans :", output_path)
