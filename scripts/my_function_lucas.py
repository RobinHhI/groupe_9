#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions pour l'Analyse de Variabilité Spectrale

Ce module fournit des fonctions pour extraire les valeurs NDVI à l'échelle du pixel,
par classe ou par polygone, calculer le centroïde et les distances au centroïde des classes 
et des polygones, ainsi que pour générer des graphiques.

Créé le : 03 Décembre 2024
Dernière modification : 11 Décembre 2024

@Auteurs : Alban Dumont, Lucas Lima, Robin Heckendorn
Fonction `custom_bg` crée par Marc LANG, Yousra HAMROUNI (marc.lang@toulouse-inp.fr)

Modifications récentes :
- Extraction du NDVI par pixel (au lieu d'utiliser zonal_stats) afin de permettre 
  le calcul des distances internes aux polygones.
- Réorganisation des données NDVI en arrays (N_pixels, N_bandes) pour chaque classe ou polygone.
- Adaptation des fonctions de calcul de distances pour travailler avec des données 2D.
"""

import os
import logging
import sys
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
from affine import Affine
import matplotlib.pyplot as plt


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def log_error_and_raise(message, exception=RuntimeError):
    """
    Enregistre un message d'erreur, inclut le traceback, et lève une exception.

    Paramètres :
    -----------
    message : str
        Message d'erreur à enregistrer et à lever.
    exception : Exception
        Type d'exception à lever (par défaut : RuntimeError).
    """
    logger.error(f"{message}\n{traceback.format_exc()}")
    raise exception(message)


def extraire_valeurs_ndvi_par_classe(shapefile_path, raster_path):
    """
    Extrait les valeurs NDVI pour chaque classe au niveau du pixel, 
    en utilisant la rasterisation du shapefile dissous par classe.

    Paramètres :
    -----------
    shapefile_path : str
        Chemin vers le fichier shapefile contenant les polygones et la colonne 'Nom' pour la classe.
    raster_path : str
        Chemin vers le raster NDVI multi-bandes (GeoTIFF).

    Retourne :
    ---------
    dict
        Un dictionnaire avec, pour chaque classe :
        {
          "total_pixels": int,
          "ndvi_means": array (N_pixels, N_bandes)
        }
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par classe...")
        gdf = gpd.read_file(shapefile_path)
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Dissoudre par classe
        logger.info("Dissolution des polygones par classe...")
        gdf_classes = gdf.dissolve(by='Nom')
        # Ajouter un ID pour chaque classe
        # range(1, len(gdf_classes)+1) assigne un ID unique par ligne
        gdf_classes['ClassID'] = range(1, len(gdf_classes)+1)

        # Sauvegarder temporairement le shapefile dissous
        temp_class_shp = "classes_dissolues.shp"
        gdf_classes.to_file(temp_class_shp)

        # Ouvrir le raster NDVI avec GDAL
        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        # Créer un raster en mémoire pour les classes
        logger.info("Rasterisation des classes...")
        driver = gdal.GetDriverByName('MEM')
        class_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int16)
        class_raster.SetGeoTransform(geotransform)
        class_raster.SetProjection(projection)

        band = class_raster.GetRasterBand(1)
        band.Fill(0)  # 0 = aucune classe
        band.SetNoDataValue(0)

        # Ouvrir le shapefile dissous avec OGR
        ds = ogr.Open(temp_class_shp)
        layer = ds.GetLayer()

        # Rasteriser avec l'attribut ClassID
        gdal.RasterizeLayer(class_raster, [1], layer, options=[
                            "ATTRIBUTE=ClassID"])

        # Lire tout le NDVI
        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()  # (n_bandes, hauteur, largeur)

        # (hauteur, largeur) avec les IDs de classe
        class_arr = class_raster.ReadAsArray()

        resultats_par_classe = {}
        # Pour chaque classe dans gdf_classes
        for idx, row in gdf_classes.iterrows():
            classe = str(idx)  # idx est la classe (Nom)
            class_id = row['ClassID']
            # Créer un masque pour cette classe
            mask_classe = (class_arr == class_id)
            # Extraire les pixels NDVI
            pixels_classe = ndvi_data[:, mask_classe].T  # (N_pixels, n_bandes)
            total_pixels = pixels_classe.shape[0]
            resultats_par_classe[classe] = {
                "total_pixels": total_pixels,
                "ndvi_means": pixels_classe
            }

        logger.info("Extraction des valeurs NDVI par classe terminée.")
        return resultats_par_classe

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors de l'extraction des valeurs NDVI par classe : {e}")


def extraire_valeurs_ndvi_par_polygone(shapefile_path, raster_path):
    """
    Extrait les valeurs NDVI pour chaque polygone au niveau du pixel
    en rasterisant le shapefile avec un ID unique par polygone.

    Paramètres :
    -----------
    shapefile_path : str
        Chemin vers le fichier shapefile contenant les polygones, 
        avec une colonne 'ID' (entier unique) et 'Nom' pour la classe.
    raster_path : str
        Chemin vers le raster NDVI multi-bandes (GeoTIFF).

    Retourne :
    ---------
    dict
        {
           poly_id: {
               "ndvi_means": array (N_pixels_poly, N_bandes),
               "total_pixels": int,
               "class": str
           },
           ...
        }
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par polygone...")
        gdf = gpd.read_file(shapefile_path)
        if 'ID' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'ID' est requise dans le shapefile.")
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Assurer que ID est un entier
        # gdf['ID'] = gdf['ID'].astype(int)  # si nécessaire

        # Sauvegarder (si nécessaire) pour OGR
        temp_poly_shp = "polygones.shp"
        gdf.to_file(temp_poly_shp)

        # Ouvrir le raster NDVI avec GDAL
        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        # Créer un raster en mémoire pour les polygones
        logger.info("Rasterisation des polygones...")
        driver = gdal.GetDriverByName('MEM')
        poly_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int32)
        poly_raster.SetGeoTransform(geotransform)
        poly_raster.SetProjection(projection)

        band = poly_raster.GetRasterBand(1)
        band.Fill(0)  # 0 = aucun polygone
        band.SetNoDataValue(0)

        ds = ogr.Open(temp_poly_shp)
        layer = ds.GetLayer()

        # Rasteriser avec l'attribut 'ID'
        gdal.RasterizeLayer(poly_raster, [1], layer, options=["ATTRIBUTE=ID"])

        # Lire tout le NDVI
        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()  # (n_bandes, hauteur, largeur)

        poly_arr = poly_raster.ReadAsArray()  # (hauteur, largeur) avec l'ID du polygone

        resultats_par_polygone = {}
        # Maintenant, on a un ID par pixel (ou 0 si rien)
        # On va itérer sur les polygones du gdf original pour extraire les pixels
        for i, row in gdf.iterrows():
            poly_id = row['ID']
            poly_class = row['Nom']
            # Masque pour ce polygone
            mask_poly = (poly_arr == poly_id)
            pixels_poly = ndvi_data[:, mask_poly].T  # (N_pixels, n_bandes)
            total_pixels = pixels_poly.shape[0]

            resultats_par_polygone[poly_id] = {
                "ndvi_means": pixels_poly,
                "total_pixels": total_pixels,
                "class": poly_class
            }

        logger.info("Extraction des valeurs NDVI par polygone terminée.")
        return resultats_par_polygone

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors de l'extraction des valeurs NDVI par polygone : {e}")


def calculer_centroide_et_distances(valeurs_ndvi, niveau="classe", classes=None):
    """
    Calcule le centroïde et les distances moyennes au centroïde.

    Paramètres :
    -----------
    valeurs_ndvi : dict
        Dictionnaire contenant les NDVI par classe ou par polygone.
        Les valeurs NDVI doivent être un array (N_observations, N_bandes) où:
        - Si niveau="classe": N_observations = N_pixels de la classe
        - Si niveau="polygone": N_observations = N_pixels du polygone
    niveau : str
        Niveau d'analyse, "classe" ou "polygone".
    classes : dict, optionnel
        Dictionnaire associant chaque polygone (ID) à sa classe.

    Retourne :
    ---------
    dict
        {
          cle: {
             "centroide": array (n_bandes,) ou None,
             "distance_moyenne": float ou None,
             "classe": str ou None
          }
        }
        Si pas de NDVI, distance_moyenne et centroide = None.
    """
    try:
        logger.info(
            f"Calcul du centroïde et distances moyennes au niveau : {niveau}")

        resultats = {}

        for cle, valeurs in valeurs_ndvi.items():
            valeurs_array = valeurs["ndvi_means"]
            if valeurs_array is None or valeurs_array.size == 0:
                logger.warning(f"Aucune valeur NDVI pour {cle}. Ignoré.")
                # On retourne quand même une entrée pour éviter KeyError plus tard
                classe_associee = valeurs.get("class", None)
                if niveau == "polygone" and classes and (cle in classes):
                    classe_associee = classes[cle]

                resultats[cle] = {
                    "centroide": None,
                    "distance_moyenne": None,
                    "classe": classe_associee
                }
                continue

            # Assurer deux dimensions
            if valeurs_array.ndim == 1:
                valeurs_array = valeurs_array[np.newaxis, :]

            centroide = np.mean(valeurs_array, axis=0)  # (n_bandes,)
            distances = np.sqrt(np.sum((valeurs_array - centroide)**2, axis=1))
            distance_moyenne = np.mean(distances)

            classe_associee = valeurs.get("class", None)
            if niveau == "polygone" and classes and (cle in classes):
                classe_associee = classes[cle]

            resultats[cle] = {
                "centroide": centroide,
                "distance_moyenne": distance_moyenne,
                "classe": classe_associee
            }

        logger.info("Calcul du centroïde et distances terminé avec succès.")
        return resultats

    except Exception as e:
        log_error_and_raise(
            f"Erreur lors du calcul du centroïde et des distances : {e}")


def plot_barchart_distance_classes(distances_par_classe, output_path, groupes):
    """
    Génère un diagramme en bâton des distances moyennes au centrôide pour chaque classe.

    Paramètres :
    -----------
    distances_par_classe : dict
        Dictionnaire {classe: distance_moyenne}
    output_path : str
        Chemin de sauvegarde du graphique.
    groupes : dict
        Dictionnaire associant chaque classe à "Pur" ou "Mélange".
    """
    # Préparation des données
    data = [(classe, distance, groupes.get(classe, "Autre"))
            for classe, distance in distances_par_classe.items()]
    data_sorted = sorted(data, key=lambda x: (x[2] != "Pur", x[0]))

    # Décomposition des données pour le graphique
    classes = [item[0] for item in data_sorted]
    distances = [item[1] for item in data_sorted]
    couleurs = ['#FF9999' if item[2] ==
                "Pur" else '#66B3FF' for item in data_sorted]

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dessiner les barres
    bars = ax.bar(classes, distances, color=couleurs, edgecolor="black")

    # Ajouter des étiquettes sur chaque barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 50, f"{height:.2f}",
                ha='center', va='bottom', fontsize=9)

    # Personnalisation des axes et du titre
    ax.set_title("Distance moyenne des pixels au centrôide de leur classe",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Distance moyenne au centrôide", fontsize=12)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)

    # Ajustement de la limite supérieure de Y
    ax.set_ylim(0, max(distances) * 1.2)

    # Ajout d'une légende
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color='#FF9999', edgecolor="black", label="Pur"),
                      plt.Rectangle((0, 0), 1, 1, color='#66B3FF', edgecolor="black", label="Mélange")]
    ax.legend(handles=legend_handles, title="Groupes", loc="upper right")

    # Amélioration de l'affichage
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Sauvegarde du graphique
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")


def plot_violin_distance_polygons(distances_par_polygone, classes, output_path, groupes):
    """
    Génère un violin plot des distances moyennes au centrôide pour chaque polygone,
    séparant les groupes "Pur" et "Mélange".

    Paramètres :
    -----------
    distances_par_polygone : list
        Liste des distances moyennes au centrôide pour chaque polygone.
    classes : list
        Liste des classes associées à chaque polygone (même ordre que distances_par_polygone).
    output_path : str
        Chemin pour sauvegarder l'image.
    groupes : dict
        Dictionnaire associant chaque classe à "Pur" ou "Mélange".
    """
    # Associer distances aux groupes
    data_pur = [dist for dist, classe in zip(
        distances_par_polygone, classes) if groupes.get(classe) == "Pur"]
    data_melange = [dist for dist, classe in zip(
        distances_par_polygone, classes) if groupes.get(classe) == "Mélange"]

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot([data_pur, data_melange],
                          showmeans=True, showmedians=True)

    # Personnalisation des couleurs
    colors = ['red', 'blue']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Ajout des étiquettes
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pur", "Mélange"])
    ax.set_title("Distribution des distances moyennes par polygone",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance moyenne au centrôide", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")
