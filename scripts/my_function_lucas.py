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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Créer des lignes dans la légende

# GDAL configuration
gdal.UseExceptions()

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


def cleanup_temp_files(*base_file_paths):
    """
    Supprime les fichiers temporaires spécifiés avec toutes leurs extensions associées.

    Paramètres :
    -----------
    *base_file_paths : str
        Chemins de base des fichiers à supprimer (sans extension spécifique).
        Ex : "temp_classes_dissolues" supprimera tous les fichiers "temp_classes_dissolues.*".
    """
    extensions = [".shp", ".shx", ".dbf", ".prj",
                  ".cpg", ".qpj", ".fix", ".shp.xml"]
    for base_path in base_file_paths:
        for ext in extensions:
            file_path = f"{base_path}{ext}"
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Fichier temporaire supprimé : {file_path}")
            except Exception as e:
                logger.error(
                    f"Erreur lors de la suppression du fichier {file_path} : {e}")


def extraire_valeurs_ndvi_par_classe(shapefile_path, raster_path, groupes):
    """
    Extrait les valeurs NDVI pour chaque classe au niveau du pixel, 
    mais seulement pour les classes présentes dans le dictionnaire 'groupes'.
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par classe...")
        gdf = gpd.read_file(shapefile_path)
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Filtrer uniquement les classes présentes dans 'groupes'
        classes_valide = set(groupes.keys())
        gdf = gdf[gdf['Nom'].isin(classes_valide)]
        if gdf.empty:
            log_error_and_raise(
                "Aucune classe dans le shapefile ne correspond aux classes du dictionnaire 'groupes'.")

        # Dissoudre par classe
        logger.info("Dissolution des polygones par classe...")
        gdf_classes = gdf.dissolve(by='Nom')
        gdf_classes['ClassID'] = range(1, len(gdf_classes) + 1)

        temp_class_shp = "temp_classes_dissolues.shp"
        gdf_classes.to_file(temp_class_shp)
        logger.info(f"Shapefile temporaire sauvegardé dans : {temp_class_shp}")

        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        logger.info("Rasterisation des classes...")
        driver = gdal.GetDriverByName('MEM')
        class_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int16)
        class_raster.SetGeoTransform(geotransform)
        class_raster.SetProjection(projection)

        band = class_raster.GetRasterBand(1)
        band.Fill(0)  # 0 = aucune classe
        band.SetNoDataValue(0)

        ds = ogr.Open(temp_class_shp)
        layer = ds.GetLayer()

        gdal.RasterizeLayer(class_raster, [1], layer, options=[
                            "ATTRIBUTE=ClassID"])

        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()

        class_arr = class_raster.ReadAsArray()

        resultats_par_classe = {}
        for idx, row in gdf_classes.iterrows():
            classe = str(idx)  # idx est la classe (Nom)
            class_id = row['ClassID']
            mask_classe = (class_arr == class_id)
            pixels_classe = ndvi_data[:, mask_classe].T
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


def extraire_valeurs_ndvi_par_polygone(shapefile_path, raster_path, groupes):
    """
    Extrait les valeurs NDVI pour chaque polygone, uniquement pour les classes présentes dans 'groupes'.
    """
    try:
        logger.info(
            "Chargement du shapefile et du raster NDVI pour extraction par polygone...")
        gdf = gpd.read_file(shapefile_path)
        if 'Nom' not in gdf.columns:
            log_error_and_raise(
                "La colonne 'Nom' est requise dans le shapefile.")

        # Filtrer les classes
        classes_valide = set(groupes.keys())
        gdf = gdf[gdf['Nom'].isin(classes_valide)]
        if gdf.empty:
            log_error_and_raise(
                "Aucune classe dans le shapefile ne correspond aux classes du dictionnaire 'groupes'.")

        if 'PolyIntID' not in gdf.columns:
            gdf['PolyIntID'] = range(1, len(gdf) + 1)

        temp_poly_shp = "temp_polygones_int_id.shp"
        gdf.to_file(temp_poly_shp)
        logger.info(f"Shapefile temporaire sauvegardé dans : {temp_poly_shp}")

        src = gdal.Open(raster_path)
        if src is None:
            log_error_and_raise("Impossible d'ouvrir le raster NDVI.")

        geotransform = src.GetGeoTransform()
        projection = src.GetProjection()
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        n_bandes = src.RasterCount

        logger.info("Rasterisation des polygones...")
        driver = gdal.GetDriverByName('MEM')
        poly_raster = driver.Create('', xsize, ysize, 1, gdal.GDT_Int32)
        poly_raster.SetGeoTransform(geotransform)
        poly_raster.SetProjection(projection)

        band = poly_raster.GetRasterBand(1)
        band.Fill(0)
        band.SetNoDataValue(0)

        ds = ogr.Open(temp_poly_shp)
        layer = ds.GetLayer()

        gdal.RasterizeLayer(poly_raster, [1], layer, options=[
                            "ATTRIBUTE=PolyIntID"])

        logger.info("Lecture de toutes les bandes NDVI...")
        ndvi_data = src.ReadAsArray()

        poly_arr = poly_raster.ReadAsArray()

        resultats_par_polygone = {}
        for i, row in gdf.iterrows():
            poly_int_id = row['PolyIntID']
            poly_class = row['Nom']
            original_id = row['ID'] if 'ID' in gdf.columns else None
            mask_poly = (poly_arr == poly_int_id)
            pixels_poly = ndvi_data[:, mask_poly].T
            total_pixels = pixels_poly.shape[0]

            resultats_par_polygone[poly_int_id] = {
                "ndvi_means": pixels_poly,
                "total_pixels": total_pixels,
                "class": poly_class,
                "original_id": original_id
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
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#FF9999', edgecolor="black", label="Pur"),
                      plt.Rectangle((0, 0), 1, 1, facecolor='#66B3FF', edgecolor="black", label="Mélange")]

    ax.legend(handles=legend_handles, title="Essences", loc="upper right")

    # Amélioration de l'affichage
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Sauvegarde du graphique
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")


def plot_violin_distance_polygons(distances_par_polygone, classes_polygones, output_path, groupes):
    """
    Génère un violin plot des distances moyennes au centrôide par classe, à l'échelle des polygones.
    Les classes pures (Pur) sont affichées en premier (en rouge), et les classes mélange (Mélange) en second (en bleu).
    Un violon par classe.

    Paramètres :
    -----------
    distances_par_polygone : list of float
        Liste des distances moyennes au centrôide pour chaque polygone.
    classes_polygones : list of str
        Liste des classes associées à chaque polygone, dans le même ordre que distances_par_polygone.
    output_path : str
        Chemin pour sauvegarder l'image.
    groupes : dict
        Dictionnaire associant chaque classe à "Pur" ou "Mélange".

    Note :
    Cette fonction suit la même logique de tri que le bar plot : d'abord les classes "Pur" puis "Mélange",
    classées alphabétiquement dans chaque groupe.
    """

    plt.style.use('ggplot')

    # Agréger les distances par classe
    distances_par_classe = {}
    for dist, cls in zip(distances_par_polygone, classes_polygones):
        if cls not in distances_par_classe:
            distances_par_classe[cls] = []
        distances_par_classe[cls].append(dist)

    # Déterminer le groupe (Pur/Mélange) de chaque classe
    classes_data = [(cls, np.median(distances_par_classe[cls]), groupes.get(cls, "Autre"))
                    for cls in distances_par_classe]

    # Trier les données : Pur d'abord, puis Mélange, puis Autre (si existe), par ordre alphabétique dans chaque groupe
    classes_data_sorted = sorted(
        classes_data, key=lambda x: (x[2] != "Pur", x[0]))

    # Extraire la liste des classes triées et des données correspondantes
    classes_ordre = [item[0] for item in classes_data_sorted]
    data = [distances_par_classe[cls] for cls in classes_ordre]
    groupes_ordre = [groupes.get(cls, "Autre") for cls in classes_ordre]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Création du violon plot par classe
    violin_parts = ax.violinplot(
        data, showmeans=True, showmedians=True, showextrema=True)

    # Appliquer les couleurs en fonction du groupe
    # Pur (rouge), Mélange (bleu)
    for i, pc in enumerate(violin_parts['bodies']):
        if groupes_ordre[i] == "Pur":
            pc.set_facecolor('#FF9999')
        else:
            pc.set_facecolor('#66B3FF')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Personnalisation des lignes (médianes, extrêmes, etc.)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')  # Couleur noire pour les lignes principales
        vp.set_linewidth(1)

    # Personnalisation des axes et du titre
    ax.set_xticks(range(1, len(classes_ordre) + 1))
    ax.set_xticklabels(classes_ordre, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel("Distance moyenne au centrôide", fontsize=12)
    ax.set_title("Distribution des distances moyennes par polygone, par classe",
                 fontsize=14, fontweight="bold")

    # Légende
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF9999',
                      edgecolor='black', label="Pur"),  # Classes pures
        plt.Rectangle((0, 0), 1, 1, facecolor='#66B3FF',
                      edgecolor='black', label="Mélange"),  # Classes mélanges
        Line2D([0], [0], color='red', linestyle='-',
               linewidth=1, label="Médiane")  # Line rouge
    ]

    fig.subplots_adjust(right=0.8)  # Réserver un espace pour la légende
    ax.legend(handles=legend_handles, title="Essences",
              loc="upper right", bbox_to_anchor=(1.13, 1.015))

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique sauvegardé : {output_path}")
