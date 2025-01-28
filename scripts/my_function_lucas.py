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

from osgeo import gdal
from rasterstats import zonal_stats
from osgeo import gdal, ogr
import time
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


# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #
# ================================= ## ================================= ## ================================= #


def cleanup_temp_files(*base_file_paths):
    """
    Supprime les fichiers temporaires spécifiés avec toutes leurs extensions associées.

    Cette fonction supprime les fichiers associés à des shapefiles (ou d'autres formats similaires)
    en supprimant toutes les extensions associées à un chemin de base donné.

    Paramètres :
    -----------
    *base_file_paths : str
        Chemins de base des fichiers à supprimer (sans extension spécifique).
        Ex : "temp_classes_dissolues" supprimera tous les fichiers "temp_classes_dissolues.*".

    Retours :
    ---------
    Aucun. Les fichiers temporaires sont supprimés, et les erreurs éventuelles sont enregistrées.
    """
    extensions = [
        ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".fix", ".shp.xml"
    ]  # Extensions associées aux shapefiles et fichiers temporaires

    for base_path in base_file_paths:
        if not base_path or not isinstance(base_path, str):
            logging.warning(f"Chemin de base invalide : {base_path}")
            continue

        for ext in extensions:
            file_path = f"{base_path}{ext}"
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Fichier temporaire supprimé : {file_path}")
                else:
                    logging.debug(
                        f"Fichier non trouvé (aucune action nécessaire) : {file_path}")
            except Exception as e:
                logging.error(
                    f"Erreur lors de la suppression du fichier {file_path} : {e}"
                )


def report_from_dict_to_df(dict_report):
    """
    Convertit un rapport de classification en un DataFrame Pandas.

    Cette fonction prend un dictionnaire généré par un rapport
    de classification (issu de `classification_report` de scikit-learn),
    le transforme en DataFrame et élimine les colonnes et lignes non pertinentes.

    Paramètres :
    ------------
    dict_report : dict
        Le dictionnaire contenant les métriques du rapport de classification.

    Retours :
    ---------
    report_df : pandas.DataFrame
        Un DataFrame contenant uniquement les métriques par classe (précision, rappel, F1).

    Exceptions :
    ------------
    Retourne un DataFrame vide si l'entrée est invalide.
    """
    try:
        # Conversion du dictionnaire en DataFrame
        report_df = pd.DataFrame.from_dict(dict_report)

        # Vérifier et supprimer les colonnes non pertinentes si elles existent
        cols_to_drop = ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
        for col in cols_to_drop:
            if col in report_df.columns:
                report_df.drop(columns=col, inplace=True)

        # Vérifier et supprimer la ligne "support" si elle existe
        if 'support' in report_df.index:
            report_df.drop(index='support', inplace=True)

        # Retourner le DataFrame nettoyé
        return report_df

    except Exception as e:
        logging.warning(f"Erreur lors de la conversion du rapport : {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur


def create_raster_sampleimage(sample_vector, reference_raster, output_path, attribute):
    """
    Crée un raster d'échantillonnage à partir d'un vecteur et d'un raster de référence.

    Cette fonction génère un raster en utilisant un vecteur d'échantillons 
    et un raster de référence, en rasterisant selon un attribut donné.

    Paramètres :
    ------------
    sample_vector : str
        Chemin vers le fichier vecteur contenant les échantillons.
    reference_raster : str
        Chemin vers le raster de référence définissant la géométrie et la projection.
    output_path : str
        Chemin pour sauvegarder le raster de sortie.
    attribute : str
        Nom de l'attribut dans le vecteur à utiliser pour la rasterisation.

    Retours :
    ---------
    Aucun. Un fichier raster est généré à l'emplacement `output_path`.

    Exceptions :
    ------------
    Lève une exception si le fichier vecteur ou raster de référence ne peut pas être ouvert,
    ou si la rasterisation échoue.
    """
    try:
        # Vérifier l'existence des fichiers d'entrée
        if not os.path.exists(sample_vector):
            raise FileNotFoundError(
                f"Fichier vecteur introuvable : {sample_vector}")
        if not os.path.exists(reference_raster):
            raise FileNotFoundError(
                f"Raster de référence introuvable : {reference_raster}")

        # Ouvrir le raster de référence
        ref_ds = gdal.Open(reference_raster, gdal.GA_ReadOnly)
        if ref_ds is None:
            raise RuntimeError(
                f"Impossible d'ouvrir le raster de référence : {reference_raster}")

        # Extraire les propriétés du raster de référence
        geo_transform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()
        x_pixels = ref_ds.RasterXSize
        y_pixels = ref_ds.RasterYSize

        # Déterminer le type de données pour le raster de sortie
        raster_dtype = gdal.GDT_UInt16 if attribute.lower() == "id" else gdal.GDT_Byte

        # Créer un raster en mémoire
        out_raster = gdal.GetDriverByName("MEM").Create(
            "", x_pixels, y_pixels, 1, raster_dtype
        )
        out_raster.SetGeoTransform(geo_transform)
        out_raster.SetProjection(projection)
        out_band = out_raster.GetRasterBand(1)
        out_band.SetNoDataValue(0)  # NoData = 0
        out_band.Fill(0)  # Remplir avec la valeur NoData

        # Ouvrir le vecteur
        vector_ds = gdal.OpenEx(sample_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            raise RuntimeError(
                f"Impossible d'ouvrir le fichier vecteur : {sample_vector}")
        vector_layer = vector_ds.GetLayer()

        # Rasteriser le vecteur
        options = [f"ATTRIBUTE={attribute}"]
        gdal.RasterizeLayer(out_raster, [1], vector_layer, options=options)

        # Sauvegarder le raster sur disque
        driver = gdal.GetDriverByName("GTiff")
        driver.CreateCopy(output_path, out_raster, options=["COMPRESS=LZW"])

        logging.info(
            f"Raster d'échantillonnage créé avec succès : {output_path}")

    except Exception as e:
        logging.error(
            f"Erreur lors de la création du raster d'échantillonnage : {e}")
        raise


def compute_zonal_statistics(gdf, raster_path, all_touched=True, nodata=0):
    """
    Calcule les statistiques zonales (catégorielles) pour chaque polygone d'un
    GeoDataFrame, en utilisant la bibliothèque rasterstats.

    Paramètres
    ----------
    gdf : geopandas.GeoDataFrame
        Le GeoDataFrame représentant les polygones (ex. BD forêt).
    raster_path : str
        Chemin vers le fichier raster (carte des essences) sur lequel faire
        les statistiques zonales.
    all_touched : bool, optionnel
        Indique si on inclut les pixels même s'ils ne sont que partiellement
        recouverts par le polygone. Par défaut True.
    nodata : int, optionnel
        Valeur NoData dans le raster, par défaut 0.

    Retourne
    -------
    stats_zonales : list of dict
        Liste de dictionnaires, chaque dictionnaire contient les comptes
        de chaque classe pour le polygone correspondant.
    """
    stats_zonales = zonal_stats(
        gdf,
        raster_path,
        all_touched=all_touched,
        categorical=True,
        nodata=nodata
    )
    return stats_zonales


def get_pixel_area(raster_path):
    """
    Renvoie la surface d'un pixel (en m²) à partir de la géotransformation
    du raster.

    Paramètres
    ----------
    raster_path : str
        Chemin vers le fichier raster.

    Retourne
    -------
    float
        Surface d'un pixel en m².
    """
    raster = gdal.Open(raster_path)
    if raster is None:
        raise FileNotFoundError(
            f"Impossible d'ouvrir le raster : {raster_path}")

    geo_transform = raster.GetGeoTransform()
    # Taille du pixel
    pixel_width = abs(geo_transform[1])
    pixel_height = abs(geo_transform[5])
    pixel_surface = pixel_width * pixel_height
    raster = None  # fermeture
    return pixel_surface


def classify_peuplement_feuillus_coniferes(pourcentage_classes,
                                           pourcentage_feuillus,
                                           pourcentage_coniferes,
                                           seuil,
                                           surf,
                                           surf_mini,
                                           feuillus_ilots,
                                           coniferes_ilots,
                                           melange_feuillus,
                                           melange_coniferes,
                                           melange_conif_prep_feuil,
                                           melange_feuil_prep_conif):
    """
    Classifie un polygone en fonction des pourcentages de classes spécifiques,
    feuillus/conifères et de sa superficie, selon les règles de décision.
    """
    # Vérifier si une classe unique domine (> 75%)
    for classe, pourcentage in pourcentage_classes.items():
        if pourcentage > seuil:
            return classe  # Polygone classé directement dans la classe dominante

    # Classification en fonction de la superficie et des pourcentages globaux
    if surf < surf_mini:
        # Petits îlots
        if pourcentage_feuillus > seuil:
            return feuillus_ilots
        elif pourcentage_coniferes > seuil:
            return coniferes_ilots
        elif pourcentage_coniferes > pourcentage_feuillus:
            return melange_conif_prep_feuil
        else:
            return melange_feuil_prep_conif
    else:
        # Grands polygones
        if pourcentage_feuillus > seuil:
            return melange_feuillus
        elif pourcentage_coniferes > seuil:
            return melange_coniferes
        elif pourcentage_coniferes > pourcentage_feuillus:
            return melange_conif_prep_feuil
        else:
            return melange_feuil_prep_conif


def compute_peuplement_class(stats_zonales,
                             pixel_surface,
                             seuil_categorie,
                             surface_mini,
                             codes_peuplement):
    """
    À partir des statistiques zonales et du code de classification,
    calcule la classe de peuplement pour chaque polygone.

    Paramètres
    ----------
    stats_zonales : list of dict
        Sortie de la fonction compute_zonal_statistics().
    pixel_surface : float
        Surface d'un pixel (m²).
    seuil_categorie : float
        Seuil pour distinguer majoritairement feuillus ou conifères (en %).
    surface_mini : float
        Seuil de superficie pour différencier les petits îlots des grands polygones.
    codes_peuplement : dict
        Dictionnaire contenant les codes de classes (ex. "feuillus_ilots": 16, etc.).

    Retourne
    -------
    classes_predites : list of int
        Liste des classes prédites (une par polygone).
    percentages_feuillus : list of float
        Liste des pourcentages feuillus.
    percentages_coniferes : list of float
        Liste des pourcentages conifères.
    surfaces : list of float
        Liste des surfaces des polygones (m²).
    """
    classes_predites = []
    percentages_feuillus = []
    percentages_coniferes = []
    surfaces = []

    # Récupération des différents codes
    feuillus_ilots = codes_peuplement["feuillus_ilots"]
    melange_feuillus = codes_peuplement["melange_feuillus"]
    coniferes_ilots = codes_peuplement["coniferes_ilots"]
    melange_coniferes = codes_peuplement["melange_coniferes"]
    melange_conif_prep_feuil = codes_peuplement["melange_conif_prep_feuil"]
    melange_feuil_prep_conif = codes_peuplement["melange_feuil_prep_conif"]

    # Parcours de chaque polygone
    for stats_poly in stats_zonales:
        total_pixels = sum(stats_poly.values())

        # Vérifier si le total est zéro (éviter division par zéro)
        if total_pixels == 0:
            logging.warning(
                "Total de pixels égal à zéro pour un polygone. Classe inconnue attribuée.")
            classes_predites.append(0)  # Code pour "classe inconnue"
            percentages_feuillus.append(0)
            percentages_coniferes.append(0)
            surfaces.append(0)
            continue

        # Calcul des pourcentages pour les feuillus et conifères
        pourcentage_feuillus = sum(
            (count / total_pixels) * 100
            for cat, count in stats_poly.items()
            if 10 < int(cat) < 15
        )
        pourcentage_coniferes = sum(
            (count / total_pixels) * 100
            for cat, count in stats_poly.items()
            if 20 < int(cat) < 26
        )

        pourcentage_classes = {
            cat: (count / total_pixels) * 100
            for cat, count in stats_poly.items()
        }

        # Vérifier si les catégories feuillus et conifères sont absentes
        if pourcentage_feuillus == 0 and pourcentage_coniferes == 0:
            logging.warning(
                "Aucune catégorie feuillus ou conifères détectée. Classe inconnue attribuée.")
            classes_predites.append(0)  # Code pour "classe inconnue"
            percentages_feuillus.append(0)
            percentages_coniferes.append(0)
            surfaces.append(total_pixels * pixel_surface)
            continue

        # Calculer la surface du polygone
        surface_poly = total_pixels * pixel_surface

        # Appliquer les règles de décision
        classe_predite = classify_peuplement_feuillus_coniferes(
            pourcentage_classes,
            pourcentage_feuillus,
            pourcentage_coniferes,
            seuil_categorie,
            surface_poly,
            surface_mini,
            feuillus_ilots,
            coniferes_ilots,
            melange_feuillus,
            melange_coniferes,
            melange_conif_prep_feuil,
            melange_feuil_prep_conif
        )

        # Stocker les résultats
        classes_predites.append(classe_predite)
        percentages_feuillus.append(pourcentage_feuillus)
        percentages_coniferes.append(pourcentage_coniferes)
        surfaces.append(surface_poly)

    return classes_predites, percentages_feuillus, percentages_coniferes, surfaces
