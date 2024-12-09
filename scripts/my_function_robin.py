#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from osgeo import gdal, ogr, osr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt



# =========================== #
# === DATA PLOTTING === #
# =========================== #

# Initialisation du logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def sample_data_analysis(shapefile_path, raster_path, classes_a_conserver, output_dir):
    """
    Fonction pour analyser les échantillons : 
    - 1 Créer un diagramme bâton du nombre de polygones par classe.
    - 2 Créer un diagramme bâton du nombre de pixels du raster de référence par classe.
    - 3 Créer un violon plot de la distribution du nombre de pixels par classe de polygone, amélioré visuellement.

    :param shapefile_path: Chemin du fichier shapefile
    :param raster_path: Chemin du fichier raster
    :param classes_a_conserver: Liste des classes à conserver
    :param output_dir: Dossier où les graphiques seront sauvegardés
    """
    # Vérifier et créer le répertoire de sortie si nécessaire
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Création du répertoire de sortie : {output_dir}")

    # Charger le fichier shapefile
    logger.info("Chargement du fichier shapefile...")
    gdf = gpd.read_file(shapefile_path)

    # Charger le raster
    logger.info("Chargement du fichier raster...")
    dataset = gdal.Open(raster_path)
    if dataset is None:
        logger.error(f"Impossible de charger le fichier raster : {raster_path}")
        return
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Spécifiez la colonne contenant les classes
    classe_colonne = "Nom"

    # Filtrer les classes spécifiées
    logger.info(f"Filtrage des classes spécifiées dans {classes_a_conserver}...")
    gdf_filtre = gdf[gdf[classe_colonne].isin(classes_a_conserver)].copy()

    if gdf_filtre.empty:
        logger.warning(f"Aucune des classes spécifiées dans {classes_a_conserver} n'a été trouvée dans le fichier.")
        return

    # 1. Diagramme bâton du nombre de polygones par classe
    logger.info("Création du diagramme en bâtons pour le nombre de polygones...")
    class_counts = gdf_filtre[classe_colonne].value_counts()
    class_counts_sorted = class_counts.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts_sorted.index, class_counts_sorted.values, color='skyblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.title("Nombre de polygones par classe", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de polygones", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "diag_baton_nb_poly_by_class.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Diagramme bâton des polygones sauvegardé : {output_path}")

    # 2. Diagramme bâton du nombre de pixels du raster par classe
    logger.info("Rasterisation des polygones pour compter les pixels par classe...")
    class_to_id = {classe: idx + 1 for idx, classe in enumerate(classes_a_conserver)}
    mem_driver = gdal.GetDriverByName('MEM')
    rasterized_ds = mem_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    rasterized_ds.SetGeoTransform(geo_transform)
    rasterized_ds.SetProjection(projection)
    rasterized_ds.GetRasterBand(1).SetNoDataValue(0)
    rasterized_ds.GetRasterBand(1).Fill(0)

    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(projection)

    for classe, id_val in class_to_id.items():
        logger.info(f"Rasterisation de la classe '{classe}' avec l'ID {id_val}...")
        gdf_classe = gdf_filtre[gdf_filtre[classe_colonne] == classe]
        if gdf_classe.empty:
            logger.warning(f"Aucune géométrie trouvée pour la classe '{classe}'.")
            continue
        gdf_classe = gdf_classe.to_crs(source_srs.ExportToWkt())
        mem_ogr_driver = ogr.GetDriverByName('Memory')
        mem_ogr_ds = mem_ogr_driver.CreateDataSource('memData')
        mem_ogr_layer = mem_ogr_ds.CreateLayer('memLayer', srs=source_srs, geom_type=ogr.wkbPolygon)
        for geom in gdf_classe.geometry:
            if geom is None or geom.is_empty:
                continue
            feature = ogr.Feature(mem_ogr_layer.GetLayerDefn())
            ogr_geom = ogr.CreateGeometryFromWkb(geom.wkb)
            feature.SetGeometry(ogr_geom)
            mem_ogr_layer.CreateFeature(feature)
            feature = None

        gdal.RasterizeLayer(rasterized_ds, [1], mem_ogr_layer, burn_values=[id_val])
        mem_ogr_ds = None

    rasterized_array = rasterized_ds.GetRasterBand(1).ReadAsArray()
    rasterized_ds = None
    pixels_per_class = {classe: np.sum(rasterized_array == id_val) for classe, id_val in class_to_id.items()}
    pixels_per_class_sorted = dict(sorted(pixels_per_class.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 6))
    classes = list(pixels_per_class_sorted.keys())
    counts = list(pixels_per_class_sorted.values())

    # Ajustement de la limite supérieure de l'axe Y
    y_max = max(counts) * 1.1
    plt.ylim(0, y_max)

    bars = plt.bar(classes, counts, color='lightgreen', edgecolor='black')

    # Ajout des étiquettes avec un décalage ajusté
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(counts) * 0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title("Nombre de pixels du raster par classe", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de pixels", fontsize=12)
    plt.xticks(rotation=45)

    # Ajustement des marges
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path_pixels = os.path.join(output_dir, "diag_baton_nb_pixels_by_class.png")
    plt.savefig(output_path_pixels, dpi=300)
    plt.close()
    logger.info(f"Diagramme bâton des pixels sauvegardé : {output_path_pixels}")


    # 3. Création du violon plot
    logger.info("Création du violon plot...")
    gdf_filtre['area'] = gdf_filtre.geometry.area
    pixel_width = geo_transform[1]
    pixel_height = -geo_transform[5]
    pixel_area = pixel_width * pixel_height
    gdf_filtre['pixel_count'] = gdf_filtre['area'] / pixel_area

    df_pixels_per_polygon = gdf_filtre[[classe_colonne, 'pixel_count']].copy()
    df_pixels_per_polygon['pixel_count_clipped'] = df_pixels_per_polygon['pixel_count'].clip(upper=15000)
    max_pixels_per_class = df_pixels_per_polygon.groupby(classe_colonne)['pixel_count_clipped'].max()
    sorted_classes = max_pixels_per_class.sort_values(ascending=False).index.tolist()

    violin_data = [
        df_pixels_per_polygon[df_pixels_per_polygon[classe_colonne] == classe]['pixel_count_clipped'].dropna().values
        for classe in sorted_classes
    ]

    plt.figure(figsize=(12, 8))

    # Création du violon plot
    violin_parts = plt.violinplot(violin_data, showmeans=False, showextrema=False, showmedians=True)

    # Choix des couleurs sur un dégradé
    cmap = plt.cm.get_cmap('viridis')
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(cmap((i+1)/len(violin_data)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Ajout de la légende pour la médiane
    plt.legend([plt.Line2D([0], [0], color='black', linewidth=1)], ["Médiane"], loc='upper right')

    # Configuration du graphique
    plt.title("Distribution du nombre de pixels par classe de polygone", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de pixels par polygone", fontsize=12)
    plt.xticks(ticks=range(1, len(sorted_classes) + 1), labels=sorted_classes, rotation=45)
    plt.ylim(0, 15000)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Sauvegarde du plot
    plt.tight_layout()
    output_path_violin = os.path.join(output_dir, "violin_plot_nb_pix_by_poly_by_class.png")
    plt.savefig(output_path_violin, dpi=300)
    plt.close()
    logger.info(f"Violin plot sauvegardé : {output_path_violin}")



def analyse_ndvi_par_classe(raster_path, shapefile_path, classes_interet, nom_champ=None, 
                            output_plot=None):
    
    """
    Analyse le NDVI par classe à partir d'un raster multibande et d'un shapefile.

    Cette fonction :
    1. Lit un raster multibande (chaque bande représentant une date de NDVI).
    2. Lit un shapefile, filtre ses entités par le champ `nom_champ` selon la liste `classes_interet`.
    3. Rasterise les géométries filtrées pour créer un masque binaire.
    4. Calcule, pour chaque classe, les valeurs moyennes et écarts-types de NDVI sur l'ensemble des pixels sélectionnés, 
       et ce pour chaque bande.
    5. Génère des subplots organisés en grille, chaque subplot représentant l'évolution 
       du NDVI pour une classe donnée. Les étiquettes temporelles sont extraites depuis les descriptions des bandes 
       du raster. L'échelle verticale (axe Y) est identique sur tous les subplots pour permettre une comparaison visuelle.
    6. Enregistre la figure sous forme de fichier PNG.

    Paramètres
    ----------
    raster_path : str
        Chemin vers le fichier raster multibande (ex : "data/raster_ndvi.tif").
    shapefile_path : str
        Chemin vers le shapefile contenant les polygones (ex : "data/parcelles.shp").
    classes_interet : list of str
        Liste des classes (valeurs de `nom_champ`) à filtrer dans le shapefile (ex : ["Forêt", "Prairie"]).
    nom_champ : str, optionnel
        Nom du champ attributaire dans le shapefile contenant la classification des entités. Par défaut "Nom".
    output_plot : str, optionnel
        Chemin du fichier PNG de sortie pour le graphique. Par défaut "groupe_9\\results\\figure\\temp_mean_ndvi.png".

    
    """

    logger.info("Démarrage de l'analyse NDVI par classe.")

    # Création du répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    logger.info(f"Le graphique sera enregistré dans {output_plot}")

    # Ouverture du raster avec GDAL
    logger.info(f"Ouverture du raster {raster_path}")
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f"Impossible d'ouvrir le raster {raster_path}.")
        raise IOError(f"Impossible d'ouvrir le raster {raster_path}.")
    
    # Lecture des dimensions et métadonnées
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    nbands = ds.RasterCount
    logger.info(f"Raster ouvert avec {nbands} bandes, {ncols} colonnes et {nrows} lignes.")

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # Lecture automatique des étiquettes de bandes (dates)
    band_labels = []
    for b in range(1, nbands+1):
        band = ds.GetRasterBand(b)
        desc = band.GetDescription()
        if desc:
            parts = desc.split('_')
            if len(parts) >= 3:
                month = parts[-2]
                year = parts[-1]
                if len(year) == 4 and year.isdigit() and month.isdigit():
                    band_label = f"{month}/{year}"
                else:
                    band_label = desc
            else:
                band_label = desc
        else:
            band_label = f"Bande_{b}"
        band_labels.append(band_label)

    # Lecture du shapefile
    logger.info(f"Lecture du shapefile {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        logger.error(f"Le shapefile {shapefile_path} est vide ou n'a pas pu être lu.")
        raise ValueError(f"Le shapefile {shapefile_path} est vide ou n'a pas pu être lu.")
    
    # Filtrage des classes d'intérêt
    logger.info(f"Filtrage sur les classes d'intérêt : {classes_interet}")
    gdf_filtre = gdf[gdf[nom_champ].isin(classes_interet)]
    classes_uniques = gdf_filtre[nom_champ].unique()

    # Dictionnaire pour stocker les stats
    stats_par_classe = {}
    
    # Création d'une datasource mémoire
    driver = ogr.GetDriverByName('Memory')
    data_source = driver.CreateDataSource('temp')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = data_source.CreateLayer('layer', srs, geom_type=ogr.wkbPolygon)
    
    field_defn = ogr.FieldDefn(nom_champ, ogr.OFTString)
    layer.CreateField(field_defn)
    
    # Ajout des entités filtrées dans la couche OGR
    logger.info("Ajout des entités filtrées.")
    for idx, row in gdf_filtre.iterrows():
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField(nom_champ, row[nom_champ])
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None

    # Lecture du raster en numpy array pour toutes les bandes
    logger.info("Lecture des valeurs du raster dans des tableaux numpy.")
    raster_arrays = []
    for b in range(1, nbands+1):
        band = ds.GetRasterBand(b)
        arr = band.ReadAsArray().astype(float)
        no_data = band.GetNoDataValue()
        if no_data is not None:
            arr[arr == no_data] = np.nan
        raster_arrays.append(arr)
    raster_arrays = np.array(raster_arrays)  

    # Calcul des stats par classe
    logger.info("Calcul des statistiques par classe.")
    for classe in classes_uniques:
        logger.info(f"Traitement de la classe : {classe}")
        layer.SetAttributeFilter(f"{nom_champ} = '{classe}'")

        # Création du raster masque en mémoire
        driver_mem = gdal.GetDriverByName('MEM')
        mask_ds = driver_mem.Create('', ncols, nrows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(projection)
        
        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        
        mask_array = mask_ds.ReadAsArray().astype(bool)
        
        masked_values = raster_arrays[:, mask_array]
        
        mean_values = np.nanmean(masked_values, axis=1)
        std_values = np.nanstd(masked_values, axis=1)
        
        stats_par_classe[classe] = {"mean": mean_values, "std": std_values}
        logger.info(f"Statistiques pour {classe} : mean = {mean_values}, std = {std_values}")
    
    ds = None
    data_source = None

    # Détermination du min et max global pour garder la même échelle en y
    global_min = float('inf')
    global_max = float('-inf')

    for classe, stats in stats_par_classe.items():
        # On considère mean ± std pour définir les bornes
        values_min = np.min(stats["mean"] - stats["std"])
        values_max = np.max(stats["mean"] + stats["std"])
        if values_min < global_min:
            global_min = values_min
        if values_max > global_max:
            global_max = values_max

    # Tracé et enregistrement du graphique
    logger.info("Création du graphique avec subplots.")
    num_classes = len(classes_uniques)
    ncols_sub = 2 if num_classes > 1 else 1
    nrows_sub = (num_classes + ncols_sub - 1) // ncols_sub

    fig, axs = plt.subplots(nrows=nrows_sub, ncols=ncols_sub, figsize=(14, 5*nrows_sub))

    if num_classes == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    x = np.arange(nbands)
    for i, classe in enumerate(classes_uniques):
        stats = stats_par_classe[classe]
        ax = axs[i]
        ax.errorbar(x, stats["mean"], yerr=stats["std"], label=classe, capsize=5, marker='o')
        ax.set_title(f"Évolution NDVI - {classe}")
        ax.set_xlabel("Date")
        ax.set_ylabel("NDVI moyen ± écart-type")
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=45, ha='right')
        ax.grid(True)
        ax.legend()
        # Application de l'échelle globale
        ax.set_ylim(global_min, global_max)

    # Masquer les axes vides s'il y en a
    if len(axs) > num_classes:
        for j in range(num_classes, len(axs)):
            axs[j].axis('off')

    fig.tight_layout()
    logger.info(f"Enregistrement du graphique sous {output_plot}")
    plt.savefig(output_plot, dpi=300)
    plt.close()
    logger.info("Analyse terminée.")

    return stats_par_classe