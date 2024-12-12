import os
import logging
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from rasterstats import zonal_stats
from affine import Affine
import numpy as np
import pandas as pd

# Initialisation du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def sample_data_analysis(shapefile_path, raster_path, classes_a_conserver, output_dir):
    """
    Fonction pour analyser les échantillons :
    - 1 Créer un diagramme bâton du nombre de polygones par classe.
    - 2 Créer un diagramme bâton du nombre de pixels du raster de référence par classe.
    - 3 Créer un violon plot de la distribution du nombre de pixels par classe de polygone.

    :param shapefile_path: Chemin du fichier shapefile
    :param raster_path: Chemin du fichier raster
    :param classes_a_conserver: Liste des classes à conserver
    :param output_dir: Dossier où les graphiques seront sauvegardés
    """
    # Vérification des chemins d'entrée
    if not os.path.exists(shapefile_path):
        logger.error(f"Le fichier shapefile {shapefile_path} n'existe pas.")
        return

    if not os.path.exists(raster_path):
        logger.error(f"Le fichier raster {raster_path} n'existe pas.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Chargement du shapefile
    logger.info("Chargement du shapefile...")
    gdf = gpd.read_file(shapefile_path)

    # Filtrage des classes
    logger.info("Filtrage des classes...")
    gdf = gdf[gdf['Nom'].isin(classes_a_conserver)]

    # 1. Diagramme bâton du nombre de polygones par classe
    logger.info("Création du diagramme bâton du nombre de polygones par classe...")

    # Calcul du nombre de polygones par classe
    polygones_par_classe = gdf['Nom'].value_counts()

    # Création du diagramme
    plt.figure(figsize=(10, 6))
    ax = polygones_par_classe.plot(kind='bar', color='skyblue', edgecolor='black')

    for i, value in enumerate(polygones_par_classe):
        ax.text(
            i,
            value + 0.5,
            str(value),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    plt.title('Nombre de polygones par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de polygones', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Sauvegarde
    output_path = os.path.join(output_dir, 'diag_baton_nb_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton sauvegardé avec succès.")

    # 2. Diagramme bâton du nombre de pixels par classe
     
    logger.info("Calcul des statistiques zonales...")
    stats = zonal_stats(gdf, raster_path, stats=['sum'], geojson_out=True)

    logger.info("Création du diagramme bâton du nombre de pixels par classe...")
    pixel_counts = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        if classe in pixel_counts:
            pixel_counts[classe] += feature['properties']['sum']
        else:
            pixel_counts[classe] = feature['properties']['sum']

        # Création du diagramme bâton
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        pixel_counts.keys(),
        pixel_counts.values(),
        color='lightgreen',
        edgecolor='black'
    )

    # Ajustement des étiquettes sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(pixel_counts.values()) * 0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Ajuster les marges pour éviter que les étiquettes ne débordent
    plt.ylim(0, max(pixel_counts.values()) * 1.1)
    plt.subplots_adjust(top=0.9)

    # Personnalisation des axes et titre
    plt.title('Nombre de pixels par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de pixels', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(output_dir, 'diag_baton_nb_pix_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton sauvegardé avec succès.")

    # 3. Violin plot des pixels par classe

    logger.info("Création du violon plot de la distribution des pixels par classe...")
    distribution_par_classe = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        valeur = feature['properties']['sum']
        if classe in distribution_par_classe:
            distribution_par_classe[classe].append(valeur)
        else:
            distribution_par_classe[classe] = [valeur]

    # Configuration du style
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))

    # Création du violon plot
    parts = plt.violinplot(
        distribution_par_classe.values(),
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.8
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('#333333')
        parts['cmedians'].set_linewidth(1.5)

    # Labels et titres
    plt.xticks(
        range(1, len(distribution_par_classe) + 1),
        distribution_par_classe.keys(),
        rotation=45,
        fontsize=12
    )
    plt.title('Distribution du nombre de pixels par classe de polygone', fontsize=18)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de pixels', fontsize=14)
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # Limites de l'axe Y
    plt.ylim(bottom=0, top=15000)
    plt.tight_layout()

    # Sauvegarde
    output_path = os.path.join(output_dir, 'violin_plot_nb_pix_by_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Analyse terminée. Les graphiques ont été sauvegardés.")

def process_ndvi(
    raster_path,
    shapefile_path,
    classes_interet,
    output_plot_path=None
):
    """
    Traite un raster NDVI multibandes et un shapefile pour calculer les statistiques NDVI
    globales par classe et tracer des séries temporelles.

    Args:
        raster_path (str): Chemin vers le fichier raster NDVI multibandes.
        shapefile_path (str): Chemin vers le fichier shapefile.
        classes_interet (list): Liste des classes à conserver.
        output_plot_path (str): Chemin de sauvegarde pour le plot généré.
    """
    logger.info("Début du traitement du NDVI.")

    # Charger le shapefile et filtrer les classes d'intérêt
    logger.info("Chargement du shapefile.")
    gdf = gpd.read_file(shapefile_path)
    initial_count = len(gdf)
    gdf = gdf[gdf['Nom'].isin(classes_interet)]
    filtered_count = len(gdf)
    logger.info(f"Filtrage des classes d'intérêt: {filtered_count} sur {initial_count} entités conservées.")

    # Ouvrir le raster avec GDAL
    logger.info("Ouverture du raster NDVI")
    dataset = gdal.Open(raster_path)
    if dataset is None:
        logger.error(f"Impossible d'ouvrir le raster: {raster_path}")
        return

    nb_bands = dataset.RasterCount
    logger.info(f"Raster chargé avec {nb_bands} bandes.")

    # Récupérer la transformation affine
    geo_transform = dataset.GetGeoTransform()
    affine_transform = Affine.from_gdal(*geo_transform)
    logger.info(f"Transformation affine : {affine_transform}")

    # Lire les labels des bandes et extraire les dates
    dates = []
    for band_idx in range(1, nb_bands + 1):
        band = dataset.GetRasterBand(band_idx)
        desc = band.GetDescription()

        # Extraction directe des parties après "NDVI_"
        parts = desc.split("NDVI_")[-1].split("_") if "NDVI_" in desc else []
        month = parts[0] if len(parts) > 0 and parts[0].isdigit() else f"{band_idx:02d}"
        year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "2022"
        dates.append(np.datetime64(f"{year}-{month}"))

    logger.info(f"Dates extraites des bandes : {dates}")

    # Initialiser les dictionnaires pour stocker les statistiques globales par classe
    stats_mean = {classe: [] for classe in classes_interet}
    stats_std = {classe: [] for classe in classes_interet}

    # Itérer sur chaque bande pour calculer les statistiques zonales
    for band_idx in range(1, nb_bands + 1):
        logger.info(f"Traitement de la bande {band_idx}/{nb_bands}.")
        band = dataset.GetRasterBand(band_idx)
        ndvi = band.ReadAsArray()

        # Récupérer la valeur nodata
        nodata_value = band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = -999 
           
        # Calculer les statistiques globales par classe
        for classe in classes_interet:
            # Filtrer uniquement les polygones de la classe actuelle
            gdf_classe = gdf[gdf['Nom'] == classe]

            # Calculer les statistiques pour la classe
            zs = zonal_stats(
                vectors=gdf_classe,
                raster=ndvi,
                affine=affine_transform,
                stats=['mean', 'std'],
                geojson_out=False,
                nodata=nodata_value
            )

            # Extraire les valeurs globales (moyenne et écart type)
            values = [stat['mean'] for stat in zs if stat['mean'] is not None]
            if values:
                stats_mean[classe].append(np.mean(values))
                stats_std[classe].append(np.std(values))
            else:
                stats_mean[classe].append(np.nan)
                stats_std[classe].append(np.nan)

    logger.info("Calcul des statistiques terminé.")

    # Tracer les séries temporelles
    logger.info("Création des graphiques des séries temporelles.")
    plt.figure(figsize=(15, 10))
    num_classes = len(classes_interet)
    cols = 2
    rows = (num_classes + 1) // cols

    for idx, classe in enumerate(classes_interet, 1):
        plt.subplot(rows, cols, idx)
        mean = np.array(stats_mean[classe])
        std = np.array(stats_std[classe])

        # Utiliser les dates extraites pour l'axe des abscisses
        plt.plot(dates, mean, '-o', label="NDVI : Moyenne ± Écart type")

        # Tracer les écarts types
        plt.fill_between(dates, mean - std, mean + std, color='skyblue', alpha=0.5)

        plt.title(f"Série Temporelle NDVI - {classe}")
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.ylim(0.5, 1)
        plt.xticks(dates, [d.astype('datetime64[M]').astype(str) for d in dates], rotation=45, ha='right')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if output_plot_path:
        plt.savefig(output_plot_path)
        logger.info(f"Graphique sauvegardé sous {output_plot_path}.")
    plt.show()