import os
import logging
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg') 
from rasterstats import zonal_stats
from affine import Affine
import numpy as np
import pandas as pd

# Initialisation du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Fonction principale

def sample_data_analysis(shapefile_path, raster_path, classes_a_conserver, output_dir):
    """
    Analyse les données à partir d'un shapefile et d'un raster pour générer des graphiques :
    - Diagramme bâton du nombre de polygones par classe.
    - Diagramme bâton du nombre de pixels par classe.
    - Violon plot de la distribution des pixels par classe.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param raster_path: Chemin vers le fichier raster.
    :param classes_a_conserver: Liste des classes à conserver dans l'analyse.
    :param output_dir: Répertoire où les graphiques seront sauvegardés.
    """
    if not validate_inputs(shapefile_path, raster_path, output_dir):
        return

    gdf = load_and_filter_shapefile(shapefile_path, classes_a_conserver)

    create_bar_plot_polygons_by_class(gdf, output_dir)

    stats = calculate_zonal_stats(gdf, raster_path)
    create_bar_plot_pixels_by_class(stats, output_dir)

    create_violin_plot_pixel_distribution(stats, output_dir)

    logger.info("Analyse terminée. Les graphiques ont été sauvegardés.")

# Validation des entrées

def validate_inputs(shapefile_path, raster_path, output_dir):
    """
    Valide les chemins d'entrée et crée le répertoire de sortie si nécessaire.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param raster_path: Chemin vers le fichier raster.
    :param output_dir: Répertoire où les graphiques seront sauvegardés.
    :return: True si les entrées sont valides, False sinon.
    """
    if not os.path.exists(shapefile_path):
        logger.error(f"Le fichier shapefile {shapefile_path} n'existe pas.")
        return False

    if not os.path.exists(raster_path):
        logger.error(f"Le fichier raster {raster_path} n'existe pas.")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return True

# Chargement et filtrage du shapefile

def load_and_filter_shapefile(shapefile_path, classes_a_conserver):
    """
    Charge le shapefile et filtre les données en fonction des classes spécifiées.

    :param shapefile_path: Chemin vers le fichier shapefile.
    :param classes_a_conserver: Liste des classes à conserver.
    :return: GeoDataFrame filtré.
    """
    logger.info("Chargement et filtrage du shapefile...")
    gdf = gpd.read_file(shapefile_path)
    return gdf[gdf['Nom'].isin(classes_a_conserver)]

# Création d'un diagramme bâton pour les polygones

def create_bar_plot_polygons_by_class(gdf, output_dir):
    """
    Crée un diagramme bâton représentant le nombre de polygones par classe.

    :param gdf: GeoDataFrame contenant les données filtrées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info("Création du diagramme bâton du nombre de polygones par classe...")

    # Compter les polygones par classe et trier dans l'ordre décroissant
    polygones_par_classe = gdf['Nom'].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = polygones_par_classe.plot(kind='bar', color='skyblue', edgecolor='black')

    # Ajouter les annotations au-dessus des barres
    for i, value in enumerate(polygones_par_classe):
        ax.text(
            i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, color='black'
        )

    plt.title('Nombre de polygones par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de polygones', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Enregistrer le graphique
    output_path = os.path.join(output_dir, 'diag_baton_nb_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton des polygones sauvegardé avec succès.")


# Calcul des statistiques zonales

def calculate_zonal_stats(gdf, raster_path):
    """
    Calcule les statistiques zonales pour les polygones dans le GeoDataFrame à partir du raster.

    :param gdf: GeoDataFrame contenant les données filtrées.
    :param raster_path: Chemin vers le fichier raster.
    :return: Liste des statistiques zonales.
    """
    logger.info("Calcul des statistiques zonales...")
    return zonal_stats(gdf, raster_path, stats=['sum'], geojson_out=True)

# Création d'un diagramme bâton pour les pixels

def create_bar_plot_pixels_by_class(stats, output_dir):
    """
    Crée un diagramme bâton représentant le nombre total de pixels par classe.

    :param stats: Liste des statistiques zonales calculées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info("Création du diagramme bâton du nombre de pixels par classe...")

    pixel_counts = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        pixel_counts[classe] = pixel_counts.get(classe, 0) + feature['properties']['sum']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        pixel_counts.keys(),
        pixel_counts.values(),
        color='lightgreen',
        edgecolor='black'
    )

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

    plt.ylim(0, max(pixel_counts.values()) * 1.1)
    plt.subplots_adjust(top=0.9)
    plt.title('Nombre de pixels par classe', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de pixels', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'diag_baton_nb_pix_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Diagramme bâton des pixels sauvegardé avec succès.")

# Création d'un violon plot pour la distribution des pixels

def create_violin_plot_pixel_distribution(stats, output_dir):
    """
    Crée un violon plot représentant la distribution des pixels par classe de polygone.

    :param stats: Liste des statistiques zonales calculées.
    :param output_dir: Répertoire où le graphique sera sauvegardé.
    """
    logger.info("Création du violon plot de la distribution des pixels par classe...")

    # Regrouper les valeurs par classe
    distribution_par_classe = {}
    for feature in stats:
        classe = feature['properties']['Nom']
        valeur = feature['properties']['sum']
        distribution_par_classe.setdefault(classe, []).append(valeur)

    # Conversion en listes
    classes = list(distribution_par_classe.keys())
    distributions = list(distribution_par_classe.values())

    # Configuration du style du graphique
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    # Création du violon plot
    parts = plt.violinplot(
        distributions,
        showmeans=True,
        showmedians=True,
        showextrema=False,
        widths=0.7
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#D3D3D3')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    # Personnalisation des lignes (médianes et moyennes)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('blue')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)

    # Configurations des axes et titres
    plt.xticks(
        range(1, len(classes) + 1),
        classes,
        rotation=45,
        fontsize=12
    )
    plt.title('Distribution des Pixels par Classe de Polygone', fontsize=18, fontweight='bold')
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre de Pixels', fontsize=14)

    # Ajout des limites de l'axe Y
    plt.ylim(bottom=0, top=15000)

    # Légende
    plt.legend([
        plt.Line2D([0], [0], color='red', lw=2),
        plt.Line2D([0], [0], color='blue', lw=1.5),
        mpatches.Patch(facecolor='#D3D3D3', edgecolor='black', alpha=0.8)
    ], ['Médiane', 'Moyenne', 'Distribution'], loc='upper right', fontsize=10)

    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(output_dir, 'violin_plot_nb_pix_by_poly_by_class.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Violon plot sauvegardé avec succès : {output_path}")


def load_and_filter_shapefile(shapefile_path, classes_interet):
    """
    Charge un shapefile et filtre les entités selon les classes d'intérêt.
    """
    logger.info("Chargement du shapefile.")
    gdf = gpd.read_file(shapefile_path)
    initial_count = len(gdf)
    gdf_filtered = gdf[gdf['Nom'].isin(classes_interet)]
    filtered_count = len(gdf_filtered)
    logger.info(f"Filtrage des classes d'intérêt: {filtered_count} sur {initial_count} entités conservées.")
    return gdf_filtered


def load_raster_and_metadata(raster_path):
    """
    Charge un raster et récupère ses métadonnées.
    """
    logger.info("Ouverture du raster NDVI.")
    dataset = gdal.Open(raster_path)
    if dataset is None:
        logger.error(f"Impossible d'ouvrir le raster: {raster_path}")
        return None, None, None
    nb_bands = dataset.RasterCount
    geo_transform = dataset.GetGeoTransform()
    affine_transform = Affine.from_gdal(*geo_transform)
    logger.info(f"Raster chargé avec {nb_bands} bandes et transformation affine : {affine_transform}.")
    return dataset, nb_bands, affine_transform


def extract_dates_from_bands(dataset, nb_bands):
    """
    Extrait les dates à partir des descriptions des bandes du raster.
    """
    dates = []
    for band_idx in range(1, nb_bands + 1):
        band = dataset.GetRasterBand(band_idx)
        desc = band.GetDescription()
        parts = desc.split("NDVI_")[-1].split("_") if "NDVI_" in desc else []
        month = parts[0] if len(parts) > 0 and parts[0].isdigit() else f"{band_idx:02d}"
        year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "2022"
        dates.append(np.datetime64(f"{year}-{month}"))
    logger.info(f"Dates extraites des bandes : {dates}.")
    return dates


def compute_zonal_statistics(dataset, gdf, classes_interet, affine_transform, nb_bands):
    """
    Calcule les statistiques zonales (moyenne et écart type) pour chaque classe et bande.
    """
    stats_mean = {classe: [] for classe in classes_interet}
    stats_std = {classe: [] for classe in classes_interet}

    for band_idx in range(1, nb_bands + 1):
        logger.info(f"Traitement de la bande {band_idx}/{nb_bands}.")
        band = dataset.GetRasterBand(band_idx)
        ndvi = band.ReadAsArray()
        nodata_value = band.GetNoDataValue() or -999

        for classe in classes_interet:
            gdf_classe = gdf[gdf['Nom'] == classe]
            zs = zonal_stats(
                vectors=gdf_classe,
                raster=ndvi,
                affine=affine_transform,
                stats=['mean', 'std'],
                geojson_out=False,
                nodata=nodata_value
            )
            values = [stat['mean'] for stat in zs if stat['mean'] is not None]
            if values:
                stats_mean[classe].append(np.mean(values))
                stats_std[classe].append(np.std(values))
            else:
                stats_mean[classe].append(np.nan)
                stats_std[classe].append(np.nan)
    return stats_mean, stats_std


def plot_ndvi_time_series(dates, stats_mean, stats_std, classes_interet, output_plot_path=None):
    """
    Trace les séries temporelles NDVI pour chaque classe.
    """
    logger.info("Création des graphiques des séries temporelles.")
    plt.figure(figsize=(15, 10))
    num_classes = len(classes_interet)
    cols = 2
    rows = (num_classes + 1) // cols

    for idx, classe in enumerate(classes_interet, 1):
        plt.subplot(rows, cols, idx)
        mean = np.array(stats_mean[classe])
        std = np.array(stats_std[classe])

        plt.plot(dates, mean, '-o', label="NDVI : Moyenne ± Écart type")
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


def process_ndvi(raster_path, shapefile_path, classes_interet, output_plot_path=None):
    """
    Traite un raster NDVI multibandes et un shapefile pour calculer les statistiques NDVI
    globales par classe et tracer des séries temporelles.
    """
    logger.info("Début du traitement du NDVI.")
    gdf = load_and_filter_shapefile(shapefile_path, classes_interet)
    dataset, nb_bands, affine_transform = load_raster_and_metadata(raster_path)
    if dataset is None:
        return
    dates = extract_dates_from_bands(dataset, nb_bands)
    stats_mean, stats_std = compute_zonal_statistics(dataset, gdf, classes_interet, affine_transform, nb_bands)
    plot_ndvi_time_series(dates, stats_mean, stats_std, classes_interet, output_plot_path)
