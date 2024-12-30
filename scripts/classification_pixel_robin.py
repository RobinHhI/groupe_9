import os
from osgeo import gdal
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
import matplotlib.pyplot as plt

import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,             
    format='[%(asctime)s] %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import sys
sys.path.append('libsigma')

import classification as cla
import read_and_write as rw
import plots
from read_and_write import (
    open_image, 
    get_image_dimension, 
    get_origin_coordinates, 
    get_pixel_size
)


# ------------------------------------------------------------------------------
# Création de my_sample_strata et my_sample_strata_id
# ------------------------------------------------------------------------------

in_vector = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
ref_image = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"

out_image_strata    = "data/images/my_sample_strata.tif"
out_image_strata_id = "data/images/my_sample_strata_id.tif"

field_name = 'Code'  # champ contenant le label numérique

# Chemin temporaire pour le shapefile avec le champ 'my_id'
temp_vector = "data/project/sample_with_numeric_id.shp"

data_set = open_image(ref_image, verbose=True)

# Récupérer les dimensions (lignes, colonnes, nb de bandes)
nb_lignes, nb_col, nb_band = get_image_dimension(data_set, verbose=True)

# Récupérer l'origine 
origin_x, origin_y = get_origin_coordinates(data_set, verbose=True)

# Récupérer la taille de pixel
psize_x, psize_y = get_pixel_size(data_set, verbose=True)

# Calculer la bounding box 
xmin = origin_x
ymax = origin_y
xmax = origin_x + psize_x * nb_col     
ymin = origin_y + psize_y * nb_lignes 

res = abs(psize_x)

cmd_pattern = (
    "gdal_rasterize -a {field_name} "
    "-tr {res} {res} "
    "-te {xmin} {ymin} {xmax} {ymax} "
    "-ot Byte -of GTiff "
    "{in_vector} {out_image}"
)

cmd = cmd_pattern.format(
    field_name=field_name,
    res=res,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    in_vector=in_vector,
    out_image=out_image_strata
)

print("Exécution de la commande (1er raster) :", cmd)
os.system(cmd)

gdf = gpd.read_file(in_vector)

# Créer un champ numérique 'my_id' (valeur unique pour chaque entité)
gdf['my_id'] = range(1, len(gdf) + 1)

# (4c) Sauvegarder sous un nouveau shapefile 
gdf.to_file(temp_vector, driver='ESRI Shapefile')

cmd_pattern_id = (
    "gdal_rasterize -a {field_name} "
    "-tr {res} {res} "
    "-te {xmin} {ymin} {xmax} {ymax} "
    "-ot UInt16 -of GTiff "
    "{in_vector} {out_image}"
)

cmd_id = cmd_pattern_id.format(
    field_name="my_id",
    res=res,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    in_vector=temp_vector,
    out_image=out_image_strata_id
)

print("Exécution de la commande (2e raster) :", cmd_id)
os.system(cmd_id)


# # ------------------------------------------------------------------------------
# # Classification pixel
# # ------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------
# # 1) Paramètres et chemins
# # ------------------------------------------------------------------------------
# logger.info("Initialisation des paramètres et des chemins...")

# # Chemins vers les rasters d'entrée
# sample_filename = "data/images/my_sample_strata.tif"  # raster des classes
# id_filename     = "data/images/my_sample_strata_id.tif"  # raster id polygon
# image_filename  = "groupe_9/results/data/img_pretraitees/Serie_temp_S2_allbands.tif"

# # Paramètres de l'échantillonnage / Cross-validation
# nb_iter  = 30     # nombre de répétitions
# nb_folds = 5      # nombre de folds pour la validation croisée
# is_point = False  # si False, on utilise cla.get_samples_from_roi

# # Hyperparamètres du classifieur RandomForest
# rf_params = {
#     'n_estimators': 100,     # valeur par défaut sklearn
#     'max_depth': 50,
#     'oob_score': True,
#     'max_samples': 0.75,
#     'class_weight': 'balanced',
# }

# # Sorties
# out_folder   = "groupe_9/results/data/classif"
# os.makedirs(out_folder, exist_ok=True)

# # Nom de la carte finale (classif à l'échelle pixel)
# out_classif  = os.path.join(out_folder, "carte_essences_echelle_pixel.tif")
# out_matrix   = os.path.join(out_folder, "ma_matrice_confusion_moyenne.png")
# out_qualite  = os.path.join(out_folder, "mes_qualites_moyennees.png")

# # ------------------------------------------------------------------------------
# # 2) Extraction des échantillons X, Y et des groupes pour la validation
# # ------------------------------------------------------------------------------
# logger.info("Extraction des échantillons (X, Y) et des groupes (pour StratifiedGroupKFold)...")

# if not is_point:
#     X, Y, _ = cla.get_samples_from_roi(image_filename, sample_filename)
#     _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)
# else:
#     list_row, list_col = rw.get_row_col_from_file(sample_filename, image_filename)
#     image = rw.load_img_as_array(image_filename)
#     X = image[(list_row, list_col)]
#     gdf = gpd.read_file(sample_filename)
#     Y = gdf.loc[:, 'Code'].values  
#     Y = np.atleast_2d(Y).T
#     groups = np.arange(len(Y))  # s’il n’y a pas d’information de polygone

# groups = np.squeeze(groups)
# Y = np.squeeze(Y)

# logger.info(f"Dimensions X : {X.shape}")
# logger.info(f"Dimensions Y : {Y.shape}")
# logger.info(f"Dimensions groups : {groups.shape}")

# # ------------------------------------------------------------------------------
# # 3) Validation Croisée (Stratified Group K-Fold) répétée nb_iter fois
# # ------------------------------------------------------------------------------
# logger.info("Démarrage de la validation croisée (StratifiedGroupKFold) sur "
#             f"{nb_folds} folds, répétée {nb_iter} fois.")

# list_cm       = []
# list_accuracy = []
# list_report   = []

# for i in range(nb_iter):
#     logger.info(f"  -> Début de l'itération {i+1}/{nb_iter}")
    
#     kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True, random_state=None)
#     fold_num = 0
    
#     for train_idx, test_idx in kf.split(X, Y, groups=groups):
#         fold_num += 1
#         logger.info(f"     ... Fold {fold_num}/{nb_folds}")
        
#         X_train, X_test = X[train_idx], X[test_idx]
#         Y_train, Y_test = Y[train_idx], Y[test_idx]

#         # Initialisation du classifieur RandomForest
#         clf = RandomForestClassifier(**rf_params)
#         logger.debug("        Entraînement du RandomForestClassifier...")
#         clf.fit(X_train, Y_train.ravel())

#         # Prédiction
#         logger.debug("        Prédiction sur le set de test...")
#         Y_pred = clf.predict(X_test)

#         # Calcul des métriques
#         cm = confusion_matrix(Y_test, Y_pred)
#         acc = accuracy_score(Y_test, Y_pred)
#         rep = classification_report(Y_test, Y_pred, 
#                                     labels=np.unique(Y_pred),
#                                     output_dict=True)

#         list_cm.append(cm)
#         list_accuracy.append(acc)
        
#         df_rep = pd.DataFrame(rep).T
#         list_report.append(df_rep)
    
#     logger.info(f"  -> Fin de l'itération {i+1}/{nb_iter}")

# # ------------------------------------------------------------------------------
# # 4) Moyenne des métriques
# # ------------------------------------------------------------------------------
# logger.info("Calcul des statistiques moyennes sur l'ensemble des itérations/folds...")

# array_cm = np.array(list_cm)
# mean_cm  = array_cm.mean(axis=0)

# array_accuracy = np.array(list_accuracy)
# mean_accuracy  = array_accuracy.mean()
# std_accuracy   = array_accuracy.std()

# array_report = np.array([df.values for df in list_report])
# mean_report  = array_report.mean(axis=0)
# std_report   = array_report.std(axis=0)

# rep_index   = list_report[0].index
# rep_columns = list_report[0].columns
# mean_df_report = pd.DataFrame(mean_report, index=rep_index, columns=rep_columns)
# std_df_report  = pd.DataFrame(std_report,  index=rep_index, columns=rep_columns)

# logger.info("\n--- RÉSULTATS VALIDATION CROISÉE ---")
# logger.info("Accuracy globale moyenne (OA) : {:.2f} +/- {:.2f}"
#             .format(mean_accuracy, std_accuracy))
# logger.info("\nMatrice de confusion moyenne :\n{}".format(mean_cm))
# logger.info("\nRapport de classification moyen :\n{}".format(mean_df_report))

# # ------------------------------------------------------------------------------
# # 5) Affichage et sauvegarde des métriques
# # ------------------------------------------------------------------------------
# logger.info("Sauvegarde de la matrice de confusion et des indicateurs de performance...")

# plots.plot_cm(mean_cm, np.unique(Y))
# plt.title("Matrice de confusion moyenne - {} folds x {} itérations"
#           .format(nb_folds, nb_iter))
# plt.savefig(out_matrix, bbox_inches='tight', dpi=150)
# plt.close()

# fig, ax = plt.subplots(figsize=(10, 7))
# mean_df_report[['precision', 'recall', 'f1-score']].T.plot.bar(
#     yerr=std_df_report[['precision', 'recall', 'f1-score']].T,
#     ax=ax, zorder=2
# )

# ax.text(
#     0.5, 0.95, 
#     f"OA moyenne : {mean_accuracy:.2f} ± {std_accuracy:.2f}",
#     transform=ax.transAxes, fontsize=14
# )

# ax.set_ylim(0, 1)
# ax.set_title("Indicateurs de performance moyens")
# ax.set_ylabel("Score")
# ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
# plt.savefig(out_qualite, bbox_inches='tight', dpi=150)
# plt.close()

# # ------------------------------------------------------------------------------
# # 6) Entraîner un classifieur final sur l'ENSEMBLE des données, puis appliquer
# # ------------------------------------------------------------------------------
# logger.info("Entraînement final du classifieur sur TOUT l'échantillon pour "
#             "générer la carte finale...")

# clf_final = RandomForestClassifier(**rf_params)
# clf_final.fit(X, Y.ravel())

# logger.info("Application du modèle à l'ensemble de l'image...")
# cla.apply_classifier_to_image(
#     clf_final,
#     image_filename,
#     out_classif
# )

# logger.info(f"La carte de classification finale est sauvegardée dans : {out_classif}")