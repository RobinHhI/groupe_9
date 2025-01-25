#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classification supervisée  à l'échelle du pixel

Ce script effecture la production d'une d'essences forestières à l'échelle du pixel sur 
l'ensemble de l'emprise :
1. 
2.
3. 
4. 

Créé le 28 décembre 2024
"""
import os
import sys
import logging
import time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd

sys.path.append('libsigma')
import classification as cla
import read_and_write as rw
import plots

# personal librairies
from my_function_alban import create_raster_sampleimage, report_from_dict_to_df

# Démarrage du chronomètre de traitement total
total_start_time = time.time()

logging.info("Debut de la production d'une d'essences forestières à l'échelle du pixel: ")

# 1 --- define parameters
# inputs
sample_folder = "groupe_9/results/data/sample"
pretraitees_folder = "groupe_9/results/data/img_pretraitees"
sample_filename = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.shp")
image_reference = os.path.join(pretraitees_folder, "masque_foret.tif")
image_filename = os.path.join(pretraitees_folder, "Serie_temp_S2_ndvi.tif")
raster_sample_filename = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.tif")
raster_sample_id_filename = os.path.join(sample_folder, "Sample_id_BD_foret_T31TCJ.tif")

# outputs
out_classif_folder = 'groupe_9/results/data/classif'
out_classif = os.path.join(out_classif_folder, 'carte_essences_echelle_pixel.tif')
out_std_dev_mean = os.path.join(out_classif_folder, 'Std_Dev_and_Mean.png')
out_std_deviation_pkl = os.path.join(out_classif_folder, 'std_deviation.pkl')
out_mean_pkl = os.path.join(out_classif_folder, 'mean.pkl')
out_std_deviation_csv = os.path.join(out_classif_folder, 'std_deviation.csv')
out_mean_csv = os.path.join(out_classif_folder, 'mean.csv')

# Sample parameters
nb_iter = 30
nb_folds = 5

logging.info("Création du raster Sample")
create_raster_sampleimage(sample_filename, image_reference, raster_sample_filename, "Code")
logging.info("Création du raster Sample Id")
create_raster_sampleimage(sample_filename, image_reference, raster_sample_id_filename, "ID")


# 2 --- extract samples
logging.info("Obtention du Sample X, Y, t ")
X, Y, t = cla.get_samples_from_roi(image_filename, raster_sample_filename)
logging.info("Obtention des groupes")
_, groups, _ = cla.get_samples_from_roi(image_filename, raster_sample_id_filename)

# Valeurs à supprimer
values_to_delete = [15, 16, 26, 27, 28, 29]
essence_tree = [11, 12, 13, 14, 21, 22, 23, 24, 25]

# suppression
Index_Essence_to_Delete = np.where(np.isin(Y, values_to_delete))

Y_skf = np.delete(Y, Index_Essence_to_Delete, 0)
X_skf = np.delete(X, Index_Essence_to_Delete, 0)
groups = np.delete(groups, Index_Essence_to_Delete, 0)

list_cm = []
list_accuracy = []
list_report = []
groups = np.squeeze(groups)

# Iter on stratified K fold
logging.info(f"Entrainement sur tous les Kfolds avec {nb_iter} itérations")
max_depth = 50
n_jobs = 50
oob_score = True
max_samples = 0.75
class_weight = "balanced"
clf = RF(max_depth=max_depth,
        oob_score=oob_score,
        n_jobs=n_jobs,
        class_weight=class_weight,
        max_samples=max_samples)

# Iter on stratified K fold
for iteration in range(nb_iter):
    logging.info(f"Iteration principal : {iteration + 1} : Création des Kfolds Stratifiés")
    skf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)

    for i, (train, test) in enumerate(skf.split(X_skf, Y_skf, groups=groups)):
        logging.info(f"Itération kfold : {i + 1}")
        X_train, X_test = X_skf[train], X_skf[test]
        Y_train, Y_test = Y_skf[train], Y_skf[test]

        # vérfie si une essence d'arbre est absente dans le jeu de test
        if not np.array_equal(np.unique(Y_test), np.array(essence_tree)):
            continue

        # 3 --- Train
        Y_train = np.ravel(Y_train)
        clf.fit(X_train, Y_train)

        # 4 --- Test
        Y_predict = clf.predict(X_test)

        list_cm.append(confusion_matrix(Y_test, Y_predict))
        list_accuracy.append(accuracy_score(Y_test, Y_predict))
        report = classification_report(Y_test, Y_predict,
                                        labels=np.unique(Y_predict), output_dict=True)

        # store them
        list_report.append(report_from_dict_to_df(report))

logging.info("Sauvegarde des résultats")
# compute mean of cm
array_cm = np.array(list_cm)
cm_mean = array_cm.mean(axis=0)

# compute mean and std of overall accuracy
array_accuracy = np.array(list_accuracy)
mean_accuracy = array_accuracy.mean()
std_accuracy = array_accuracy.std()

# compute mean and std of classification report
array_report = np.array(list_report)
mean_report = array_report.mean(axis=0)
std_report = array_report.std(axis=0)
a_report = list_report[0]

mean_df_report = pd.DataFrame(mean_report, index=a_report.index,
                              columns=a_report.columns)
std_df_report = pd.DataFrame(std_report, index=a_report.index,
                             columns=a_report.columns)

# sauvegarde les résultats en format pickle
# permet de sauvegarder le type de données de chaque colonne 
mean_df_report.to_pickle(out_mean_pkl)
std_df_report.to_pickle(out_std_deviation_pkl)
# et en csv, pour être plus facilement manipulable en dehors de python
mean_df_report.to_csv(out_mean_csv)
std_df_report.to_csv(out_std_deviation_csv)
# sauvegarde avec en image
plots.plot_mean_class_quality(list_report, list_accuracy, out_std_dev_mean)

logging.info("Prédiction sur toute la zone")
Y_predict = clf.predict(X)

# reshape
logging.info("Sauvegarde de la nouvelle Carte")
ds = rw.open_image(image_filename)
nb_row, nb_col, _ = rw.get_image_dimension(ds)

img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
img[t[0], t[1], 0] = Y_predict

# write image
rw.write_image(out_classif, img, data_set=ds, nb_band=1)

# Fin du chronomètre pour tout le traitement
total_end_time = time.time()
total_duration = total_end_time - total_start_time
minutes, seconds = divmod(total_duration, 60)
logging.info(
    f"Total processing time: {int(minutes)} minutes and {seconds:.2f} seconds.")

print(
    f"Traitement terminé avec succès en {int(minutes)} minutes et {seconds:.2f} secondes")
