#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classification supervisée  à l'échelle du pixel 

Ce script effectue les étapes nécessaires pour effectuer classification supervisée avec 
le classifieur RandomForestClassifier :
1. 
2.
3. 
4. 

Créé le 28 décembre 2024
"""
import os
import sys
sys.path.append('libsigma')
import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
from datetime import datetime


from my_function_alban import create_raster_sampleimage

# personal librairies
import classification as cla
import read_and_write as rw
import plots


def report_from_dict_to_df(dict_report):


    # convert report into dataframe
    report_df = pd.DataFrame.from_dict(dict_report)

    # drop unnecessary rows and columns
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    except KeyError:
        print(dict_report)
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    report_df = report_df.drop(['support'], axis=0)

    return report_df

print(" Debut : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
'2018-07-17 22:54:25'
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
out_matrix = os.path.join(out_classif_folder, 'ma_matrice.png')
out_qualite = os.path.join(out_classif_folder, 'mes_qualites.png')
out_std_dev_mean = os.path.join(out_classif_folder, 'Std_Dev_and_Mean.png')
out_std_deviation_pkl =  os.path.join(out_classif_folder, 'std_deviation.pkl')
out_mean_pkl =  os.path.join(out_classif_folder, 'out_mean.pkl')
out_std_deviation_csv =  os.path.join(out_classif_folder, 'std_deviation.csv')
out_mean_csv =  os.path.join(out_classif_folder, 'mean.csv')

# Sample parameters
nb_iter = 30

print(" Création du raster Sample : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
create_raster_sampleimage(sample_filename, image_reference, raster_sample_filename, "Code")
print(" Création du raster Sample Id : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
create_raster_sampleimage(sample_filename, image_reference, raster_sample_id_filename, "ID")


# 2 --- extract samples
print(" Obtention du Sample X, Y, t  : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
X, Y, t = cla.get_samples_from_roi(image_filename, raster_sample_filename)
print(" Obtention des groupes : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
_, groups, _ = cla.get_samples_from_roi(image_filename, raster_sample_id_filename)

# Valeurs à remplacer
values_to_replace = [15, 16, 26,27, 28 ,29]

# Remplacement
Y[np.isin(Y, values_to_replace)] = 0

list_cm = []
list_accuracy = []
list_report = []
groups = np.squeeze(groups)

# Iter on stratified K fold
print(" Création Kfold Statifié : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
kf = StratifiedKFold(n_splits=nb_iter)
for train, test in kf.split(X, Y):
    print(" Entrainement sur tous les Kfolds : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]

    # 3 --- Train
    #clf = SVC(cache_size=6000)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)

    # 4 --- Test
    Y_predict = clf.predict(X_test)

    Y_predict[Y_predict == 4] = 3
    Y_test[Y_test == 4] = 3
    # compute quality
    list_cm.append(confusion_matrix(Y_test, Y_predict))
    list_accuracy.append(accuracy_score(Y_test, Y_predict))
    report = classification_report(Y_test, Y_predict,
                                   labels=np.unique(Y_predict), output_dict=True)

    # store them
    list_report.append(report_from_dict_to_df(report))

print(" Sauvegarde des résultats : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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

print(" Prédiction sur toute la zone : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
Y_predict = clf.predict(X)
print(np.unique(Y_predict))

# reshape
print(" Sauvegarde de la nouvelle Carte : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
ds = rw.open_image(image_filename)
nb_row, nb_col, _ = rw.get_image_dimension(ds)

img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
img[t[0], t[1], 0] = Y_predict

# write image
rw.write_image(out_classif, img, data_set=ds, nb_band=1)
print(" Fin : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

