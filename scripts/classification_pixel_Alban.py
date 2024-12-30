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
import re  # pour utiliser fonction regex
import time  # Pour calculer le temp de traitement
import logging
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
import geopandas as gpd
from datetime import datetime

from my_function_alban import create_raster_sampleimage

# personal librairies
import classification as cla
import read_and_write as rw
import plots
"""
from my_function import (find_raster_bands, reproject_raster, clip_raster_to_extent,
                         resample_raster, apply_mask, merge_rasters,
                         calculate_ndvi_from_processed_bands)
"""

# Démarrage du chronomètre de traitement total
total_start_time = time.time()


# 1 --- define parameters
# inputs
sample_folder = "groupe_9/results/data/sample"
pretraitees_folder = "groupe_9/results/data/img_pretraitees"
sample_filename = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.shp")
image_reference = os.path.join(pretraitees_folder, "masque_foret.tif")
image_filename = os.path.join(pretraitees_folder, "Serie_temp_S2_ndvi.tif")
raster_sample_filename = os.path.join(sample_folder, "Sample_BD_foret_T31TCJ.tif")


# Sample parameters
test_size = 0.5

# outputs
out_classif_folder = 'groupe_9/results/data/classif'
classif_filename = 'carte_essences_echelle_pixel.tif'
out_classif = os.path.join(out_classif_folder, classif_filename)
out_matrix = os.path.join(out_classif_folder, 'ma_matrice.png')
out_qualite = os.path.join(out_classif_folder, 'mes_qualites.png')


# create_raster_sampleimage(sample_filename, image_reference, raster_sample_filename)

# 2 --- extract samples
X, Y, t = cla.get_samples_from_roi(image_filename, raster_sample_filename)
    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

Y_train = np.ravel(Y_train)
print(Y_train[0:10])

max_depth =	50
oob_score = True
max_samples = 0.75
class_weight = "balanced"

# 3 --- Train
#clf = SVC(cache_size=6000)
clf = RF(max_depth=max_depth, oob_score=oob_score, max_samples=max_samples, class_weight=class_weight)
#clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
clf.fit(X_train, Y_train)

# 4 --- Test
Y_predict = clf.predict(X_test)

#Y_test = np.squeeze(Y_test)
#print(Y_test[0:5])

# Convert it into two dimensions array
Y_predict = np.atleast_2d(Y_predict)

# Inverse the dimensions
Y_predict = Y_predict.T

print(np.unique(Y_predict))
print(np.unique(Y))


# compute quality
cm = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict, labels=np.unique(Y_predict), output_dict=True)
accuracy = accuracy_score(Y_test, Y_predict)

# display and save quality
plots.plot_cm(cm, np.unique(Y_predict), out_filename=out_matrix)
plots.plot_class_quality(report, accuracy, out_filename=out_qualite)

Y_predict = clf.predict(X)
print(np.unique(Y_predict))

# reshape
ds = rw.open_image(image_filename)
nb_row, nb_col, _ = rw.get_image_dimension(ds)

img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
img[t[0], t[1], 0] = Y_predict

# write image
ds = rw.open_image(image_filename)
rw.write_image(out_classif, img, data_set=ds, gdal_dtype=None,
            transform=None, projection=None, driver_name=None,
            nb_col=None, nb_ligne=None, nb_band=1)