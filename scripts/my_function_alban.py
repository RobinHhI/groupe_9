#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Geospatial Data Processing

This module provides functions to filter and reclassify geospatial data,
manage vector files, and process rasters with GDAL, all with integrated logging.

Created on Dec 03, 2024
Last modified: Dec 03, 2024

@author: Alban Dumont, Lucas Lima, Robin Heckendorn
"""
import os
import sys
import traceback
import logging
import numpy as np
from osgeo import gdal
from osgeo import ogr

import geopandas as gpd

# fonction déjà existante dans my_function.py - à ne pas copier
# Configuration GDAL
gdal.UseExceptions()

# Configuration logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Création de la variable logger
logger = logging.getLogger(__name__)


# ================================= #
# ===  FONCTIONS UTILITAIRES  ===== #
# ================================= #

def log_error_and_raise(message, exception=RuntimeError):
    """
    Enregistre un message d'erreur, inclut la trace d'exécution, et déclenche une exception.

    Paramètres :
    -----------
    message : str
        Message d'erreur à enregistrer et à lever.
    exception : Exception
        Type d'exception à lever (par défaut : RuntimeError).
    """
    logger.error(f"{message}\n{traceback.format_exc()}")
    raise exception(message)


def get_raster_properties(dataset):
    """
    Récupère les propriétés d'un raster à partir d'un dataset GDAL.

    Paramètres :
    -----------
    dataset : gdal.Dataset
        Dataset GDAL en entrée.

    Retourne :
    ---------
    tuple
        Propriétés du raster : (largeur_pixel, hauteur_pixel, xmin, ymin, xmax, ymax, crs).

    Exceptions :
    -----------
    ValueError
        Si le dataset est None ou si la résolution ne correspond pas à 10m attendus.
    """
    if dataset is None:
        raise ValueError(
            "Le dataset d'entrée est None. Veuillez vérifier le raster en entrée.")

    geotransform = dataset.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = abs(geotransform[5])  # Assure une hauteur de pixel positive
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + (dataset.RasterXSize * pixel_width)
    # Ajustement pour une hauteur négative
    ymin = ymax - (dataset.RasterYSize * pixel_height)
    crs = dataset.GetProjection()

    # Vérification si la résolution est environ 10m
    if not (abs(pixel_width - 10) < 1e-6 and abs(pixel_height - 10) < 1e-6):
        raise ValueError(
            f"La résolution du raster ne correspond pas à 10m : ({pixel_width}, {pixel_height})")

    return pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs
#fin des fonctions déjà existantes


def create_raster_sampleimage(sample_vector, reference_raster, output_path, attribute):
    """
    Crée un raster forêt pour la classification   :
        - .

    Paramètres :
    -----------
    sample_vector : str
        Chemin vers le fichier vecteur représentant les zones forestières.
    reference_raster : str
        Chemin vers le raster de référence.
    clip_vector : str
        Chemin vers le shapefile définissant l'étendue de la zone d'étude.
    output_path : str
        Chemin pour sauvegarder le raster masque final.

    Exceptions :
    -----------
    RuntimeError
        Si une étape du processus échoue.
    """
    try:
        logger.info("Rasteration à partir du fichier sample forêt...")
        # Ouvrir le raster du masque
        reference_ds = gdal.Open(reference_raster)
        if reference_ds is None:
            log_error_and_raise(
                f"Impossible d'ouvrir le raster de reference : {reference_raster}")

        logger.info("récupération des propriétés du raster de référence")
        pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs = get_raster_properties(
            reference_ds)

        # Calcul du nombre de pixels en directions x et y
        x_pixels = int((xmax - xmin) / pixel_width)
        y_pixels = int((ymax - ymin) / pixel_height)

        # Étape 2 : Création d'un raster en mémoire pour rasteriser le masque vecteur
        logger.info("Rasterisation du sample forêt...")
        mem_driver = gdal.GetDriverByName('MEM')

        if (attribute == "ID") :
            out_raster = mem_driver.Create(
                '', x_pixels, y_pixels, 1, gdal.GDT_UInt16)
        else :    
            out_raster = mem_driver.Create(
                '', x_pixels, y_pixels, 1, gdal.GDT_Byte)
        out_raster.SetGeoTransform(
            (xmin, pixel_width, 0, ymax, 0, -pixel_height))
        out_raster.SetProjection(crs)

        # Initialiser le raster avec la valeur 0 (non-forêt)
        # Les bandes sont indexées à partir de 1
        out_band = out_raster.GetRasterBand(1)
        out_band.Fill(0)
        out_band.SetNoDataValue(0)

        # Ouvrir le masque vecteur
        vector_ds = gdal.OpenEx(sample_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            log_error_and_raise( 
                f"Impossible d'ouvrir le fichier vecteur : {sample_vector}")
        vector_layer = vector_ds.GetLayer()
        if (attribute == "ID") :
            
            sample_folder = "groupe_9/results/data/sample"
            filename_tmp = os.path.join(sample_folder, "tmp_id.shp")
                                          
            # Vérifier si le fichier destination existe déjà
            if os.path.exists(filename_tmp):
                os.remove(filename_tmp)

            # Créer un nouveau fichier de destination
            driver = ogr.GetDriverByName("ESRI Shapefile")
            destination_dataset = driver.CreateDataSource(filename_tmp)
            if destination_dataset is None:
                log_error_and_raise("Impossible de créer le fichier tmp.")
                
            # Créer une nouvelle couche dans le dataset de destination
            destination_layer = destination_dataset.CreateLayer(
                vector_layer.GetName(), vector_layer.GetSpatialRef(), 
                geom_type=vector_layer.GetGeomType(), options=["ENCODING=UTF-8"])

            # Copier les champs de la couche source
            source_layer_def = vector_layer.GetLayerDefn()
            for i in range(source_layer_def.GetFieldCount()):
                field_def = source_layer_def.GetFieldDefn(i)
                destination_layer.CreateField(field_def)

            # Parcourir les entités de la couche source
            for feature in vector_layer:
                # Créer une nouvelle entité dans la couche de destination
                new_feature = ogr.Feature(destination_layer.GetLayerDefn())

                # Copier les valeurs des champs
                FieldCount = source_layer_def.GetFieldCount()
                for i in range(FieldCount):
                    Nameref = source_layer_def.GetFieldDefn(i).GetNameRef()
                    Field = feature.GetField(i)
                    new_feature.SetField(Nameref, Field)

                # Copier la géométrie
                geom = feature.GetGeometryRef()
                new_feature.SetGeometry(geom.Clone())

                # Ajouter la nouvelle entité à la couche de destination
                destination_layer.CreateFeature(new_feature)

                # Nettoyer
                new_feature = None
            
            # Définir le nom des champs
            new_field_name = "temp_" + attribute

            # Obtenir la définition des champs
            layer_def = destination_layer.GetLayerDefn()

            # Vérifier si le champ à modifier existe
            old_field_index = layer_def.GetFieldIndex(attribute)
            if old_field_index == -1:
                log_error_and_raise( f"Erreur : le champ '{attribute}' n'existe pas.")

            # Ajouter un nouveau champ temporaire avec le type souhaité
            new_field = ogr.FieldDefn(new_field_name, ogr.OFTInteger)  # Exemple : Integer
            destination_layer.CreateField(new_field)

            # Copier les données de l'ancien champ vers le nouveau champ
            for feature in destination_layer:
                FID = feature.GetFID()
                feature.SetField(new_field_name, FID + 1)  # Conversion en entier
                destination_layer.SetFeature(feature)

            # Supprimer l'ancien champ
            destination_layer.DeleteField(old_field_index)

            # Renommer le nouveau champ pour qu'il remplace l'ancien
            layer_def = destination_layer.GetLayerDefn()  # Rafraîchir la définition
            src_fdef = layer_def.GetFieldDefn(layer_def.GetFieldIndex(new_field_name))
            fdef = ogr.FieldDefn(src_fdef.GetName(), src_fdef.GetType())
            fdef.SetWidth(src_fdef.GetWidth())
            fdef.SetPrecision(src_fdef.GetPrecision())
            fdef.SetName(attribute)
            fdef.SetType(ogr.OFTInteger)
            new_field_index = layer_def.GetFieldIndex(new_field_name)
            
            destination_layer.AlterFieldDefn(new_field_index, fdef, ogr.ALTER_NAME_FLAG)
            
            
        # Rasteriser le sample forêt avec la valeur de l'attribut Code
        if (attribute == 'ID') : 
            err = gdal.RasterizeLayer(
                out_raster, [1], destination_layer, options=["ATTRIBUTE={}".format(attribute)])
            destination_dataset = None
        else :
            err = gdal.RasterizeLayer(
                out_raster, [1], vector_layer, options=["ATTRIBUTE={}".format(attribute)])
        if err != 0:
            log_error_and_raise("Échec de la rasterisation.")

        # Fermer le fichier vecteur
        vector_ds = None

        # Étape 4 : Sauvegarde du masque découpé
        # Supprimer le fichier existant avant sauvegarde
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Fichier existant supprimé : {output_path}")
            except PermissionError:
                log_error_and_raise(
                    f"Impossible de supprimer le fichier existant : {output_path}")

        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.CreateCopy(
            output_path, out_raster, options=["COMPRESS=LZW"])
        if output_ds is None:
            log_error_and_raise(
                f"Échec de la sauvegarde du raster sample forêt : {output_path}")

        # Finaliser et fermer les datasets
        output_ds.FlushCache()
        output_ds = None
        out_raster = None

        logger.info(f"Fiche sample forêt rasterisé sauvegardé : {output_path}")

    except Exception as e:
        log_error_and_raise(
            f"Erreur pendant la création du raster sample forêt : {e}")