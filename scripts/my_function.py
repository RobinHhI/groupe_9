#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Geospatial Data Processing

This module provides functions to filter and reclassify geospatial data,
manage vector files, and process rasters with GDAL.

Created on Dec 03, 2024
Last modified: Dec 03, 2024

@author: Alban Dumont, Lucas Lima, Robin Heckendorn
"""

import geopandas as gpd
from osgeo import gdal
gdal.UseExceptions()

# === RECLASSIFICATION FUNCTIONS ===


def reclassification_dictionary():
    """
    Return the reclassification dictionary for BD Forêt.

    Returns:
    -------
    dict
        A dictionary mapping the `TFV` field to a tuple with:
        - `Code` (int): The classification code.
        - `Nom` (str): The classification name.
    """
    return {
        "Forêt fermée de châtaignier pur": (11, "Autres feuillus"),
        "Forêt fermée de hêtre pur": (11, "Autres feuillus"),
        "Forêt fermée de chênes décidus purs": (12, "Chêne"),
        "Forêt fermée de robinier pur": (13, "Robinier"),
        "Peupleraie": (14, "Peupleraie"),
        "Forêt fermée à mélange de feuillus": (15, "Mélange de feuillus"),
        "Forêt fermée de feuillus purs en îlots": (16, "Feuillus en îlots"),
        "Forêt fermée d’un autre conifère pur autre que pin": (21, "Autres conifères autre que pin"),
        "Forêt fermée de mélèze pur": (21, "Autres conifères autre que pin"),
        "Forêt fermée de sapin ou épicéa": (21, "Autres conifères autre que pin"),
        "Forêt fermée à mélange d’autres conifères": (21, "Autres conifères autre que pin"),
        "Forêt fermée de pin sylvestre pur": (22, "Autres Pin"),
        "Forêt fermée à mélange de pins purs": (22, "Autres Pin"),
        "Forêt fermée de douglas pur": (23, "Douglas"),
        "Forêt fermée de pin laricio ou pin noir pur": (24, "Pin laricio ou pin noir"),
        "Forêt fermée de pin maritime pur": (25, "Pin maritime"),
        "Forêt fermée à mélange de conifères": (26, "Mélange conifères"),
        "Forêt fermée de conifères prépondérants et feuillus": (28, "Mélange de conifères prépondérants et feuillus"),
        "Forêt fermée à mélange de feuillus prépondérants et conifères": (29, "Mélange de feuillus prépondérants et conifères"),
    }


def filter_and_reclassify(gdf):
    """
    Filter and reclassify a GeoDataFrame based on the BD Forêt system.

    Parameters:
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame containing the BD Forêt data.

    Returns:
    -------
    GeoDataFrame
        A filtered and reclassified GeoDataFrame with only the attributes `Nom`, `Code`, and `geometry`.
    """
    # Get the reclassification dictionary
    reclassification = reclassification_dictionary()

    # Filter based on the categories in the dictionary
    categories_to_keep = list(reclassification.keys())
    filtered_gdf = gdf[gdf["TFV"].isin(categories_to_keep)].copy()

    # Add `Nom` and `Code` attributes based on the reclassification
    filtered_gdf["Nom"] = filtered_gdf["TFV"].map(
        lambda x: reclassification[x][1])
    filtered_gdf["Code"] = filtered_gdf["TFV"].map(
        lambda x: reclassification[x][0])

    # Keep only the columns `Nom`, `Code`, and `geometry`
    filtered_gdf = filtered_gdf[["Nom", "Code", "geometry"]]

    return filtered_gdf

# === VECTOR FILE FUNCTIONS ===


def save_vector_file(gdf, output_path):
    """
    Save a GeoDataFrame to a vector file (e.g., Shapefile, GeoJSON).

    Parameters:
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame to save.
    output_path : str
        Path to save the output file, including the filename and extension.

    Returns:
    -------
    None
    """
    gdf.to_file(output_path, driver="ESRI Shapefile")

# === RASTER PROCESSING FUNCTIONS ===


def get_raster_properties(dataset):
    """
    Get raster properties from a GDAL dataset.

    Parameters:
    ----------
    dataset : gdal.Dataset
        Input GDAL dataset.

    Returns:
    -------
    tuple
        Raster properties: (pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs).
    """
    geotransform = dataset.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + (dataset.RasterXSize * pixel_width)
    ymin = ymax + (dataset.RasterYSize * pixel_height)
    crs = dataset.GetProjection()
    return pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs


def reproject_raster_to_memory(input_raster, target_srs):
    """
    Reproject a raster in memory to a target CRS.

    Parameters:
    ----------
    input_raster : str
        Path to the input raster.
    target_srs : str
        Target CRS (e.g., 'EPSG:2154').

    Returns:
    -------
    gdal.Dataset
        GDAL dataset of the reprojected raster in memory.
    """
    return gdal.Warp('', input_raster, dstSRS=target_srs, format="MEM", xRes=10, yRes=10, resampleAlg="bilinear")
