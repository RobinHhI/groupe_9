#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Geospatial Data Filtering and Reclassification

This module provides functions to filter and reclassify geospatial data based
on the BD Forêt classification system.

Created on Dec 03, 2024
Last modified: Dec 03, 2024

@author: Lucas Pereira das Neves Souza Lima
"""

import geopandas as gpd


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
        A filtered and reclassified GeoDataFrame with only the attributes `Nom` and `Code`.
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

    # Keep only the columns `Nom` and `Code`
    filtered_gdf = filtered_gdf[["Nom", "Code", "geometry"]]

    return filtered_gdf


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
