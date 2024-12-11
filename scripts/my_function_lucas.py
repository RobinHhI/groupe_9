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
import logging
import numpy as np
from osgeo import gdal
import geopandas as gpd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

# GDAL configuration
gdal.UseExceptions()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("process_log.log")]
)


def filter_classes(shapefile_path, classe_col, classes_to_keep):
    gdf = gpd.read_file(shapefile_path)
    gdf_filtered = gdf[gdf[classe_col].isin(classes_to_keep)].copy()
    return gdf_filtered


def compute_pixels_by_class(shapefile_path, raster_path, classe_col):
    # Usar zonal_stats com categorical=True
    zs = zonal_stats(shapefile_path, raster_path, categorical=True, nodata=99)
    gdf = gpd.read_file(shapefile_path)
    class_poly_pixels = {}

    for i, stats in enumerate(zs):
        classe = gdf.iloc[i][classe_col]
        forest_pixels = stats.get(1, 0)  # Supondo 1=floresta no raster
        if classe not in class_poly_pixels:
            class_poly_pixels[classe] = []
        class_poly_pixels[classe].append(forest_pixels)
    return class_poly_pixels


def plot_bar_nb_polygons(class_poly_pixels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    class_num_polygons = {cls: len(lst)
                          for cls, lst in class_poly_pixels.items()}
    classes_sorted = sorted(
        class_num_polygons, key=class_num_polygons.get, reverse=True)
    values = [class_num_polygons[cls] for cls in classes_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(classes_sorted, values, color='skyblue', edgecolor='black')
    plt.title("Nombre de polygones par classe", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de polygones", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "diag_baton_nb_poly_by_class_lucas.png"), dpi=300)
    plt.close()


def plot_bar_nb_pixels(class_poly_pixels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    class_total_pixels = {cls: sum(lst)
                          for cls, lst in class_poly_pixels.items()}
    classes_sorted = sorted(
        class_total_pixels, key=class_total_pixels.get, reverse=True)
    values = [class_total_pixels[cls] for cls in classes_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(classes_sorted, values, color='lightgreen', edgecolor='black')
    plt.title("Nombre de pixels du raster par classe", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de pixels", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "diag_baton_nb_pix_by_class_lucas.png"), dpi=300)
    plt.close()


def plot_violin_distribution(class_poly_pixels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    classes = []
    for cls, lst in class_poly_pixels.items():
        data.append(lst)
        classes.append(cls)

    # Ordenar por max de pixels
    max_pixels = [max(l) if len(l) > 0 else 0 for l in data]
    idx_sorted = np.argsort(max_pixels)[::-1]
    data = [data[i] for i in idx_sorted]
    classes = [classes[i] for i in idx_sorted]

    plt.figure(figsize=(12, 8))
    violin_parts = plt.violinplot(
        data, showmeans=False, showextrema=False, showmedians=True)
    cmap = plt.cm.get_cmap('Pastel1')
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(cmap((i+1)/len(data)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    plt.title("Distribution du nombre de pixels par classe de polygone", fontsize=16)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre de pixels par polygone", fontsize=12)
    plt.xticks(np.arange(1, len(classes)+1), classes, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "violin_plot_nb_pix_by_poly_by_class_lucas.png"), dpi=300)
    plt.close()
