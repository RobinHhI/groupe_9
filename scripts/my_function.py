#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Geospatial Data Processing

This module provides functions to filter and reclassify geospatial data,
manage vector files, and process rasters with GDAL, all with integrated logger.

Created on Dec 03, 2024
Last modified: Dec 11, 2024

@author: Alban Dumont, Lucas Lima, Robin Heckendorn
"""
import os
import logging
import sys
import traceback
import numpy as np
from osgeo import gdal
import geopandas as gpd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

# GDAL configuration
gdal.UseExceptions()

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Using the logger
logger = logging.getLogger(__name__)

# =========================== #
# === UTILITY FUNCTIONS ===== #
# =========================== #


def log_error_and_raise(message, exception=RuntimeError):
    """
    Logs an error message, includes traceback, and raises an exception.

    Parameters:
    ----------
    message : str
        Error message to log and raise.
    exception : Exception
        Type of exception to raise (default: RuntimeError).
    """
    logger.error(f"{message}\n{traceback.format_exc()}")
    raise exception(message)

# ================================== #
# === RECLASSIFICATION BD FÔRET ==== #
# ================================== #


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
        "Forêt fermée d’un autre feuillu pur": (11, "Autres feuillus"),
        "Forêt fermée de châtaignier pur": (11, "Autres feuillus"),
        "Forêt fermée de hêtre pur": (11, "Autres feuillus"),

        "Forêt fermée de chênes décidus purs": (12, "Chêne"),
        "Forêt fermée de robinier pur": (13, "Robinier"),
        "Peupleraie": (14, "Peupleraie"),
        "Forêt fermée à mélange de feuillus": (15, "Mélange de feuillus"),
        "Forêt fermée de feuillus purs en îlots": (16, "Feuillus en îlots"),

        "Forêt fermée d’un autre conifère pur autre que pin ": (21, "Autres conifères autre que pin"),
        "Forêt fermée de mélèze pur": (21, "Autres conifères autre que pin"),
        "Forêt fermée de sapin ou épicéa": (21, "Autres conifères autre que pin"),
        "Forêt fermée à mélange d’autres conifères": (21, "Autres conifères autre que pin"),

        "Forêt fermée d’un autre pin pur": (22, "Autres pin"),
        "Forêt fermée de pin sylvestre pur": (22, "Autres pin"),
        "Forêt fermée à mélange de pins purs": (22, "Autres pin"),

        "Forêt fermée de douglas pur": (23, "Douglas"),
        "Forêt fermée de pin laricio ou pin noir pur": (24, "Pin laricio ou pin noir"),
        "Forêt fermée de pin maritime pur": (25, "Pin maritime"),
        "Forêt fermée à mélange de conifères": (26, "Mélange conifères"),
        "Forêt fermée de conifères purs en îlots": (27, "Conifères en îlots"),

        "Forêt fermée à mélange de conifères prépondérants et feuillus": (28, "Mélange de conifères prépondérants et feuillus"),
        "Forêt fermée à mélange de feuillus prépondérants et conifères": (29, "Mélange de feuillus préponderants conifères")

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
        A filtered and reclassified GeoDataFrame with existing attributes plus `Nom` and `Code`.
    """
    try:
        logger.info("Starting filter and reclassification of GeoDataFrame.")

        # Get the reclassification dictionary
        reclassification = reclassification_dictionary()

        # Filter based on the categories in the dictionary
        categories_to_keep = list(reclassification.keys())
        filtered_gdf = gdf[gdf["TFV"].isin(categories_to_keep)].copy()

        if filtered_gdf.empty:
            log_error_and_raise(
                "No features found after filtering based on TFV categories.")

        logger.info(
            f"Filtered GeoDataFrame to {len(filtered_gdf)} features based on TFV categories.")

        # Add `Nom` and `Code` attributes based on the reclassification
        filtered_gdf["Nom"] = filtered_gdf["TFV"].map(
            lambda x: reclassification[x][1])
        filtered_gdf["Code"] = filtered_gdf["TFV"].map(
            lambda x: reclassification[x][0])

        logger.info("Added 'Nom' and 'Code' attributes to GeoDataFrame.")

        return filtered_gdf

    except Exception as e:
        log_error_and_raise(
            f"Error during filtering and reclassification: {e}")


# ============================= #
# === VECTOR FILE FUNCTIONS === #
# ============================= #
def clip_vector_to_extent(gdf, clip_shapefile):
    """
    Clip a GeoDataFrame to the extent of a shapefile without adding attributes from the clipping shapefile.

    Parameters:
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame to clip.
    clip_shapefile : str
        Path to the shapefile defining the clipping extent.

    Returns:
    -------
    GeoDataFrame
        Clipped GeoDataFrame with attributes only from the input GeoDataFrame.

    Raises:
    ------
    RuntimeError
        If the clipping process fails.
    """
    try:
        logger.info(
            f"Clipping GeoDataFrame using shapefile: {clip_shapefile}")

        # Read the clipping shapefile
        clip_gdf = gpd.read_file(clip_shapefile)
        if clip_gdf.empty:
            log_error_and_raise(
                f"Clipping shapefile {clip_shapefile} is empty.")

        # Ensure CRS matches
        if gdf.crs != clip_gdf.crs:
            logger.info(
                "Reprojecting GeoDataFrame to match clipping shapefile CRS.")
            gdf = gdf.to_crs(clip_gdf.crs)

        # Perform the clipping using geopandas.clip
        clipped_gdf = gpd.clip(gdf, clip_gdf)

        if clipped_gdf.empty:
            log_error_and_raise("No features found after clipping operation.")

        logger.info(
            f"Clipped GeoDataFrame contains {len(clipped_gdf)} features.")

        return clipped_gdf

    except Exception as e:
        log_error_and_raise(f"Error during clipping GeoDataFrame: {e}")


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
    try:
        logger.info(f"Saving GeoDataFrame to file: {output_path}")
        gdf.to_file(output_path, driver="ESRI Shapefile")
        logger.info("GeoDataFrame saved successfully.")
    except Exception as e:
        log_error_and_raise(f"Error saving GeoDataFrame to file: {e}")


# =================================== #
# === RASTER PROCESSING FUNCTIONS === #
# =================================== #

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

    Raises:
    ------
    ValueError
        If the dataset is None or the resolution does not match the expected 10m.
    """
    if dataset is None:
        raise ValueError(
            "Input dataset is None. Please verify the input raster.")

    geotransform = dataset.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = abs(geotransform[5])  # Ensure positive pixel height
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + (dataset.RasterXSize * pixel_width)
    # Adjust for negative height
    ymin = ymax - (dataset.RasterYSize * pixel_height)
    crs = dataset.GetProjection()

    # Validate resolution is approximately 10m
    if not (abs(pixel_width - 10) < 1e-6 and abs(pixel_height - 10) < 1e-6):
        raise ValueError(
            f"Raster resolution does not match 10m: ({pixel_width}, {pixel_height})")

    return pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs


def reproject_raster(input_raster, target_srs, x_res=10, y_res=10, resample_alg="bilinear"):
    """
    Reproject a raster to a target CRS in memory with specified resolution.

    Parameters:
    ----------
    input_raster : str
        Path to the input raster.
    target_srs : str
        Target spatial reference system (e.g., 'EPSG:2154').
    x_res : float, optional
        Desired pixel width (default: 10).
    y_res : float, optional
        Desired pixel height (default: 10).
    resample_alg : str, optional
        Resampling algorithm to use (default: "bilinear").

    Returns:
    -------
    gdal.Dataset
        Reprojected raster in memory.

    Raises:
    ------
    RuntimeError
        If the reprojection fails.
    """
    try:
        logger.info(
            f"Reprojecting raster: {input_raster} to {target_srs} with resolution {x_res}x{y_res}")
        reprojected = gdal.Warp(
            '', input_raster, dstSRS=target_srs, format="MEM",
            xRes=x_res, yRes=y_res, resampleAlg=resample_alg
        )
        if reprojected is None:
            log_error_and_raise(f"Failed to reproject raster: {input_raster}")
        return reprojected
    except Exception as e:
        log_error_and_raise(f"Error during reprojection: {e}")


def create_forest_mask(mask_vector, reference_raster, clip_vector, output_path):
    """
    Create a forest mask raster aligned with a reference raster, with:
        - 1 for forest
        - 0 for non-forest
        - 99 for NoData areas.

    Parameters:
    ----------
    mask_vector : str
        Path to the vector file representing forest areas.
    reference_raster : str
        Path to the reference raster.
    clip_vector : str
        Path to the shapefile defining the study area extent.
    output_path : str
        Path to save the final forest mask raster.

    Raises:
    ------
    RuntimeError
        If any step in the process fails.
    """
    try:
        logger.info("Creating forest mask...")

        # Step 1: Reproject the reference raster to get the desired properties
        logger.info("Reprojecting reference raster to EPSG:2154...")
        ref_raster_reprojected = reproject_raster(
            reference_raster, "EPSG:2154")
        pixel_width, pixel_height, xmin, ymin, xmax, ymax, crs = get_raster_properties(
            ref_raster_reprojected)

        # Release the reference raster after extracting properties
        ref_raster_reprojected = None

        # Align bounds to match the resolution
        xmin_aligned = xmin - (xmin % pixel_width)
        ymin_aligned = ymin - (ymin % pixel_height)
        xmax_aligned = xmax + (pixel_width - (xmax % pixel_width))
        ymax_aligned = ymax + (pixel_height - (ymax % pixel_height))

        # Calculate the number of pixels in x and y directions
        x_pixels = int((xmax_aligned - xmin_aligned) / pixel_width)
        y_pixels = int((ymax_aligned - ymin_aligned) / pixel_height)

        # Step 2: Create an in-memory raster to rasterize the vector mask
        logger.info("Rasterizing forest mask...")
        mem_driver = gdal.GetDriverByName('MEM')
        out_raster = mem_driver.Create(
            '', x_pixels, y_pixels, 1, gdal.GDT_Byte)
        out_raster.SetGeoTransform(
            (xmin_aligned, pixel_width, 0, ymax_aligned, 0, -pixel_height))
        out_raster.SetProjection(crs)

        # Initialize raster with value 0 (non-forest)
        out_band = out_raster.GetRasterBand(1)  # Bands are 1-indexed
        out_band.Fill(0)
        out_band.SetNoDataValue(99)

        # Open the vector mask
        vector_ds = gdal.OpenEx(mask_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            log_error_and_raise(f"Cannot open vector file: {mask_vector}")

        # Rasterize the vector mask onto the raster, burning value 1 (forest)
        err = gdal.RasterizeLayer(
            out_raster, [1], vector_ds.GetLayer(), burn_values=[1])
        if err != 0:
            log_error_and_raise("Rasterization failed.")

        # Close the vector dataset
        vector_ds = None

        # Step 3: Clip the rasterized mask to the study area
        logger.info("Clipping the forest mask to the study area...")
        clipped_mask = clip_raster_to_extent(
            out_raster, clip_vector, nodata_value=99)

        if clipped_mask is None:
            log_error_and_raise("Failed to clip the forest mask.")

        # Step 4: Save the clipped mask to the output path
        # Before saving, ensure any existing file is deleted
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Removed existing file: {output_path}")
            except PermissionError:
                log_error_and_raise(
                    f"Failed to remove existing file: {output_path}")

        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.CreateCopy(
            output_path, clipped_mask, options=["COMPRESS=LZW"])
        if output_ds is None:
            log_error_and_raise(
                f"Failed to save the forest mask to: {output_path}")

        # Flush and close the output dataset
        output_ds.FlushCache()
        output_ds = None

        # Release datasets
        out_raster = None
        clipped_mask = None

        logger.info(f"Forest mask saved to: {output_path}")

    except Exception as e:
        log_error_and_raise(f"Error during forest mask creation: {e}")


def clip_raster_to_extent(input_raster, clip_vector, nodata_value):
    """
    Clip a raster to a shapefile's extent.

    Parameters:
    ----------
    input_raster : gdal.Dataset or str
        Input raster in memory or file path.
    clip_vector : str
        Path to the shapefile for clipping.
    nodata_value : int or float
        NoData value to set in the output raster.

    Returns:
    -------
    gdal.Dataset
        Clipped raster in memory.

    Raises:
    ------
    RuntimeError
        If the clipping process fails.
    """
    try:
        logger.info(f"Clipping raster with shapefile: {clip_vector}")
        if not os.path.exists(clip_vector):
            log_error_and_raise(
                f"Shapefile not found: {clip_vector}", FileNotFoundError)

        clipped = gdal.Warp(
            '', input_raster, format="MEM",
            cutlineDSName=clip_vector, cropToCutline=True,
            dstNodata=nodata_value
        )
        if clipped is None:
            log_error_and_raise(
                f"Failed to clip raster with shapefile: {clip_vector}")

        # Release the input raster after clipping
        if isinstance(input_raster, gdal.Dataset):
            input_raster = None
        return clipped
    except Exception as e:
        log_error_and_raise(f"Error during clipping: {e}")


def resample_raster(input_raster, pixel_size):
    """
    Resample a raster to a specific resolution.

    Parameters:
    ----------
    input_raster : gdal.Dataset
        Input raster in memory.
    pixel_size : float
        Desired pixel size.

    Returns:
    -------
    gdal.Dataset
        Resampled raster in memory.
    """
    try:
        logger.info(f"Resampling raster to pixel size: {pixel_size}")
        resampled = gdal.Warp('', input_raster, format="MEM",
                              xRes=pixel_size, yRes=pixel_size, resampleAlg="bilinear")
        if resampled is None:
            log_error_and_raise("Failed to resample raster.")
        return resampled
    except Exception as e:
        log_error_and_raise(f"Error during resampling: {e}")


def apply_mask(input_raster, mask_raster_path, nodata_value):
    """
    Apply a raster mask to a raster, setting non-forest areas to the NoData value.

    Parameters:
    ----------
    input_raster : gdal.Dataset
        Input raster in memory.
    mask_raster_path : str
        Path to the mask raster file.
    nodata_value : int or float
        Value for NoData areas (non-forest and invalid data).

    Returns:
    -------
    gdal.Dataset
        Masked raster in memory.

    Raises:
    ------
    RuntimeError
        If the masking process fails.
    """
    try:
        logger.info("Applying mask to raster...")

        # Open the mask raster
        mask_ds = gdal.Open(mask_raster_path)
        if mask_ds is None:
            log_error_and_raise(f"Cannot open mask raster: {mask_raster_path}")

        # Ensure the input raster and mask raster have the same dimensions
        if (input_raster.RasterXSize != mask_ds.RasterXSize) or (input_raster.RasterYSize != mask_ds.RasterYSize):
            log_error_and_raise(
                "Input raster and mask raster have different dimensions.")

        # Read the input raster and mask raster as arrays
        input_band = input_raster.GetRasterBand(1)
        input_data = input_band.ReadAsArray()

        mask_band = mask_ds.GetRasterBand(1)
        mask_data = mask_band.ReadAsArray()

        # Apply the mask: set non-forest areas to nodata_value
        input_data = np.where(mask_data == 1, input_data, nodata_value)

        # Create a new in-memory raster to hold the masked data
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', input_raster.RasterXSize,
                               input_raster.RasterYSize, 1, input_band.DataType)
        out_ds.SetGeoTransform(input_raster.GetGeoTransform())
        out_ds.SetProjection(input_raster.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(input_data)
        out_band.SetNoDataValue(nodata_value)

        # Release datasets
        mask_ds = None
        input_raster = None

        return out_ds

    except Exception as e:
        log_error_and_raise(f"Error during masking: {e}")


def calculate_ndvi_from_processed_bands(red_raster, nir_raster, nodata_value=-9999):
    """
    Calculate NDVI from already processed red and NIR rasters.

    Parameters:
    ----------
    red_raster : gdal.Dataset
        GDAL dataset for the processed red band.
    nir_raster : gdal.Dataset
        GDAL dataset for the processed NIR band.
    nodata_value : float, optional
        NoData value to set in the output raster (default: -9999).

    Returns:
    -------
    gdal.Dataset
        GDAL in-memory dataset containing the NDVI values.

    Raises:
    ------
    RuntimeError
        If any step in the process fails.
    """
    try:
        logger.info("Calculating NDVI from processed bands...")

        # Ensure the rasters have the same dimensions
        if (red_raster.RasterXSize != nir_raster.RasterXSize) or (red_raster.RasterYSize != nir_raster.RasterYSize):
            log_error_and_raise(
                "Red and NIR rasters have different dimensions.")

        # Read the data as arrays
        red_band = red_raster.GetRasterBand(1)
        nir_band = nir_raster.GetRasterBand(1)
        red_data = red_band.ReadAsArray().astype('float32')
        nir_data = nir_band.ReadAsArray().astype('float32')

        # Get NoData values
        red_nodata = red_band.GetNoDataValue()
        nir_nodata = nir_band.GetNoDataValue()

        # Create a mask for NoData values
        mask = (red_data == red_nodata) | (nir_data == nir_nodata)

        # Suppress division warnings
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate NDVI
        ndvi = (nir_data - red_data) / (nir_data + red_data)
        ndvi = np.where(mask, nodata_value, ndvi)

        # Create an in-memory raster for NDVI
        driver = gdal.GetDriverByName('MEM')
        ndvi_raster = driver.Create(
            '', red_raster.RasterXSize, red_raster.RasterYSize, 1, gdal.GDT_Float32)
        ndvi_raster.SetGeoTransform(red_raster.GetGeoTransform())
        ndvi_raster.SetProjection(red_raster.GetProjection())
        ndvi_band = ndvi_raster.GetRasterBand(1)
        ndvi_band.WriteArray(ndvi)
        ndvi_band.SetNoDataValue(nodata_value)

        # Release datasets
        red_raster = None
        nir_raster = None

        return ndvi_raster

    except Exception as e:
        log_error_and_raise(f"Error during NDVI calculation: {e}")


def merge_rasters(raster_list, output_path, band_names, pixel_type, compression="LZW", nodata_value=None):
    """
    Merge multiple rasters into a single multiband raster.

    Parameters:
    ----------
    raster_list : list
        List of GDAL datasets to merge.
    output_path : str
        Path to save the final multiband raster.
    band_names : list, optional
        List of band names corresponding to the rasters.
    pixel_type : str
        Data type of the output raster (e.g., "UInt16", "Float32").
    compression : str
        Compression type for the output raster (default: "LZW").
    nodata_value : int or float, optional
        NoData value to set in the output raster bands.

    Returns:
    -------
    None
    """
    try:
        logger.info("Merging rasters into a single multiband raster...")

        # Build VRT (Virtual Dataset)
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg="nearest", addAlpha=False, separate=True)
        vrt = gdal.BuildVRT('', raster_list, options=vrt_options)
        if vrt is None:
            log_error_and_raise("Failed to build VRT for merging rasters.")

        # Set band descriptions and NoData values if provided
        for i in range(vrt.RasterCount):
            band = vrt.GetRasterBand(i + 1)
            if band_names and i < len(band_names):
                band.SetDescription(band_names[i])
            if nodata_value is not None:
                band.SetNoDataValue(nodata_value)

        # Prepare creation options
        creation_options = [f"COMPRESS={compression}", "BIGTIFF=YES"]

        # Translate VRT to GeoTIFF
        gdal.Translate(output_path, vrt, format="GTiff",
                       creationOptions=creation_options,
                       outputType=getattr(gdal, f"GDT_{pixel_type}"))

        # Release datasets
        vrt = None
        for ds in raster_list:
            ds = None

        logger.info(f"Multiband raster saved to: {output_path}")

    except Exception as e:
        log_error_and_raise(f"Error during merging rasters: {e}")


# ================================= #
# === FILE MANAGEMENT FUNCTIONS === #
# ================================= #


def find_raster_bands(folder_path, band_prefixes):
    """
    Find and filter raster bands based on prefixes.

    Parameters:
    ----------
    folder_path : str
        Path to the folder containing raster files.
    band_prefixes : list
        List of band prefixes to filter (e.g., ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B8"]).

    Returns:
    -------
    list
        List of paths to the filtered raster files ending with '.tif' and matching the band prefixes.
    """
    raster_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with '.tif'
            if file.endswith('.tif'):
                # Check if the file contains any of the specified band prefixes
                if any(prefix in file for prefix in band_prefixes):
                    raster_files.append(os.path.join(root, file))
    return raster_files
