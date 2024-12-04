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
from osgeo import gdal
import geopandas as gpd

# GDAL configuration
gdal.UseExceptions()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("process_log.log")]
)

# =========================== #
# === UTILITY FUNCTIONS === #
# =========================== #


def log_error_and_raise(message, exception=RuntimeError):
    """
    Logs an error message and raises an exception.

    Parameters:
    ----------
    message : str
        Error message to log and raise.
    exception : Exception
        Type of exception to raise (default: RuntimeError).
    """
    logging.error(message)
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

# ============================= #
# === VECTOR FILE FUNCTIONS === #
# ============================= #


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
        logging.info(
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
        logging.info("Creating forest mask...")

        # Step 1: Reproject the reference raster to get the desired properties
        logging.info("Reprojecting reference raster to EPSG:2154...")
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
        logging.info("Rasterizing forest mask...")
        mem_driver = gdal.GetDriverByName('MEM')
        out_raster = mem_driver.Create(
            '', x_pixels, y_pixels, 1, gdal.GDT_Byte)
        out_raster.SetGeoTransform(
            (xmin_aligned, pixel_width, 0, ymax_aligned, 0, -pixel_height))
        out_raster.SetProjection(crs)

        # Initialize raster with value 1 (forest)
        out_band = out_raster.GetRasterBand(1)
        out_band.Fill(1)
        out_band.SetNoDataValue(99)

        # Open the vector mask
        vector_ds = gdal.OpenEx(mask_vector, gdal.OF_VECTOR)
        if vector_ds is None:
            log_error_and_raise(f"Cannot open vector file: {mask_vector}")

        # Rasterize the vector mask onto the raster, burning value 0 (non-forest)
        err = gdal.RasterizeLayer(
            out_raster, [1], vector_ds.GetLayer(), burn_values=[0])
        if err != 0:
            log_error_and_raise("Rasterization failed.")

        # Close the vector dataset
        vector_ds = None

        # Step 3: Clip the rasterized mask to the study area
        logging.info("Clipping the forest mask to the study area...")
        clipped_mask = clip_raster_to_extent(out_raster, clip_vector)

        if clipped_mask is None:
            log_error_and_raise("Failed to clip the forest mask.")

        # Step 4: Save the clipped mask to the output path
        # Before saving, ensure any existing file is deleted
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.info(f"Removed existing file: {output_path}")
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

        logging.info(f"Forest mask saved to: {output_path}")

    except Exception as e:
        log_error_and_raise(f"Error during forest mask creation: {e}")


def clip_raster_to_extent(input_raster, clip_vector):
    """
    Clip a raster to a shapefile's extent.

    Parameters:
    ----------
    input_raster : gdal.Dataset
        Input raster in memory.
    clip_vector : str
        Path to the shapefile for clipping.

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
        logging.info(f"Clipping raster with shapefile: {clip_vector}")
        if not os.path.exists(clip_vector):
            log_error_and_raise(
                f"Shapefile not found: {clip_vector}", FileNotFoundError)

        clipped = gdal.Warp('', input_raster, format="MEM",
                            cutlineDSName=clip_vector, cropToCutline=True)
        if clipped is None:
            log_error_and_raise(
                f"Failed to clip raster with shapefile: {clip_vector}")

        # Release the input raster after clipping
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
        logging.info(f"Resampling raster to pixel size: {pixel_size}")
        resampled = gdal.Warp('', input_raster, format="MEM",
                              xRes=pixel_size, yRes=pixel_size, resampleAlg="bilinear")
        if resampled is None:
            log_error_and_raise("Failed to resample raster.")
        return resampled
    except Exception as e:
        log_error_and_raise(f"Error during resampling: {e}")


def apply_mask(input_raster, mask_raster_path, nodata_value=0):
    """
    Apply a raster mask to a raster.

    Parameters:
    ----------
    input_raster : gdal.Dataset
        Input raster in memory.
    mask_raster_path : str
        Path to the mask raster file.
    nodata_value : int
        Value for masked areas.

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
        logging.info("Applying mask to raster...")

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

        # Apply the mask: set nodata_value where mask_data == 0
        input_data[mask_data == 0] = nodata_value

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


def merge_rasters(raster_list, output_path, band_names=None, pixel_type="UInt16", compression="LZW"):
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
        Data type of the output raster (e.g., "UInt16").
    compression : str
        Compression type for the output raster (default: "LZW").

    Returns:
    -------
    None
    """
    try:
        logging.info("Merging rasters into a single multiband raster...")

        # Build VRT (Virtual Dataset)
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg="bilinear", addAlpha=False, separate=True)
        vrt = gdal.BuildVRT('', raster_list, options=vrt_options)
        if vrt is None:
            log_error_and_raise("Failed to build VRT for merging rasters.")

        # Set band descriptions if band_names are provided
        if band_names:
            for i, band_name in enumerate(band_names):
                band = vrt.GetRasterBand(i + 1)
                band.SetDescription(band_name)

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

        logging.info(f"Multiband raster saved to: {output_path}")

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
    return sorted(raster_files)
