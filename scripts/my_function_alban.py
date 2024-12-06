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
