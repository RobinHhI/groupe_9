# Importation des libraires
import sys
import os
from osgeo import gdal
gdal.UseExceptions()
sys.path.append('libsigma')


# Chemins d'accès relatifs
in_vector = "groupe_9/results/data/sample/Sample_BD_foret_T31TCJ.shp"
ref_image = "data/rasters/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0/SENTINEL2A_20220209-105857-811_L2A_T31TCJ_C_V3-0_FRE_B2.tif"
out_image = "groupe_9/results/data/img_pretraitees/mask_forest.tif"


def get_reprojected_raster_properties(input_raster, target_srs):
    """Reprojete un raster en entrée dans le même EPSG qu'un raster de référence."""
    temp_raster = gdal.Warp('', input_raster, format='MEM', dstSRS=target_srs)

    geotransform = temp_raster.GetGeoTransform()
    xmin = geotransform[0]
    ymax = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    xmax = xmin + (temp_raster.RasterXSize * pixel_width)
    ymin = ymax + (temp_raster.RasterYSize * pixel_height)

    spatial_resolution = abs(pixel_width)

    temp_raster = None

    return spatial_resolution, xmin, ymin, xmax, ymax


# Extraire les propriétés du raster
sptial_resolution, xmin, ymin, xmax, ymax = get_reprojected_raster_properties(
    ref_image, "EPSG:2154")

# commande cmd pour le masque avec compression
cmd_pattern = (
    "gdal_rasterize -burn 1 "  # foret = 1
    "-init 0 "
    "-a_nodata 99 "  # definition du NoData
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
    "-co COMPRESS=LZW "
    "{in_vector} {out_image}"
)

# Ajout des propriétés au masque raster
cmd = cmd_pattern.format(
    in_vector=in_vector,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    out_image=out_image,
    sptial_resolution=sptial_resolution
)

# Execution du cmd
os.system(cmd)

print("Masque forêt compressé et enregistré")
