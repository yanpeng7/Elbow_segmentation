import os
from itertools import count
from pathlib import Path
import skimage.io
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import shape
import PIL.Image
import os
import geojson
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
import h5py

PIL.Image.MAX_IMAGE_PIXELS = 1000000000


def generateMask(raster, image, annotation, fullImg=False):
    # convert to geoPandas frame
    for d in annotation:
        d['geometry'] = shape(d['geometry'])
    gdf = gpd.GeoDataFrame(annotation).set_geometry('geometry')
    # check convert to MultiPolygon if Polygon
    trace = gdf.iloc[1].loc['geometry']
    if trace.geom_type == 'Polygon':
        polygon = wkt.loads(str(trace))
        trace = MultiPolygon([polygon])
    # convert trace to binary mask
    out_mask, out_transform = rasterio.mask.mask(raster, trace)
    out_mask = out_mask[0]
    out_mask[out_mask > 1] = 1
    if fullImg == True:
        return image, out_mask
    else:
        box = gdf.iloc[0].loc['geometry']
        x, y = box.exterior.coords.xy
        xBoxCoor = [int(x[0]), int(x[2])]
        yBoxCoor = [int(y[0]), int(y[2])]
        croppedImg = image[yBoxCoor[0]:yBoxCoor[1], xBoxCoor[0]:xBoxCoor[1]]
        croppedMask = out_mask[yBoxCoor[0]:yBoxCoor[1], xBoxCoor[0]:xBoxCoor[1]]
        return croppedImg, croppedMask


def generateTiles(img, mask, tileSize=200):
    mask = np.expand_dims(mask, axis=-1)
    imgTiles = [img[x:x + tileSize, y:y + tileSize] for x in range(0, img.shape[0] - tileSize, tileSize) for y in
                range(0, img.shape[1] - tileSize, tileSize)]
    imgStack = np.stack(imgTiles, axis=0)  # return shape (B,C,tileSize,tileSize)
    maskTiles = [mask[x:x + tileSize, y:y + tileSize] for x in range(0, img.shape[0] - tileSize, tileSize) for y in
                 range(0, img.shape[1] - tileSize, tileSize)]
    maskStack = np.stack(maskTiles, axis=0)
    return imgStack, maskStack


def generateH5(target_dir, imgStack, maskStack, oldName, newName):
    dset = h5py.File(target_dir + '/' + newName + ".h5", 'w')
    dset.create_dataset('images', data=imgStack, compression="gzip")
    dset.create_dataset('masks', data=maskStack, compression="gzip")
    dset.attrs['oldName'] = oldName
    dset.close()


def preProcessing(data_dir, destination_dir, tileSize=200, fullImg=False):
    numberedFileName = ("elbow_%03i" % i for i in count(1))
    num = 0
    for entry in os.scandir(data_dir):
        if os.path.isdir(entry.path):
            round_dir = entry.path
            image_dir = os.path.join(round_dir, 'image results')
            annotation_dir = os.path.join(round_dir, 'annotation results')
            for file in os.listdir(image_dir):
                imgName = os.fsdecode(file)
                if imgName.endswith(".jpg"):
                    try:
                        annotation = geojson.load(open(os.path.join(annotation_dir, Path(imgName).stem + '.txt')))
                    except NameError:
                        print('can not find matching annotation for ' + imgName)
                    raster = rasterio.open(os.path.join(image_dir, imgName))
                    image = skimage.io.imread(os.path.join(image_dir, imgName))
                    img, mask = generateMask(raster, image, annotation, fullImg=fullImg)
                    imgStack, maskStack = generateTiles(img, mask, tileSize=tileSize)
                    generateH5(destination_dir, imgStack, maskStack, Path(imgName).stem, next(numberedFileName))
                    print('finished ' + str(num))
                    num = num + 1


if __name__ == "__main__":
    preProcessing('/export/project/y.yanpeng/elbow_data', '/export/project/y.yanpeng/h5_data', tileSize=200,
                  fullImg=False)
