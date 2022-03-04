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
from PIL import Image
import cv2

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


def annotation2mask(data_dir):
    i = 0
    for entry in os.scandir(data_dir):
        if os.path.isdir(entry.path):
            round_dir = entry.path
            image_dir = os.path.join(round_dir, 'image results')
            annotation_dir = os.path.join(round_dir, 'annotation results')
            masks_dir = os.path.join(round_dir, 'segmentation masks')
            Path(masks_dir).mkdir(exist_ok=True)
            annotation = None
            for file in os.listdir(image_dir):
                imgName = os.fsdecode(file)
                if imgName.endswith(".jpg"):
                    try:
                        annotation = geojson.load(open(os.path.join(annotation_dir, Path(imgName).stem + '.txt')))
                    except NameError:
                        print('can not find matching annotation for ' + imgName)
                    raster = rasterio.open(os.path.join(image_dir, imgName))
                    for d in annotation:
                        d['geometry'] = shape(d['geometry'])
                    gdf = gpd.GeoDataFrame(annotation).set_geometry('geometry')
                    # assuming index 0 is always the bounding ROI box so no need to convert the first one

                    ROI_box, ROI_mask, cartilage_mask, humerus_mask = None, None, None, None

                    for index, row in gdf.iterrows():
                        if index != 0 and row.loc['geometry'].geom_type == 'Polygon':
                            polygon = wkt.loads(str(row.loc['geometry']))
                            row.loc['geometry'] = MultiPolygon([polygon])
                        if row.loc['properties']['name'] == 'Anterior ROI':
                            x, y = row.loc['geometry'].exterior.coords.xy
                            # xBoxCoor = [int(x[0]), int(x[2])]
                            # yBoxCoor = [int(y[0]), int(y[2])]
                            # x0, x1, y0, y1
                            ROI_box = [int(x[0]), int(x[2]), int(y[0]), int(y[2])]
                            polygon = wkt.loads(str(row.loc['geometry']))
                            row.loc['geometry'] = MultiPolygon([polygon])
                            ROI_mask, _ = rasterio.mask.mask(raster, row.loc['geometry'])
                        elif row.loc['properties']['name'] == 'Anterior Articular Cartilage':
                            cartilage_mask, _ = rasterio.mask.mask(raster, row.loc['geometry'])
                        elif row.loc['properties']['name'] == 'Anterior Humerus':
                            humerus_mask, _ = rasterio.mask.mask(raster, row.loc['geometry'])
                        else:
                            raise NameError('annotation name not defined')

                    assert ROI_box is not None and cartilage_mask is not None and humerus_mask is not None

                    mask_h5 = h5py.File(os.path.join(masks_dir, Path(file).stem) + ".h5", 'w')
                    mask_h5.create_dataset('Anterior ROI', data=ROI_box)
                    mask_h5.create_dataset('Anterior ROI mask', data=ROI_mask[0, :, :], compression="gzip")
                    mask_h5.create_dataset('Anterior Articular Cartilage', data=cartilage_mask[0, :, :],
                                           compression="gzip")
                    mask_h5.create_dataset('Anterior Humerus', data=humerus_mask[0, :, :], compression="gzip")
                    mask_h5.close()
                    quit()
                    print("finished: " + str(i))
                    i = i + 1


def visualize_mask_outline(data_dir, img_name):
    image = skimage.io.imread(os.path.join(data_dir, 'image results', img_name + '.jpg'))
    masks_pth = os.path.join(data_dir, 'segmentation masks', img_name + '.h5')
    mask_h5 = h5py.File(masks_pth, 'r')
    ROI_box = mask_h5['Anterior ROI']
    cartilage_mask = np.uint8(mask_h5['Anterior Articular Cartilage'])
    humerus_mask = np.uint8(mask_h5['Anterior Humerus'])
    cartilage_mask[cartilage_mask > 0] = 255
    humerus_mask[humerus_mask > 0] = 255
    image = cv2.rectangle(image, (ROI_box[0], ROI_box[2]), (ROI_box[1], ROI_box[3]), (255, 0, 0), 30)
    contours, hierarchy = cv2.findContours(cartilage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 25)
    contours, hierarchy = cv2.findContours(humerus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 15)
    plt.imshow(image)
    plt.show()
    cv2.imwrite(img_name + '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def combineMask(masks_pth, mode):
    mask_h5 = h5py.File(masks_pth, 'r')
    ROI_mask = np.uint8(mask_h5['Anterior ROI mask'])
    cartilage_mask = np.uint8(mask_h5['Anterior Articular Cartilage'])
    humerus_mask = np.uint8(mask_h5['Anterior Humerus'])
    if mode == 'cartilage':
        mask = cv2.bitwise_and(ROI_mask, humerus_mask)
    elif mode == 'tissue':
        mask = cartilage_mask
    else:
        raise NotImplementedError('either set to cartilage(entire ROI) or tissue(tiles)')
    return mask


def showMask(img_pth, mask):
    image = skimage.io.imread(img_pth)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 25)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    mask_in = combineMask(
        '/Users/yuanyanpeng1/Desktop/elbowML/elbow_data/combined/segmentation masks/[#047] 2020-03-01 14.18.05.ndpi.h5',
        mode='cartilage')
    showMask(
        '/Users/yuanyanpeng1/Desktop/elbowML/elbow_data/combined/image results/[#047] 2020-03-01 14.18.05.ndpi.jpg',
        mask_in)
    # preProcessing('/export/project/y.yanpeng/elbow_data', '/export/project/y.yanpeng/h5_data', tileSize=200,
    #               fullImg=False
    # annotation2mask('/Users/yuanyanpeng1/Desktop/elbowML/elbow_data')
    # visualize_mask_outline('/Users/yuanyanpeng1/Desktop/elbowML/elbow_data/combined', '[#047] 2020-03-01 14.18.05.ndpi')
