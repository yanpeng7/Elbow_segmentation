import h5py
import numpy as np
import skimage.io
import os
import cv2
from matplotlib import pyplot as plt
import PIL.Image
import rasterio

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
import geojson
from shapely.geometry import shape
import geopandas as gpd

imageName = '[#013] 2020-03-01 13.12.44.ndpi'
imageDir = './elbow_data/combined/image results'
maskDir = './elbow_data/combined/segmentation masks'
annoDir = './elbow_data/combined/annotation results'
image = skimage.io.imread(os.path.join(imageDir, imageName + '.jpg'))
mask = skimage.io.imread(os.path.join(maskDir, imageName + '.jpg'))
raster = rasterio.open(os.path.join(imageDir, imageName + '.jpg'))
annotation = geojson.load(open(os.path.join(annoDir, imageName + '.txt')))
# --------------------
for d in annotation:
    d['geometry'] = shape(d['geometry'])
gdf = gpd.GeoDataFrame(annotation).set_geometry('geometry')
box = gdf.iloc[0].loc['geometry']
x, y = box.exterior.coords.xy

xBoxCoor = [int(x[0]), int(x[2])]
yBoxCoor = [int(y[0]), int(y[2])]
# --------------------
mask[mask > 0] = 255
H, W = mask.shape
redImg = np.zeros((H, W, 3), image.dtype)
redImg[:, :] = (255, 0, 0)
print('done half')

# redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
# dst = cv2.addWeighted(image, 0.8, redMask, 1, 0)

image = cv2.rectangle(image, (xBoxCoor[0], yBoxCoor[0]), (xBoxCoor[1], yBoxCoor[1]), (255, 0, 0), 30)
print(mask.shape)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dst = cv2.drawContours(image, contours, -1, (0, 0, 255), 15)

plt.imshow(dst)
plt.show()
cv2.imwrite('image_plus_mask_5.jpg', cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
