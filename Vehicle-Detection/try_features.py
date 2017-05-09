import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import svm
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


from lesson_functions import *

vehicles_img=glob.glob(r'./vehicles/*/*.png')
non_vehicles_img=glob.glob(r'./non-vehicles/*/*.png')

veh_ind= np.random.randint(0,len(vehicles_img))
non_veh_ind= np.random.randint(0,len(non_vehicles_img))

veh= mpimg.imread(vehicles_img[veh_ind])
non_veh= mpimg.imread(non_vehicles_img[non_veh_ind])

imgs =[veh,non_veh]
titles = ['car', 'Not car']
fig=plt.figure(figsize=(12,3))
visualize(fig,1,2,imgs,titles)
plt.show()

spatial = 32
histbin = 32
hist_bins = 32
spatial_size=(spatial, spatial)
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
t=time.time()
car_features , car_image = single_img_features(veh, color_space=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True,vis=True)

notcar_features, notcar_image = single_img_features(non_veh, color_space=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True,vis=True)

imgs = [veh,car_image, non_veh,notcar_image ]
titles = ['car','Car Hog', 'Not car', 'not car hog']
fig=plt.figure(figsize=(12,3))
visualize(fig,1,4,imgs,titles)
plt.show()


