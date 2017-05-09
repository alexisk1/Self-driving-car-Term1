import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
import glob 

dist_pickle = pickle.load( open("./svcs_lin_YCRCB_all_32.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["c_space"]
hog_channel = dist_pickle["hog_channel"]


test_images = glob.glob('./test_images/*')

images = []
titles = []
ovelap = 0.5

for img_src in test_images:
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    windows =slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 700], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(img, windows, svc, X_scaler, color_space='YCrCb', 
                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block= cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)
    window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255),thick=6)
    images.append (window_img)
    titles.append('')

print (len(images))
fig = plt.figure(figsize=(30,40),dpi=100)
visualize(fig,6,2,images,titles)

for img in images:
    plt.imshow(img)
    plt.show()
