import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from lesson_functions import *
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip

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

ovelap = 0.5
ystart=400
ystop=656
scale = 1
first = 0
l_heapmap=[]
carlist=[]
def process_image(image):
    global l_heapmap
    out_img, heat_map64 =  find_cars(image, ystart, 500, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img, heat_map128 = find_cars(image, ystart, 650, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat_map = heat_map64 +heat_map128
    heatmap = apply_threshold(heat_map, 3)
    l_heapmap.append(heatmap)
    if(len(l_heapmap)>=5):
       l_heapmap.pop(0)
    flt_heatmap=l_heapmap[0]
    for i in range(1,len(l_heapmap)):
          flt_heatmap+=l_heapmap[i]

    labels = label(flt_heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image),labels)

    return draw_img


output ='./output_2.mp4'
clip1 = VideoFileClip("./project_video.mp4").subclip(10,20)
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile(output, audio=False)



