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
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


from lesson_functions import *

vehicles_img=glob.glob(r'./vehicles/*/*.png')
non_vehicles_img=glob.glob(r'./non-vehicles/*/*.png')



print("vehicles", len(vehicles_img) )
print("non vehicles", len(non_vehicles_img) )

# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 32
histbin = 32
hist_bins = 32
spatial_size=(spatial, spatial)
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
t=time.time()
car_features = extract_features(vehicles_img, color_space=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(non_vehicles_img, color_space=colorspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract  features...')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack( (np.ones(len(car_features)) , np.zeros(len(notcar_features)) ) )


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)



print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = len(X_test)
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')



data= {'svc':svc, 'X_scaler': X_scaler,'orient':orient, 'pix_per_cell':pix_per_cell,'cell_per_block' :cell_per_block,'spatial_size':spatial_size,'hist_bins':hist_bins,'c_space':colorspace,'hog_channel':hog_channel}
f = open('./svcs_lin_YCRCB_all_32.p', 'wb')
pickle.dump(data,f)

