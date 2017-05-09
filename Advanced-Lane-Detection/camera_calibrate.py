import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import pickle

images=glob.glob('./camera_cal/calibration*.jpg')
print(len(images))
objpoints = []
imgpoints = []

objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for fn in images:
  img = mpimg.imread(fn)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

  if(ret==True):
     imgpoints.append(corners)
     objpoints.append(objp)
img = cv2.imread(images[1])
print (images[1])
imgsize= (img.shape[0],img.shape[1])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None)



dst = cv2.undistort(img, mtx, dist, None, mtx)

data= {'mtx':mtx,'dist': dist, 'rvecs': rvecs,'tvecs':tvecs}
f = open('./calibration.p', 'wb')
pickle.dump(data,f)


plt.imshow(dst)
plt.show()


