##Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Camera Calibration

The code for this step is contained in the file called `camera_calibrate.py`).  

I start reading with glob all the filenames. Then I start preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![][undistorted.png]

Lastly, I save the params in a pickle file so they can be reused later on.

###Pipeline (single images)

#1. Distortion-correction
To test the distortion correction algorithm I used the test6.jpg image which has the a big tree covering the top left corner.
Here is the original test image 
![][test6.jpg]
Here is the undistorted image 
![][undistorted-test6.png]

#2. Color transforms, Magnitude and Direction of the gradient 
I used a combination of color, Magnitude and Direction of the gradient thresholds to generate a binary image (thresholding steps at lines # through # in `pipeline.py`). In addition I use a color space conversion. The `pipe()` function in the `pipeline.py` is responsible for all afore mentioned steps.
The steps within `pipe()` function defined in lines 55-100 in the `pipeline.py` are :
1) Undistort the image   -- cv2.undistort() 
2) Color Space conversion to HLV  and HSV -- cv2.cvtColor() and max-min threshold all channels to identify white and yellow lines
3) Magnitude of gradient -- Thresholded and calculated by  `mag_thresh()` function defined in lines 27-40 in the `pipeline.py`
4) Angle of gradient -- Thresholded and calculated by  `dir_thresh()` function defined in lines 42-51 in the `pipeline.py`
5) All previous layers are combined
Here's an example of my output for this step.  (note: source is test6.jpg image)

![][binary_combo.png]

#3. Perspective transform

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 105 through 118 in the file `pipeline.py`.  The `warp_image()` function takes as inputs an image (`img`) and does the transformation given 4 points from source to destination. I chose the hardcode the source and destination points by extending the lanes of test image `straight_lines1.png` and transform them in a rectangular.

```
	src = np.float32([[570, 460], [710, 460],[1030, 680] ,[260, 680]])
	dest = np.float32([[260, 0],[1030, 0],[1030, 720], [260, 720]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 460      | 260, 0        | 
| 710, 460      | 1130, 0       |
| 1030, 680     | 1130, 720     |
| 260, 680      | 260, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dest` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The following output is based on `test2.jpg` file as input.

![][warped.png]

#4. Finding lanes

Then I did a implementation of window search and fit my lane lines with a 2nd order polynomial, the results of fiting the line can be seen in the following picture:

![][polyfit.png]

The code for my sliding window search and polynomial fit is  included in a function called `find_lanes()`, which appears in lines 121 through 250 in the file `pipeline.py`. 

I also implemented drop off frames with no results and exponential smoothing in lines 186 - 217

#5. Curvature and distance of car from the center of the lane

I did this in lines 237 through 248 in my code in `pipeline.py` in the function called `find_lanes()`
I estimated the correction but looking up the length of the lanes on the internet and adjusting to the number of pixels.


#6. Output result

I implemented this step in lines 253 through 279 in my code in `pipeline.py` in the function `draw_result()`.  Here is an example of my result on a test image:

![][result.png]

---

#Pipeline (video)

..
Here's a [link to my video result](./output.mp4)

---
I used `pics.py` script to generate all the above images.

###Discussion

My main strategy was involving playing with the warped space. I started using the fixed points for both source and destination but also expiremented with an  offset. Then I also played with the thresholds and only the angle of the gradients made an improvement
Certainly, expotential filtering  improved a lot the quality of the lines.

Although the current min-max values are good for this video for challenge_video , they fail . More tuning should be done to find better values. Even extra color spaces and a good sanity check that the top left and right extension of the lines do not meet. Maybe, also some thersholding only on some channels could help a lot.

The video fails for the challenge video 1:
[](./cha_output.mp4)
The video also fails for the challenge video 2:
[](./hard_cha_output.mp4)

Sanity check will certainly help both videos but video2 might also require more experimenting with other approximation functions.


