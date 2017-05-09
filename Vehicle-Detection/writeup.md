##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---
#Writeup / README



###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 8 through 25 of the file called `lesson_functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Based on the initial study on HOGs orientations can improve classification up to 9 that is why I used 8.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

I created `try_features.py` to produce the above images and experiment with HOG features.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but my main intuintion on the orientation was based on the initial study on HOGs orientations can improve classification up to 9 that is why I used 8 for orientation.
`pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` were chosen based on observations on the output images  with the  `try_features.py`. Using the same script I used decised to use HOG from all channels. Basicly, my main technique was to examine the output hog feature images and see that t differed the most from the non car feature images.

####3. Classifier - Training and features

I trained a linear SVM using LinearSVC classifier from sklearn.svm library. Although, I tried non-linear SVMs and played with the C and gamma parameter it seemed that a linear SVM was good enough to produce a nice result and generally trainging a non-linear SVM takes to much time on my laptop. Therefore parameterization of non linear SVMs would take to much time to compete the already could results I got from the Linear SVM.

I played around with different color spaces, started with RGB but it produced many false positives. Then HSV was quite good but found YCbCr to work better on the video. I also tried different parameters for the spatial size mainly 16 and 32 , but 32 was better on the video.

In total, I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I defined `slide_window()` function in `lesson_functions.py` in lines 102 through 141 that produced the windows to focus at. Then with I defined `search_windows()`  function in `lesson_functions.py` in lines 210 through 237 to extract features and locate cars.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I defined `find_cars` function in `lesson_functions.py` in lines 265 through 361 and used downsampling(resizing) for the image. Then I extracted the hog features for all the images and windowed the images by subsampling the hog features per window. 

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

I used scale 1-2 which can be translated for windon sizes of 64x64 and 128x128. I didn't use 96x96 because it was producing more false positive results. 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video for windows of 64x64 and 128x128.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

Moreover, I kept the last 5 frames and summed them up in order to filter frames that didn't produce any detections. This is done in video.py in lines 33 through 38.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overlaping vehicles is one of the main things that the video is going to fail. Tracking the previous detections and predicting with a filter the expected position.
Possibly to fix this I should track the current detection, register  and predict the position of vehicles.



