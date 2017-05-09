import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


with open('./calibration.p', 'rb') as f:
    data = pickle.load(f)
img = mpimg.imread("./test_images/test2.jpg")
global first
first=False
mtx=data['mtx']
dist=data['dist']
rvecs=data['rvecs']
tvecs=data['tvecs']

class pipeline:
    def __init__(self):
        self.first=False
        self.l_lefty= np.ndarray(3)
        self.l_righty= np.ndarray(3)
        self.src = np.float32([[560, 470], [710, 470],[1030, 680] ,[260, 680]])

        self.dest = np.float32([[260, 0],[1030, 0],[1030, 720], [260, 720]])

    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
    
        # Apply the followindef abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel= np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255*sobel/np.max(sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        binary_output=sxbinary
  
        #binary_output = np.copy(img) # Remove this line
        return binary_output

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx =np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        sobel= np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(sobel)
        sxbinary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
        return sxbinary
    


    def pipe(self,img, s_thresh=(175, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        img = cv2.undistort(img, mtx, dist, None, mtx)
        kernel = np.ones((5,5),np.uint8)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #HSV Color space
        #Yellow max-min in HSV
        ymin = np.array([0, 100, 200], np.uint8)
        ymax = np.array([50, 255, 255], np.uint8) 
        yellow = cv2.inRange(hsv, ymin, ymax)
        #White max-min in HSV
        wmin = np.array([0, 0, 220], np.uint8)
        wmax = np.array([255, 30, 255], np.uint8)
        white = cv2.inRange(hsv, wmin, wmax)
        #binary output for hsv
        b_color = np.zeros_like(hsv[:, :, 0])
        b_color[( (yellow != 0) |(white != 0) )] = 1
        # Threshold x gradient
        direction = self.dir_threshold(img,  sobel_kernel=5, thresh=(0.7, 1.3))
        #  magnitude
        magnitude = self.mag_thresh(img, sobel_kernel=5, mag_thresh=(50, 200))

        #HLS Color space

        # yellow
        hls_wmin_y = np.array([0, 90, 230], np.uint8)
        hls_wmax_y = np.array([80, 240, 255], np.uint8)  
        #Yellow max-min in HLS
        # white
        hls_wmin = np.array([0, 220, 0], np.uint8)
        hls_wmax = np.array([150, 255, 255], np.uint8)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # Convert to HLS color space
        #Yellow max-min in HLS
        white_hls = cv2.inRange(hls, hls_wmin, hls_wmax)
        #White max-min in HLS
        yellow_hls = cv2.inRange(hls, hls_wmin_y, hls_wmax_y)
        b_hls = np.zeros_like(hls[:, :, 0])
        b_hls[(white_hls != 0)| (yellow_hls!=0) ] = 1

        res = np.zeros_like(b_color)
        #produce result by ORing the color spaces and with the OR of direction and magnitude
        res[  ((b_color==1) | ( b_hls==1))  & ((direction==1)|(magnitude==1)) ] =1 
        #binary_noise_reduction(res,1)
        return res




    def warp_image(self,img):
       img_size = (img.shape[1], img.shape[0])
       img_size = (img.shape[1], img.shape[0])
       src = np.float32([[570, 460], [710, 460],[1030, 680] ,[260, 680]])
       dest = np.float32([[260, 0],[1030, 0],[1030, 720], [260, 720]])

       #print (img_size[0]) #x
       #print (img_size[1]) #y

       # Given src and dest points, calculate the perspective transform matrix
       M = cv2.getPerspectiveTransform(src, dest)
       # Warp the image using OpenCV warpPerspective()
       warped = cv2.warpPerspective(img, M, img_size)
       return warped


    def find_lanes(self,warped):

        # Assuming you have created a warped binary image called "warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        global l_lefty #=[]
        global l_leftx#=[]
        global l_righty#=[]
        global l_rightx#=[]
        # Fit a second order polynomial to each
        if(lefty.shape[0]==0):
            lefty=l_lefty
            leftx=l_leftx
        l_lefty=lefty
        l_leftx=leftx

        if(rightx.shape[0]==0):
            rightx=l_rightx
            righty=l_righty
        l_rightx=rightx
        l_righty=righty
  
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        if(self.first==False):
            self.first=True
            self.l_lefty=left_fit
            self.l_righty=right_fit
        #Here I use exponential smoothing 
        gamma=0.7
        self.l_lefty =self.l_lefty.astype(float)
        self.l_righty = self.l_righty.astype(float)
        left_fit =gamma*left_fit +(1-gamma)*self.l_lefty
        self.l_lefty=left_fit 
        right_fit=gamma*right_fit+(1-gamma)*self.l_righty
        self.l_righty=right_fit

         
        # Generate x and y values for plottingx
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        y_eval =  np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        #print(left_curverad, right_curverad)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        left_d=left_fit[0]*warped.shape[1]**2 + left_fit[1]*warped.shape[1] + left_fit[2]
        right_d = right_fit[0]*warped.shape[1]**2 + right_fit[1]*warped.shape[1] + right_fit[2]

        return left_fitx, right_fitx, ploty,left_curverad,right_curverad,((right_d-left_d)-warped.shape[1]/2)*xm_per_pix

    #Here a draw the found lanes
    def draw_result(self,img,dst,left_fitx, right_fitx, ploty ):
        img_size = (img.shape[1], img.shape[0])
        img_size = (img.shape[1], img.shape[0])

        src = np.float32([[570, 460], [710, 460],[1030, 680] ,[260, 680]])
        dest = np.float32([[260, 0],[1030, 0],[1030, 720], [260, 720]])

        Minv = cv2.getPerspectiveTransform(dest, src)


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(dst).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        return result



    def lanes(self,img):
        img =  cv2.undistort(img, mtx, dist, None, mtx)

        dst=self.pipe(img)
        dst=self.warp_image(dst)
        left_fitx, right_fitx, ploty,left_curverad,right_curverad,center =self.find_lanes(dst)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Left_curveture:'+str(left_curverad),(10,40), font, 1,(255,255,255),2)
        cv2.putText(img,'Right_curveture:'+str(right_curverad),(10,80), font, 1,(255,255,255),2)
        cv2.putText(img,'Distance from center:'+str(center/10),(10,120), font, 1,(255,255,255),2)
        result=self.draw_result(img,dst,left_fitx, right_fitx, ploty )
        return result


