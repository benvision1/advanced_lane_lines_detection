[//]: # (Image References)


[image1]: ./output_images/test_undist.jpg "Undistorted"
[image2]: ./camera_cal/calibration4.jpg "Chessboard Image"
[image3]: ./output_images/thresholded_binary_image.png "Binary Example"
[image4]: ./output_images/warped_binary_image.png "Warp Example"
[image5]: ./output_images/line_fitting.png "Fit Visual"
[image6]: ./output_images/output_image.png "Output"
[image8]: ./output_images/line_search_windows.png "Search Windows"
[video1]: ./output_images/project_video.mp4 "Video"



# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![Output Image][image6]


### Camera Calibration
The codes for camera calibration are in camera_calibration.py.
This the step I compute the camera intrinsic parameters (camera matrix and distortion coeffients) given a set of chessboard images.
![Example of chessboard][image2]


### Distortion Correction.

I corrected distortion in the image by using the camera parameters obtained from calibration.
![alt text][image1]

#### Create a Thresholded Binary Image

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 38 through 82 in `perspective_transform.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### Apply Perspective Transform
I apply perspective transform to create a bird eye view of the binary image

![alt text][image4]

#### Lane Detection
In this step, I searched for lane pixels in the warped binary image.

!["Search Windows"][image8]

### Lane Fitting
After obtaining lane pixels, I use second order polinomial to fit a lane
!["Lane Fitting"][image5]

#### Curvature Estimation
Lane curvature calculation code can be found in `lines.py` in a function called `update_curvature`
after, I obtained individual lane curvature, I calculated the mean of the two lanes curvature to obtain the estimated lane curvature.

#### Vehicle Position Estimation
Vehicle position estimation calculation code can be found in the `main.py` ( from line # 96 to 105)

---
Here's a [link to my video result](./output_images/project_video.mp4)

----


### Discussion

#### 1. Parameter Tuning

It takes a little tweeking to get right params for the project.
I had to experiment with a lot colors and gradients thresholds to obtain filtered binary image

I had to experiment a lot lane search parameters.

#### 2. Handling Bad Frames and Bad Curvature

I checked to make sure I obain raisonable amount of pixels from each lane for that fram to be considered a good frame.

I also make sure that the covarience of the lane curvature is not too high.