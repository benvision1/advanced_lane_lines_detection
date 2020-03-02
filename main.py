import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from perspective_transforms import apply_perspective_transform, region_of_interest, create_thresholded_image, project_into_the_road, undist_img
from lane_search_tools import find_lane_pixels
from lines import Line


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "configs/camera_params.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# project Videos
project_video = 'project_video.mp4'

# project_video = 'challenge_video.mp4'

# video output path
video_output_path = 'output_images/project_video.mp4'

# Declare lines
right_line = Line()
left_line = Line()

# create a video capture
video_capture = cv2.VideoCapture(project_video)

# define a video writer
video_writer = cv2.VideoWriter(video_output_path, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=30, frameSize= (1280, 720))

## Parameter settings
# Thresholding params
color_thresh=(170,255)
sobel_thresh=(50,100)
# Lane search params
nwindows=10
margin=100
minpix=10

while(True):

    # reading from frame
    ret,img = video_capture.read()

    if ret:

        right_line.img_size = (img.shape[1], img.shape[0])
        left_line.img_size = (img.shape[1], img.shape[0])

        # Undistort camera image
        undist = undist_img(img, mtx, dist)

         # create a binary thresholded image
        binary_img = create_thresholded_image(undist, color_thresh=color_thresh, sobelx_thresh=sobel_thresh)

        # apply perspective transform to get warped image
        warped_img, M = apply_perspective_transform (binary_img)

        # find lane pixels
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_img, nwindows=nwindows, margin=margin, minpix=minpix, visualuze=False)

        # plt.imshow(out_img)
        # plt.show()
        if len(leftx) < 10:
            left_line.detected = False
        else:
            left_line.detected = True

        if len(rightx) < 10:
            right_line.detected = False
        else:
            right_line.detected = True

        if right_line.detected == True and left_line.detected == True:

            # fit polynomial
            left_line.fit_polynomial(leftx, lefty)
            right_line.fit_polynomial(rightx, righty)

            if (right_line.covariance[0]) < 1e-3 and (left_line.covariance[0]) < 1e-3:
                # get lane estimated lane curvature
                curvature = np.mean([right_line.curvature_rad, left_line.curvature_rad])

                # get best fit
                right_fitx = right_line.best_line_fit_x
                left_fitx = left_line.best_line_fit_x

                # vehicle position
                vehicle_pos = (img.shape[1] / 2 , (img.shape[0] + img.shape[0]) / 2)

                # get lane center pos
                right_lane_base_x = right_line.base_x
                left_lane_base_x = left_line.base_x

                lane_center_pos = ((left_lane_base_x + right_lane_base_x) / 2 ,  (img.shape[0] + img.shape[0]) / 2 )

                # get distance from lane center
                vehicle_distance_from_lane_center =( vehicle_pos[0] - lane_center_pos[0]) * 3.7/700
                # result = draw_line(warped_img, left_fitx, right_fitx)
                result = project_into_the_road(img, warped_img, right_fitx, left_fitx, np.round(curvature, 2), np.round(vehicle_distance_from_lane_center, 2), M)

                video_writer.write(result)
    else:
        video_writer.release()
        break