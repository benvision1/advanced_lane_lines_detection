import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "configs/camera_params.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def undist_img(img, mtx, dist):
    # Undistort using mtx and dist
    return cv2.undistort(img, mtx, dist, None, mtx)

def apply_perspective_transform(img):
    """
    This function takes in a binary threshold image and retuned a warped bird eyes view of the image
    """
    img_size = (img.shape[1], img.shape[0])

    # # source points
    src = np.float32([[699,462],[1084,img_size[1]],[227,img_size[1]],[585,462]])

    # 4 destination points to transfer
    offset = 400# offset for dst points
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped, M

def create_thresholded_image(img, color_thresh=(170, 255), sobelx_thresh=(50, 100)):
    """
    Apply a combination of color and gradient thresholds and
    return a binary filtered image
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sobelx_thresh[0]) & (scaled_sobelx <= sobelx_thresh[1])] = 1

    # Sobel y
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in y
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from vertical
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sobelx_thresh[0]) & (scaled_sobely <= sobelx_thresh[1])] = 1

    ## combine xy gradients
    sxybinary = np.zeros_like(sxbinary)
    sxybinary[(sxbinary == 1) | (sybinary == 1)]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two  color and gradient binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxybinary == 1)] = 1

    return combined_binary

def project_into_the_road(img, warped, right_fitx, left_fitx, curvature, distance_from_center, M):
    """
    Draw the line back into the original image space
    """
    img_size = (img.shape[1], img.shape[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img_size[1] - 1, img_size[1])

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    cv2.putText(img, "Radius of Curvature = {} m".format(curvature), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255,0,0],thickness=2 )

    if distance_from_center == 0:
        cv2.putText(img, "Vehicle is at center of the lane".format(curvature), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255,0,0],thickness=2 )
    if distance_from_center > 0:
        cv2.putText(img, "Vehicle is {} m right of lane center".format(abs(distance_from_center)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255,0,0],thickness=2 )
    if distance_from_center < 0:
        cv2.putText(img, "Vehicle is {} m left of lane center".format(abs(distance_from_center)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255,0,0],thickness=2 )

    # Warp the blank back to original image space using inverse matrix
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
if __name__ == "__main__":
    pass
