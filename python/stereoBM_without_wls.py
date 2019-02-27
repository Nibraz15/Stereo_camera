import numpy as np
import cv2

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 1

calibration = np.load("outputFile.npz", allow_pickle=False)
imageSize = calibration["imageSize"]
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
Q = calibration["disparityToDepthMap"]

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)

left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

window_size = 3
left_matcher = cv2.StereoBM_create(16, 15)

while(True):
    if not left.grab() or not right.grab():
        print("No more frames")
        break

    _, fixedLeft = left.retrieve()   
    _, fixedRight = right.retrieve()

    #fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    #fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    displ = left_matcher.compute(grayLeft, grayRight)  #.astype(np.float32)/16
    displ = np.int16(displ)
    
    filteredImg = cv2.normalize(src=displ, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)
    
    cv2.imshow('leftFrame', fixedLeft)
    cv2.imshow('rightFrame', fixedRight)
    cv2.imshow('Disparity Map', filteredImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()
