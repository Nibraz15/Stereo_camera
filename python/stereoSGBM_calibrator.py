import numpy as np
import cv2
import cvui

WINDOW_NAME	= 'Trackbar'

blockSize = [11]
numDisparities = [16]
P1 = [0]
P2 = [0]
disp12MaxDiff = [0]
preFilterCap = [0]
uniquessRatio = [0]
speckleRange = [0]
speckleWindow = [0]

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

lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
	
frame = np.zeros((700, 700, 3), np.uint8)
cvui.init(WINDOW_NAME)

while (True):
    if not left.grab() or not right.grab():
        print("No more frames")
        break

    _, fixedLeft = left.retrieve()   
    _, fixedRight = right.retrieve()

    #fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    #fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)
    
    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    left_matcher = cv2.StereoSGBM_create(numDisparities = int(numDisparities[0]),
                                       blockSize = int(blockSize[0]),
                                       P1 = int(P1[0]),
                                       P2 = int(P2[0]),
                                       disp12MaxDiff = int(disp12MaxDiff[0]),
                                       preFilterCap = int(preFilterCap[0]),
                                       uniquenessRatio = int(uniquessRatio[0]) ,
                                       speckleWindowSize = int(speckleWindow[0]),
                                       speckleRange = int(speckleRange[0]))
                                       
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(grayLeft, grayRight)  #.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = right_matcher.compute(grayRight, grayLeft) # .astype(np.float32)/16
    dispr = np.int16(dispr)
    
    filteredImg = wls_filter.filter(displ, fixedLeft, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)
    
    # Fill the frame with a nice color
    frame[:] = (49, 52, 49)

    cvui.text(frame, 50, 20, 'Block Size: ')
    cvui.trackbar(frame, 25, 40, 600, blockSize, 5, 101, 2, '%.1Lf', cvui.TRACKBAR_DISCRETE, 2)

    cvui.text(frame, 50, 90, 'numDisparites from -100 to ')
    cvui.trackbar(frame, 25, 110, 600, numDisparities, 16, 96, 16, '%.1Lf', cvui.TRACKBAR_DISCRETE, 16)

    cvui.text(frame, 50, 160, 'P1: ')
    cvui.trackbar(frame, 25, 180, 600, P1, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 230, 'P2 (should be larger than p2: ')
    cvui.trackbar(frame, 25, 250, 600, P2, 0, 3000, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 300, 'disp12MaxDiff: ')
    cvui.trackbar(frame, 25, 320, 600, disp12MaxDiff, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 370, 'preFilterCap: ')
    cvui.trackbar(frame, 25, 390, 600, preFilterCap, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 440, 'Uniqueness Ratio: ')
    cvui.trackbar(frame, 25, 460, 600, uniquessRatio, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 510, 'speckleRange: ')
    cvui.trackbar(frame, 25, 530, 600, speckleRange, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)

    cvui.text(frame, 50, 580, 'speckleWindow: ')
    cvui.trackbar(frame, 25, 600, 600, speckleWindow, 0, 100, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 1)
    cvui.update()

    # Show everything on the screen
    cv2.imshow(WINDOW_NAME, frame)
    cv2.imshow("Disparity", filteredImg)
    # Check if ESC key was pressed

    if cv2.waitKey(100) == 27:
        break
