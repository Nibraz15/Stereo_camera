import numpy as np
import cv2

LEFT_PATH = "capture/left/{:01d}.jpg"
RIGHT_PATH = "capture/right/{:01d}.jpg"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CHESSBOARD_SIZE = (7, 7)

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)

left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

frameId = 0

while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    grayLeft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

    hasLeft, cornerLeft = cv2.findChessboardCorners(grayLeft, CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
    hasRight, cornerRight = cv2.findChessboardCorners(grayRight, CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
    
    if hasLeft and hasRight:
        cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
        cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)

        cv2.drawChessboardCorners(leftFrame, CHESSBOARD_SIZE, cornerLeft, hasLeft)
        cv2.drawChessboardCorners(rightFrame, CHESSBOARD_SIZE, cornerRight, hasRight)
        
        cv2.imshow('left', leftFrame)
        cv2.imshow('right', rightFrame)

        frameId += 1
        print "Captured image : ", frameId
        cv2.waitKey(1)

left.release()
right.release()
cv2.destroyAllWindows()
