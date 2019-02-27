import numpy as np
import cv2

blockSize = 11
numDisparities = 16
uniquessRatio = 0
speckleRange = 0
speckleWindow = 0

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

left_matcher = cv2.StereoBM_create(numDisparities, blockSize)

left_matcher.setNumDisparities(numDisparities)
left_matcher.setBlockSize(blockSize)
left_matcher.setUniquenessRatio(uniquessRatio)
left_matcher.setSpeckleRange(speckleRange)
left_matcher.setSpeckleWindowSize(speckleWindow)
    

def clusterRate(disparityMap, threshold):
      mask = disparityMap>threshold
      value = np.count_nonzero(disparityMap[mask])
      return float(value/disparityMap.size)

def averageDisparity(disparityMap):
      return np.average(disparityMap)
    

def decision(disparityMap, rate, threshold):
      rows, cols = disparityMap.shape
      mapLeft = disparityMap[:, :int(cols*0.3)]
      mapMiddle = disparityMap[:, int(cols*0.3):int(cols*0.7)]
      mapRight = disparityMap[:, int(cols*0.7):int(cols*1)]
      
      avgLeft = averageDisparity(mapLeft)
      rateMiddle = clusterRate(mapMiddle, threshold)
      avgRight = averageDisparity(mapRight)
      
      if (rateMiddle<=rate):
            print "move forward"
      elif (avgLeft>avgRight):
            print "turn Right"
      else:
            print "turn Left"

            
while(True):
    if not left.grab() or not right.grab():
        print("No more frames")
        break

    _, leftFrame = left.retrieve()   
    _, rightFrame = right.retrieve()

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LANCZOS4)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LANCZOS4)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    displ = left_matcher.compute(grayLeft, grayRight)  #.astype(np.float32)/16
    displ = np.int16(displ)
    
    filteredImg = cv2.normalize(src=displ, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    decision(filteredImg, 0.2, 120)
    
    filteredImg=cv2.applyColorMap(filteredImg, cv2.COLORMAP_BONE)
                    
    
    cv2.imshow('leftFrame', fixedLeft)
    cv2.imshow('rightFrame', fixedRight)
    cv2.imshow('Disparity Map', filteredImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()
