#pragma once
#include "opencv2\opencv.hpp"
#include "ximgproc.hpp"

class DepthMapCreator
{
private:
	cv::VideoCapture leftCam, rightCam;
	cv::Mat leftFrame, rightFrame, greyLeft, greyRight, fixedLeft, fixedRight, disparityLeft, disparityRight, filteredDisparityMap, newLeftCameraMat, newRightCameraMat,
		leftCameraMat, leftDistroCoeff, rightCameraMat, rightDistroCoeff, Q, leftMapX, leftMapY, rightMapX, rightMapY;
	cv::Size im_size;
	cv::Ptr <cv::StereoBM> leftMatcher;
	cv::Ptr <cv::StereoMatcher> rightMatcher;
	cv::Ptr <cv::ximgproc::DisparityWLSFilter> wlsFilter;
public:
	DepthMapCreator(int left, int right, int width, int height, int numDisparities, int blockSize);
	void build();
	~DepthMapCreator();
};

