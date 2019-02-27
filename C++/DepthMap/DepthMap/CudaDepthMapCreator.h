#pragma once
#include "opencv2\opencv.hpp"
#include "ximgproc.hpp"
#include "cudastereo.hpp"

class CudaDepthMapCreator
{
private:
	cv::VideoCapture leftCam, rightCam;
	cv::Mat leftFrame, rightFrame, greyLeft, greyRight, fixedLeft, fixedRight, disparityLeft, disparityRight, filteredDisparityMap, newLeftCameraMat, newRightCameraMat,
		leftCameraMat, leftDistroCoeff, rightCameraMat, rightDistroCoeff, Q, leftMapX, leftMapY, rightMapX, rightMapY;
	cv::Size im_size;
	cv::Ptr <cv::cuda::StereoBM> leftMatcher;
	cv::Ptr <cv::cuda::StereoBM> rightMatcher;
	cv::Ptr <cv::ximgproc::DisparityWLSFilter> wlsFilter;
	cv::cuda::GpuMat gpuLeft, gpuRight, gpuDisparityLeft, gpuDisparityRight;
public:
	CudaDepthMapCreator(int left, int right, int width, int height, int numDisparities, int blockSize);
	void build();
	~CudaDepthMapCreator();
};
