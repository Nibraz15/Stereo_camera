#include "CudaDepthMapCreator.h"

CudaDepthMapCreator::CudaDepthMapCreator(int left, int right, int width, int height, int numDisparities, int blockSize)
{
	std::cout << "loading calculated parameters...." << std::endl;
	cv::FileStorage fs("Config.yml", cv::FileStorage::READ);
	//fs["im_size"] >> im_size;
	//fs["leftMapX"] >> leftMapX;
	//fs["leftMapY"] >> leftMapY;
	//fs["rightMapX"] >> rightMapX;
	//fs["rightMapY"] >> rightMapY;
	fs["Q"] >> Q;
	fs["leftCameraMat"] >> leftCameraMat;
	fs["leftDistroCoeff"] >> leftDistroCoeff;
	fs["rightCameraMat"] >> rightCameraMat;
	fs["rightDistroCoeff"] >> rightDistroCoeff;
	fs["newLeftCameraMat"] >> newLeftCameraMat;
	fs["newRightCameraMat"] >> newRightCameraMat;

	leftCam = cv::VideoCapture(left);
	rightCam = cv::VideoCapture(right);

	leftCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	leftCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	rightCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	rightCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	leftMatcher = cv::cuda::createStereoBM(numDisparities, blockSize);
	rightMatcher = cv::cuda::createStereoBM(numDisparities, blockSize);

	wlsFilter = cv::ximgproc::createDisparityWLSFilter(leftMatcher);
	wlsFilter->setLambda(8000);
	wlsFilter->setSigmaColor(1.2);
}

void CudaDepthMapCreator::build() {
	while (true)
	{
		if (!leftCam.grab() && !rightCam.grab()) {
			std::cout << "No more breaks" << std::endl;
			break;
		}

		if (leftCam.retrieve(leftFrame) && rightCam.retrieve(rightFrame)) {
			cv::undistort(leftFrame, fixedLeft, leftCameraMat, leftDistroCoeff, newLeftCameraMat);
			cv::undistort(rightFrame, fixedRight, rightCameraMat, rightDistroCoeff, newRightCameraMat);

			//cv::remap(leftFrame, fixedLeft, leftMapX, leftMapY, CV_INTER_LINEAR);
			//cv::remap(rightFrame, fixedRight, rightMapX, rightMapY, CV_INTER_LINEAR);

			cv::cvtColor(fixedLeft, greyLeft, cv::COLOR_BGR2GRAY);
			cv::cvtColor(fixedRight, greyRight, cv::COLOR_BGR2GRAY);

			gpuLeft.upload(greyLeft);
			gpuRight.upload(greyRight);

			leftMatcher->compute(gpuLeft, gpuRight, gpuDisparityLeft);
			rightMatcher->compute(gpuRight, gpuLeft, gpuDisparityRight);

			gpuDisparityLeft.convertTo(gpuDisparityLeft, CV_16S);
			gpuDisparityRight.convertTo(gpuDisparityRight, CV_16S);

			gpuDisparityLeft.download(disparityLeft);
			gpuDisparityRight.download(disparityRight);

			wlsFilter->filter(disparityLeft, fixedLeft, filteredDisparityMap, disparityRight);

			cv::normalize(filteredDisparityMap, filteredDisparityMap, (double)255.0, (double)0.0, cv::NORM_MINMAX);
			filteredDisparityMap.convertTo(filteredDisparityMap, CV_8U);
			cv::applyColorMap(filteredDisparityMap, filteredDisparityMap, cv::COLORMAP_JET);

			cv::imshow("left Cam", fixedLeft);
			cv::imshow("right Cam", fixedRight);
			cv::imshow("Disparity Map", filteredDisparityMap);
			cv::waitKey(10);
		}
	}
}


CudaDepthMapCreator::~CudaDepthMapCreator()
{
	leftCam.release();
	rightCam.release();
	cv::destroyAllWindows();
}
