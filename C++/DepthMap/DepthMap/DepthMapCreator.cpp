#include "DepthMapCreator.h"

DepthMapCreator::DepthMapCreator(int left, int right, int width, int height, int numDisparities, int blockSize)
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
	fs["newLeftCameraMat"]>> newLeftCameraMat;
	fs["newRightCameraMat"]>> newRightCameraMat;

	leftCam = cv::VideoCapture(left);
	rightCam = cv::VideoCapture(right);

	leftCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	leftCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	rightCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	rightCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	leftMatcher = cv::StereoBM::create(numDisparities, blockSize);
	rightMatcher = cv::ximgproc::createRightMatcher(leftMatcher);

	wlsFilter = cv::ximgproc::createDisparityWLSFilter(leftMatcher);
	wlsFilter->setLambda(8000);
	wlsFilter->setSigmaColor(1.2);
}

void DepthMapCreator::build() {
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

			leftMatcher->compute(greyLeft, greyRight, disparityLeft);
			rightMatcher->compute(greyRight, greyLeft, disparityRight);

			disparityLeft.convertTo(disparityLeft, CV_16S);
			disparityLeft.convertTo(disparityRight, CV_16S);

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


DepthMapCreator::~DepthMapCreator()
{
	leftCam.release();
	rightCam.release();
	cv::destroyAllWindows();
}
