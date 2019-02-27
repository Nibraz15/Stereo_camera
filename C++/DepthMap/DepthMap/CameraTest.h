#pragma once
#include "opencv2\opencv.hpp"


class CameraTest
{
private:
	cv::VideoCapture leftCam, rightCam;
	cv::Mat leftFrame, rightFrame;

public:
	CameraTest(int left, int right, int width, int height);
	void test();
	~CameraTest();
};

