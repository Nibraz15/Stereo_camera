#pragma once
#include "opencv2\opencv.hpp"
#include <sstream>

class CameraCapture
{
private:
	cv::VideoCapture leftCam, rightCam;
	cv::Mat leftFrame, rightFrame, greyLeft, greyRight, cornerLeft, cornerRight;
	cv::Size chessboardSize = cv::Size(7, 7);
	bool hasLeft, hasRight;

public:
	CameraCapture(int left, int right, int width, int height);
	void capture();
	~CameraCapture();
};

