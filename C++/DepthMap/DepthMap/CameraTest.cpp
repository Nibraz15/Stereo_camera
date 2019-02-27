#include "CameraTest.h"

CameraTest::CameraTest(int left, int right, int width, int height)
{
	leftCam = cv::VideoCapture(left);
	rightCam = cv::VideoCapture(right);

	leftCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	leftCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	rightCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	rightCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

void CameraTest::test() {
	while (true)
	{
		if (!leftCam.grab() && !rightCam.grab()) {
			std::cout << "No more breaks" << std::endl;
			break;
		}

		if (leftCam.retrieve(leftFrame) && rightCam.retrieve(rightFrame)) {
			cv::imshow("Left View", leftFrame);
			cv::imshow("Right View", rightFrame);
			cv::waitKey(10);
		}
	}
}


CameraTest::~CameraTest()
{
	leftCam.release();
	rightCam.release();
	cv::destroyAllWindows();
}
