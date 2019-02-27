#include "CameraCapture.h"

CameraCapture::CameraCapture(int left, int right, int width, int height)
{
	leftCam = cv::VideoCapture(left);
	rightCam = cv::VideoCapture(right);

	leftCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	leftCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	rightCam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	rightCam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

void CameraCapture::capture() {
	int frameId = 0;

	std::string folderLeft = "testImages\\left";
	std::string folderRight = "testImages\\right";

	std::string folderCreateCommandLeft = "mkdir " + folderLeft;
	system(folderCreateCommandLeft.c_str());

	std::string folderCreateCommandRight = "mkdir " + folderRight;
	system(folderCreateCommandRight.c_str());

	while (true)
	{
		std::ostringstream leftName, rightName;

		if (!leftCam.grab() && !rightCam.grab()) {
			std::cout << "No more breaks" << std::endl;
			break;
		}

		if (leftCam.retrieve(leftFrame) && rightCam.retrieve(rightFrame)) {
			cv::cvtColor(leftFrame, greyLeft, cv::COLOR_BGR2GRAY);
			cv::cvtColor(rightFrame, greyRight, cv::COLOR_BGR2GRAY);

			hasLeft = cv::findChessboardCorners(leftFrame, chessboardSize, cornerLeft );
			hasRight = cv::findChessboardCorners(rightFrame, chessboardSize, cornerRight);

			if (hasLeft && hasRight) {
				frameId++;

				leftName << "testImages\\left\\" << frameId << ".jpg";
				rightName << "testImages\\right\\" << frameId << ".jpg";

				std::cout << "Saving img pair " << frameId << std::endl;
				cv::imwrite(leftName.str(), leftFrame);
				cv::imwrite(rightName.str(), rightFrame);

				cv::drawChessboardCorners(leftFrame, chessboardSize, cornerLeft, hasLeft);
				cv::drawChessboardCorners(rightFrame, chessboardSize, cornerRight, hasRight);

				cv::imshow("Left View", leftFrame);
				cv::imshow("Right View", rightFrame);
				cv::waitKey(10);

			}
		}
	}
}

CameraCapture::~CameraCapture()
{
	leftCam.release();
	rightCam.release();
	cv::destroyAllWindows();
}
