#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(int width, int height)
{
	im_size = cv::Size(width, height);
}

void CameraCalibration::Calibrate(int num_imgs, int boardWidth, int boardHeight) {
	board_size = cv::Size(boardWidth, boardHeight);
	std::vector< cv::Point3f > obj;

	std::cout << "extracting image points..." << std::endl;
	for (int i = 0; i < boardWidth; i++) {
		for (int j = 0; j < boardHeight; j++) {
			obj.push_back(cv::Point3f((float)i, (float)j, 0.0f));
		}
	}

	for (int k = 1; k <= num_imgs; k++) {

		std::ostringstream leftName, rightName;
		bool foundLeft = false;
		bool foundRight = false;

		//creating path to image as a stream of string
		leftName << "testImages\\left\\" << k << ".jpg";
		rightName << "testImages\\right\\" << k << ".jpg";

		//reading images according to string stream
		imgLeft = cv::imread(leftName.str(), CV_LOAD_IMAGE_COLOR);
		imgRight = cv::imread(rightName.str(), CV_LOAD_IMAGE_COLOR);

		//converting to grey color format for processing
		cv::cvtColor(imgLeft, greyLeft, CV_BGR2GRAY);
		cv::cvtColor(imgRight, greyRight, CV_BGR2GRAY);

		imageSizeLeft = greyLeft.size();
		imageSizeRight = greyRight.size();

		if (imageSizeLeft != im_size) {
			std::cout << "captured left images are not the intended size for calibration" << std::endl;
			break;
		}

		if (imageSizeRight != im_size) {
			std::cout << "captured right images are not the intended size for calibration" << std::endl;
			break;
		}

		//checking for chessboard reqired for calibration 
		foundLeft = cv::findChessboardCorners(imgLeft, board_size, cornersLeft);
		foundRight = cv::findChessboardCorners(imgRight, board_size, cornersRight);

		//if chessboards are found, further processing to extract the pixels related to chessbaord corners as image points and object points
		if (foundLeft && foundRight) {
			
			cv::cornerSubPix(greyLeft, cornersLeft, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001));
			cv::cornerSubPix(greyRight, cornersRight, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001));

			image_points_left.push_back(cornersLeft);
			image_points_right.push_back(cornersRight);

			object_points.push_back(obj);
			
			cv::drawChessboardCorners(greyLeft, board_size, cornersLeft, foundLeft);
			cv::drawChessboardCorners(greyRight, board_size, cornersRight, foundRight);

			cv::imshow("left", greyLeft);
			cv::imshow("right", greyRight);
			cv::waitKey(1);
		}
		std::cout << "image " << k << " analyzed..." << std::endl;
	}

	//Calibrating cameras
	std::cout << "calibrating left camera...." << std::endl;
	cv::calibrateCamera(object_points, image_points_left, im_size, leftCameraMat, leftDistroCoeff, rvecL, tvecL);

	std::cout << "calibrating right camera...." << std::endl;
	cv::calibrateCamera(object_points, image_points_right, im_size, rightCameraMat, rightDistroCoeff, rvecR, tvecR);

	//Calibrating as a stereo cameras
	std::cout << "calibrating as a stereo camera...." << std::endl;
	cv::stereoCalibrate(object_points, image_points_left, image_points_right, leftCameraMat, leftDistroCoeff, rightCameraMat, rightDistroCoeff, im_size, R, T, E, F, CV_CALIB_FIX_INTRINSIC, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	
	//calculating rectifying parameters
	std::cout << "calculating rectifying parameters...." << std::endl;
	cv::stereoRectify(leftCameraMat, leftDistroCoeff, rightCameraMat, rightDistroCoeff, im_size, R, T, RectLeft, RectRight, projLeft, projRight, Q, CV_CALIB_ZERO_DISPARITY, 0, im_size);
	
	//simplifying all above parameters into mapping parameters
	std::cout << "calculating mapping paramaeters for remapping...." << std::endl;
	newLeftCameraMat = cv::getOptimalNewCameraMatrix(leftCameraMat, leftDistroCoeff, im_size, 1, im_size);
	newRightCameraMat = cv::getOptimalNewCameraMatrix(rightCameraMat, rightDistroCoeff, im_size, 1, im_size);
	
	//cv::initUndistortRectifyMap(leftCameraMat, leftDistroCoeff, RectLeft, projLeft, im_size, CV_32FC1, leftMapX, leftMapY);
	//cv::initUndistortRectifyMap(rightCameraMat, rightDistroCoeff, RectRight, projRight, im_size, CV_32FC1, rightMapX, rightMapY);

	//Saving calculated parameters
	std::cout << "saving calculated parameters...." << std::endl;
	cv::FileStorage fs("Config.yml", cv::FileStorage::WRITE);
	fs << "im_size" << im_size;
	//fs << "leftMapX" << leftMapX;
	//fs << "leftMapY" << leftMapY;
	//fs << "rightMapX" << rightMapX;
	//fs << "rightMapY" << rightMapY;
	fs << "Q" << Q;
	fs << "leftCameraMat" << leftCameraMat;
	fs << "leftDistroCoeff" << leftDistroCoeff;
	fs << "rightCameraMat" << rightCameraMat;
	fs << "rightDistroCoeff" << rightDistroCoeff;
	fs << "newLeftCameraMat" << newLeftCameraMat;
	fs << "newRightCameraMat" << newRightCameraMat;
}

CameraCalibration::~CameraCalibration()
{
}