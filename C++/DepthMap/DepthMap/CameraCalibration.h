#pragma once
#include "opencv2\opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

class CameraCalibration
{
private:
	int width, height;
	std::vector < std::vector< cv::Point3f > > object_points;
	std::vector < std::vector< cv::Point2f > > image_points_left, image_points_right;
	std::vector< cv::Point2f > cornersLeft, cornersRight;
	cv::Vec3d T;
	cv::Mat imgLeft, imgRight, greyLeft, greyRight, leftCameraMat, leftDistroCoeff, rightCameraMat, rightDistroCoeff, newLeftCameraMat, newRightCameraMat,
		RectLeft, RectRight, projLeft, projRight, Q, leftMapX, leftMapY, rightMapX, rightMapY, rvecL, tvecL, rvecR, tvecR, R, E, F;
	cv::Size imageSizeLeft, imageSizeRight, im_size, board_size;
	int board_n;

public:
	CameraCalibration(int width, int height);
	void Calibrate(int num_imgs, int boardWidth, int boardHeight);
	~CameraCalibration();
};

