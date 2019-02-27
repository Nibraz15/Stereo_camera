#include "CameraTest.h"
#include "CameraCalibration.h"
#include "CameraCapture.h"
#include "DepthMapCreator.h"
#include "CudaDepthMapCreator.h"

int main()
{
	int choice, numImages;

	std::cout << " Press " << std::endl;
	std::cout << std::endl;
	std::cout << "  1. Test cameras " << std::endl;
	std::cout << "  2. Capture Calibration Images  " << std::endl;
	std::cout << "  3. Calibrate Cameras " << std::endl;
	std::cout << "  4. Display Depth Map " << std::endl;
	std::cout << "  5. Display Cuda Depth Map " << std::endl;
	std::cout << std::endl;
	std::cout << "  Choice : ";
	std::cin >> choice;

	if (choice == 1) 
	{
		CameraTest test = CameraTest(0, 1, 640, 480);
		test.test();
	}
	else if (choice == 2)
	{
		CameraCapture capture = CameraCapture(0, 1, 640, 480);
		capture.capture();
	}
	else if (choice == 3)
	{
		std::cout << std::endl;
		std::cout << " Number of images to use? ";
		std::cin >> numImages;
		std::cout << std::endl;

		CameraCalibration calibrator = CameraCalibration(640, 480);
		calibrator.Calibrate(numImages, 7, 7);
	}
	else if (choice == 4)
	{
		DepthMapCreator creator = DepthMapCreator(0, 1, 640, 480, 32, 15);
		creator.build();
	}
	else if (choice == 5)
	{
		CudaDepthMapCreator creator = CudaDepthMapCreator(0, 1, 640, 480, 64, 19);
		creator.build();
	}
}

