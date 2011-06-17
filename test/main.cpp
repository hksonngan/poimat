#include <stdio.h>
#include "filter.h"
#include "opencv2/opencv.hpp"
using namespace cv;

#include <cutil_inline.h>
#include <math.h>

int main(int argc, char* argv[])
{
	if(argc != 3)
		return 0;

	Mat image = imread(argv[1]);
	Mat trimap = imread(argv[2], 0);

	printf("image size: %dx%d\n", image.cols, image.rows);
	printf("trimap size: %dx%d\n", trimap.cols, trimap.rows);

	cvtColor(image, image, CV_BGR2RGBA);

	float *result;
	cudaMalloc((void**)&result, image.cols*image.rows*4*sizeof(float));

	initializeFilter(image.cols, image.rows);
	poissonFilter(image.data, trimap.data, result);
	trimapFilter(image.data, trimap.data);
	releaseFilter();

	cudaFree(result);
}