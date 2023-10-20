#ifndef INCLUDES_H
#define INCLUDES_H

#define OPENCV_ENABLE_NONFREE 0

#define GLOG_NO_ABBREVIATED_SEVERITIES
#define _CRT_NONSTDC_NO_DEPRECATE
#define NOMINMAX
#define _CRT_NONSTDC_NO_WARNINGS
#pragma warning(disable: 4996)

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>


using namespace cv;
using namespace std;
#endif // !INCLUDES_H

#pragma once
