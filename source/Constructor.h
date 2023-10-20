#ifndef CONSTRUCTOR_H
#define CONSTRUCTOR_H

#include "Includes.h"
#include "Images.h"

class Constructor
{
public:
	// 输入K，图1的匹配点，图2的匹配点；输出R，T；点经过筛选
	static void findCamera(Mat K, vector<Point2f>& point1, vector<Point2f>& point2, Mat& output_R, Mat& output_T, vector<uchar>& mask);
	
	// 输入图匹配点，内点标记mask；返回mask后的vector<Point2f>匹配点
	static void maskoutPoints(vector<Point2f>& input_points, vector<uchar>& input_mask);

	// 输入图一的R，T，匹配点，图二的R，T，匹配点；返回vector<Point3f>三维点
	static vector<Point3d>& pointsReconstruct(const Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& points1, vector<Point2f>& points2);
};


#endif // !CONSTRUCTOR_H

#pragma once
