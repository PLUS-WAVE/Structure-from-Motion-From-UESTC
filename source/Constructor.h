#ifndef CONSTRUCTOR_H
#define CONSTRUCTOR_H

#include "Includes.h"
#include "Images.h"

class Constructor
{
public:
	// ����K��ͼ1��ƥ��㣬ͼ2��ƥ��㣻���R��T���㾭��ɸѡ
	static void findCamera(Mat K, vector<Point2f>& point1, vector<Point2f>& point2, Mat& output_R, Mat& output_T, vector<uchar>& mask);
	
	// ����ͼƥ��㣬�ڵ���mask������mask���vector<Point2f>ƥ���
	static void maskoutPoints(vector<Point2f>& input_points, vector<uchar>& input_mask);

	// ����ͼһ��R��T��ƥ��㣬ͼ����R��T��ƥ��㣻����vector<Point3f>��ά��
	static vector<Point3d>& pointsReconstruct(const Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& points1, vector<Point2f>& points2);
};


#endif // !CONSTRUCTOR_H

#pragma once
