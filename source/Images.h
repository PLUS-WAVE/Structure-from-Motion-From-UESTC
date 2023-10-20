#ifndef IMAGES_H
#define IMAGES_H

#include "Includes.h"

class Images
{
public:
	Mat image; // 存储图像
	vector<KeyPoint> keyPoints; // 存储特征点
	Mat descriptor; // 存储特征描述符
	vector<int> correspond_struct_idx; // 匹配点所对应的空间点在点云中的索引
	vector<Point2f> matchedPoints; // 存储匹配点
	vector<Vec3b> colors; // 存储匹配点的颜色信息
	Mat R, T; // 存储相机的旋转矩阵和平移向量

	vector<Point3f> object_points; // 前一张图中匹配点对应的三维点
	vector<Point2f> image_points; // 在现图像中对应的像素点

	// 构造函数，从指定路径读取图像，并提取SIFT特征点和描述符
	Images(string const image_paths);

	// 特征匹配函数，将当前图像与另一个图像进行特征匹配
	void matchFeatures(Images& otherImage, vector<DMatch>& outputMatches);

	// 从匹配点中提取颜色信息
	void findColor();

	// 遍历当前匹配，找出当前匹配中已经在点云中的点，获取object_points，以及image_points
	void getObjPointsAndImgPoints(vector<DMatch>& matches, vector<Point3d>& all_reconstructed_points, Images& preImage);
};


#endif // !IMAGES_H
#pragma once
