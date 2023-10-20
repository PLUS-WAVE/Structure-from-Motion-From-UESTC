#include "Images.h"


Images::Images(string const image_path) 
{
	// 读取图像
	this->image = imread(image_path);
	if (this->image.empty()) 
	{
		cout << "Could not read image: " << image_path << endl;
	}

	// 提取SIFT特征点和描述符
	Ptr<SIFT> sift = SIFT::create(0, 17, 0.0000000001, 16);
	sift->detectAndCompute(this->image, noArray(), this->keyPoints, this->descriptor);

	for (int i = 0; i < keyPoints.size(); i++)
	{
		correspond_struct_idx.push_back(-1);
	}
}


void Images::findColor()
{
	// 遍历所有匹配点
	for (Point2f& Points : this->matchedPoints)
	{	
		// 获取像素点的颜色
		Vec3b color = this->image.at<Vec3b>(Points.y, Points.x);

		// 将颜色存储在颜色向量中
		this->colors.push_back(color);
	}
}


void Images::matchFeatures(Images& otherImage, vector<DMatch>& outputMatches)
{
	// 清空匹配点
	otherImage.matchedPoints.clear();
	this->matchedPoints.clear();

	vector<vector<DMatch>> matches;
	FlannBasedMatcher matcher;

	// 使用FlannBasedMatcher进行特征匹配
	matcher.knnMatch(this->descriptor, otherImage.descriptor, matches, 2);

	// 计算最小距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < matches.size(); ++r)
	{
		// 如果最近邻距离大于次近邻距离的2.5倍，则跳过该匹配点
		if (matches[r][0].distance < 2.5 * matches[r][1].distance)
		{
			// 计算最小距离
			float dist = matches[r][0].distance;
			if (dist < min_dist)
			{
				min_dist = dist;
			}
		}
	}


	// 筛选出好的匹配点
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.76 * matches[i][1].distance && matches[i][0].distance < 8 * max(min_dist, 10.0f))
		{
			outputMatches.push_back(matches[i][0]);
		}
	}

	// 将匹配点存储在matchedPoints向量中
	for (int i = 0; i < outputMatches.size(); ++i)
	{
		this->matchedPoints.push_back(this->keyPoints[outputMatches[i].queryIdx].pt);
		otherImage.matchedPoints.push_back(otherImage.keyPoints[outputMatches[i].trainIdx].pt);
	}
}

// 从匹配点中获取三维空间点和图像点
void Images::getObjPointsAndImgPoints(vector<DMatch>& matches, vector<Point3d>& all_reconstructed_points, Images& preImage)
{
	// 清空object_points和image_points
	this->object_points.clear();
	this->image_points.clear();

	// 遍历所有匹配点
	for (int i = 0; i < matches.size(); i++)
	{
		// 获取匹配点在前一张图像中对应的三维空间点的索引
		int matched_world_point_indices = preImage.correspond_struct_idx[matches[i].queryIdx];

		// 如果匹配点在前一张图像中对应的三维空间点存在
		if (matched_world_point_indices > 0)
		{
			// 将其（前一张图像中的三维点）添加到object_points中
			this->object_points.push_back(all_reconstructed_points[matched_world_point_indices]);

			// 将匹配点（该新图像的二维点）添加到image_points中
			this->image_points.push_back(this->keyPoints[matches[i].trainIdx].pt);
		}
	}
}
