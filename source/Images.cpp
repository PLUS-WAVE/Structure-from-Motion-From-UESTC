#include "Images.h"


Images::Images(string const image_path) 
{
	// ��ȡͼ��
	this->image = imread(image_path);
	if (this->image.empty()) 
	{
		cout << "Could not read image: " << image_path << endl;
	}

	// ��ȡSIFT�������������
	Ptr<SIFT> sift = SIFT::create(0, 17, 0.0000000001, 16);
	sift->detectAndCompute(this->image, noArray(), this->keyPoints, this->descriptor);

	for (int i = 0; i < keyPoints.size(); i++)
	{
		correspond_struct_idx.push_back(-1);
	}
}


void Images::findColor()
{
	// ��������ƥ���
	for (Point2f& Points : this->matchedPoints)
	{	
		// ��ȡ���ص����ɫ
		Vec3b color = this->image.at<Vec3b>(Points.y, Points.x);

		// ����ɫ�洢����ɫ������
		this->colors.push_back(color);
	}
}


void Images::matchFeatures(Images& otherImage, vector<DMatch>& outputMatches)
{
	// ���ƥ���
	otherImage.matchedPoints.clear();
	this->matchedPoints.clear();

	vector<vector<DMatch>> matches;
	FlannBasedMatcher matcher;

	// ʹ��FlannBasedMatcher��������ƥ��
	matcher.knnMatch(this->descriptor, otherImage.descriptor, matches, 2);

	// ������С����
	float min_dist = FLT_MAX;
	for (int r = 0; r < matches.size(); ++r)
	{
		// �������ھ�����ڴν��ھ����2.5������������ƥ���
		if (matches[r][0].distance < 2.5 * matches[r][1].distance)
		{
			// ������С����
			float dist = matches[r][0].distance;
			if (dist < min_dist)
			{
				min_dist = dist;
			}
		}
	}


	// ɸѡ���õ�ƥ���
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.76 * matches[i][1].distance && matches[i][0].distance < 8 * max(min_dist, 10.0f))
		{
			outputMatches.push_back(matches[i][0]);
		}
	}

	// ��ƥ���洢��matchedPoints������
	for (int i = 0; i < outputMatches.size(); ++i)
	{
		this->matchedPoints.push_back(this->keyPoints[outputMatches[i].queryIdx].pt);
		otherImage.matchedPoints.push_back(otherImage.keyPoints[outputMatches[i].trainIdx].pt);
	}
}

// ��ƥ����л�ȡ��ά�ռ���ͼ���
void Images::getObjPointsAndImgPoints(vector<DMatch>& matches, vector<Point3d>& all_reconstructed_points, Images& preImage)
{
	// ���object_points��image_points
	this->object_points.clear();
	this->image_points.clear();

	// ��������ƥ���
	for (int i = 0; i < matches.size(); i++)
	{
		// ��ȡƥ�����ǰһ��ͼ���ж�Ӧ����ά�ռ�������
		int matched_world_point_indices = preImage.correspond_struct_idx[matches[i].queryIdx];

		// ���ƥ�����ǰһ��ͼ���ж�Ӧ����ά�ռ�����
		if (matched_world_point_indices > 0)
		{
			// ���䣨ǰһ��ͼ���е���ά�㣩��ӵ�object_points��
			this->object_points.push_back(all_reconstructed_points[matched_world_point_indices]);

			// ��ƥ��㣨����ͼ��Ķ�ά�㣩��ӵ�image_points��
			this->image_points.push_back(this->keyPoints[matches[i].trainIdx].pt);
		}
	}
}
