#include "Constructor.h"


void Constructor::findCamera(Mat K, vector<Point2f>& point1, vector<Point2f>& point2, Mat& output_R, Mat& output_T, vector<uchar>& mask)
{
	vector<uchar> inliers;

	Mat F;
	F = findFundamentalMat(point1, point2, inliers, FM_RANSAC, 1, 0.5);
	Mat E = K.t() * F * K;

	//Mat E = findEssentialMat(point1, point2, K, RANSAC, 0.5, 1.0, inliers);
	
	mask = inliers;

	// �����ڵ�ɸѡ���µ�ƥ���
	Constructor::maskoutPoints(point1, inliers);
	Constructor::maskoutPoints(point2, inliers);

	// �ֽ�E���󣬻�ȡR��T����
	int pass_count = recoverPose(E, point1, point2, K, output_R, output_T);
}



void Constructor::maskoutPoints(vector<Point2f>& input_points, vector<uchar>& input_mask)
{
	vector<Point2f> temp_points(input_points);
	input_points.clear();

	for (int i = 0; i < temp_points.size(); ++i)
	{
		if (input_mask[i])
		{
			input_points.push_back(temp_points[i]);
		}
	}
}


vector<Point3d>& Constructor::pointsReconstruct(const Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& points1, vector<Point2f>& points2)
{
	// ����ͶӰ����
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	// ����ת�����ƽ�������ϲ�ΪͶӰ����
	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	// ���ڲξ�����ͶӰ������ˣ��õ����յ�ͶӰ����
	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// ���ǻ����õ��������
	Mat point4D_homogeneous(4, points1.size(), CV_64F);
	triangulatePoints(proj1, proj2, points1, points2, point4D_homogeneous);

	//// ���������ת��Ϊ��ά����
	vector<Point3d> point3D;
	point3D.clear();
	point3D.reserve(point4D_homogeneous.cols);
	for (int i = 0; i < point4D_homogeneous.cols; ++i)
	{
		Mat_<float> col = point4D_homogeneous.col(i);
		col /= col(3);
		point3D.push_back(Point3d(col(0), col(1), col(2)));
	}

	// ����ά����洢��Point3d�����в�����
	return point3D;
}