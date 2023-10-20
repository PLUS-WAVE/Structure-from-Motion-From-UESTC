#ifndef IMAGES_H
#define IMAGES_H

#include "Includes.h"

class Images
{
public:
	Mat image; // �洢ͼ��
	vector<KeyPoint> keyPoints; // �洢������
	Mat descriptor; // �洢����������
	vector<int> correspond_struct_idx; // ƥ�������Ӧ�Ŀռ���ڵ����е�����
	vector<Point2f> matchedPoints; // �洢ƥ���
	vector<Vec3b> colors; // �洢ƥ������ɫ��Ϣ
	Mat R, T; // �洢�������ת�����ƽ������

	vector<Point3f> object_points; // ǰһ��ͼ��ƥ����Ӧ����ά��
	vector<Point2f> image_points; // ����ͼ���ж�Ӧ�����ص�

	// ���캯������ָ��·����ȡͼ�񣬲���ȡSIFT�������������
	Images(string const image_paths);

	// ����ƥ�亯��������ǰͼ������һ��ͼ���������ƥ��
	void matchFeatures(Images& otherImage, vector<DMatch>& outputMatches);

	// ��ƥ�������ȡ��ɫ��Ϣ
	void findColor();

	// ������ǰƥ�䣬�ҳ���ǰƥ�����Ѿ��ڵ����еĵ㣬��ȡobject_points���Լ�image_points
	void getObjPointsAndImgPoints(vector<DMatch>& matches, vector<Point3d>& all_reconstructed_points, Images& preImage);
};


#endif // !IMAGES_H
#pragma once
