// ����ͼ���ļ�·���ͱ�������·��
#define IMG_PATH1 "test_img\\images\\100_7105.jpg"
#define IMG_PATH2 "test_img\\images\\100_7106.jpg"
#define PLY_SAVE_PATH "test_img\\results\\output.ply"
#define K_NUM 2905.88, 0, 1416, 0, 2905.88, 1064, 0, 0, 1 // 3*3

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
    // �׶�һ------------------------------------------------------------------------------------
    // ��ȡ����ͼ��
    Mat img1 = imread(IMG_PATH1);
    Mat img2 = imread(IMG_PATH2);
    if (img1.empty() || img2.empty())
    {
        cout << "�޷���ȡͼ��" << endl;
        return -1;
    }

    // ����SIFT����
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // ���ؼ��㲢����������
    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // ʹ��FLANN��������ƥ��
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);
    vector<DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i)
    {
        const float ratio = 0.7f;
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }
    
    
    // �׶ζ�------------------------------------------------------------------------------------
    // �������ڱ���ƥ���Ե�����
    vector<Point2f> matchedPoints1, matchedPoints2;

    for (int i = 0; i < good_matches.size(); ++i)
    {
        matchedPoints1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        matchedPoints2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    

    // ���л�������F�Ĺ��Ʋ�ʹ��RANSACɸѡ
    Mat F;
    vector<uchar> inliers;
    F = findFundamentalMat(matchedPoints1, matchedPoints2, inliers, FM_RANSAC);
    cout << F << endl;

    vector<Point2f> inlierPoints1;
    vector<Point2f> inlierPoints2;
    for (int i = 0; i < inliers.size(); ++i)
    {
        if (inliers[i])
        {
			inlierPoints1.push_back(matchedPoints1[i]);
			inlierPoints2.push_back(matchedPoints2[i]);
		}
	}

    // ����ڲξ���K
    Mat K = (Mat_<double>(3, 3) << K_NUM);
    cout << K << endl;

    ////���㱾�ʾ���E
    //Mat E = findEssentialMat(matchedPoints1, matchedPoints2, K, RANSAC, 0.999, 1.0, inliers);
    Mat E = K.t() * F * K;
    cout << "Essential Matrix (E):" << endl;
    cout << E << endl;
    
    //�����̬�ָ������R,t,ͶӰ����
    Mat R, t;
    recoverPose(E, inlierPoints1, inlierPoints2, K, R, t);
    cout << "recoverpose" << endl;
    cout << "R:" << R << endl;
    cout << "t:" << t << endl;
    
    // �������������ͶӰ���� [R T]
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);
    
    // ���õ�һ�������ͶӰ����Ϊ��λ���� [I | 0]
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    
    // ���õڶ��������ͶӰ����Ϊ�������ת���� R ��ƽ������ T
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    t.convertTo(proj2.col(3), CV_32FC1);
    
    // ת������ڲξ��� K Ϊ������
    Mat fK;
    K.convertTo(fK, CV_32FC1);
    
    // ����ͶӰ���� [K * [R|T]]
    proj1 = fK * proj1;
    proj2 = fK * proj2;
    
    // ���Ƿ����ϡ����ά����
    Mat point4D_homogeneous(4, inlierPoints1.size(), CV_64F);
    triangulatePoints(proj1, proj2, inlierPoints1, inlierPoints2, point4D_homogeneous);
   

    // ���������ת��Ϊ��ά����
    Mat point3D;
    convertPointsFromHomogeneous(point4D_homogeneous.t(), point3D);
    cout << point3D << endl;

    // ��ȡ���������ɫ��Ϣ
    vector<Vec3b> colors1, colors2; // ��ɫ��Ϣ

    for (Point2f& inlierPoints : inlierPoints1)
    {
        int x = cvRound(inlierPoints.x); // �ؼ����x����
        int y = cvRound(inlierPoints.y); // �ؼ����y����
        Vec3b color = img1.at<Vec3b>(y, x);
        colors1.push_back(color);
    }

    for (Point2f& inlierPoints : inlierPoints2)
    {
        int x = cvRound(inlierPoints.x); // �ؼ����x����
        int y = cvRound(inlierPoints.y); // �ؼ����y����
        Vec3b color = img2.at<Vec3b>(y, x);
        colors2.push_back(color);
    }

    // ��������ɫ�ĵ������ݽṹ
    vector<Point3f> pointCloud;
    vector<Vec3b> pointColors;

    // ������ɫ��Ϣ������
    for (int i = 0; i < point3D.rows; ++i)
    {
        Point3f point = point3D.at<Point3f>(i);
        Vec3b color1 = colors1[i];
        Vec3b color2 = colors2[i];

        // ��������Ը�����Ҫѡ��ʹ���ĸ���ɫ�����߽�����ɫ��ֵ�ȴ���
        Vec3b finalColor = color1; // ����ʾ��ʹ�õ�һ���������ɫ

        pointCloud.push_back(point);
        pointColors.push_back(finalColor);
    }

    // �ֶ��������ply�ļ�
    ofstream plyFile(PLY_SAVE_PATH);

    // ply��ͷ����Ϣ
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << point3D.rows << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "property uchar blue\n";
    plyFile << "property uchar green\n";
    plyFile << "property uchar red\n";
    plyFile << "end_header\n";

    // д���������
    for (int i = 0; i < point3D.rows; ++i)
    {
        Vec3b color = pointColors[i];
        const float* point = point3D.ptr<float>(i);
        plyFile << point[0] << " " << point[1] << " " << point[2] << " "
                << static_cast<int>(color[0]) << " "
                << static_cast<int>(color[1]) << " "
                << static_cast<int>(color[2]) << endl;
    }

    plyFile.close();   
    return 0;
}
