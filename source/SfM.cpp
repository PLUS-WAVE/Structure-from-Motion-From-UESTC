// 定义图像文件路径和保存结果的路径
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
    // 阶段一------------------------------------------------------------------------------------
    // 读取两幅图像
    Mat img1 = imread(IMG_PATH1);
    Mat img2 = imread(IMG_PATH2);
    if (img1.empty() || img2.empty())
    {
        cout << "无法读取图像" << endl;
        return -1;
    }

    // 创建SIFT对象
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // 检测关键点并计算描述子
    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // 使用FLANN进行特征匹配
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
    
    
    // 阶段二------------------------------------------------------------------------------------
    // 声明用于保存匹配点对的容器
    vector<Point2f> matchedPoints1, matchedPoints2;

    for (int i = 0; i < good_matches.size(); ++i)
    {
        matchedPoints1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        matchedPoints2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    

    // 进行基本矩阵F的估计并使用RANSAC筛选
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

    // 相机内参矩阵K
    Mat K = (Mat_<double>(3, 3) << K_NUM);
    cout << K << endl;

    ////计算本质矩阵E
    //Mat E = findEssentialMat(matchedPoints1, matchedPoints2, K, RANSAC, 0.999, 1.0, inliers);
    Mat E = K.t() * F * K;
    cout << "Essential Matrix (E):" << endl;
    cout << E << endl;
    
    //相机姿态恢复，求解R,t,投影矩阵
    Mat R, t;
    recoverPose(E, inlierPoints1, inlierPoints2, K, R, t);
    cout << "recoverpose" << endl;
    cout << "R:" << R << endl;
    cout << "t:" << t << endl;
    
    // 创建两个相机的投影矩阵 [R T]
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);
    
    // 设置第一个相机的投影矩阵为单位矩阵 [I | 0]
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    
    // 设置第二个相机的投影矩阵为输入的旋转矩阵 R 和平移向量 T
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    t.convertTo(proj2.col(3), CV_32FC1);
    
    // 转换相机内参矩阵 K 为浮点型
    Mat fK;
    K.convertTo(fK, CV_32FC1);
    
    // 计算投影矩阵 [K * [R|T]]
    proj1 = fK * proj1;
    proj2 = fK * proj2;
    
    // 三角法求解稀疏三维点云
    Mat point4D_homogeneous(4, inlierPoints1.size(), CV_64F);
    triangulatePoints(proj1, proj2, inlierPoints1, inlierPoints2, point4D_homogeneous);
   

    // 将齐次坐标转换为三维坐标
    Mat point3D;
    convertPointsFromHomogeneous(point4D_homogeneous.t(), point3D);
    cout << point3D << endl;

    // 获取特征点的颜色信息
    vector<Vec3b> colors1, colors2; // 颜色信息

    for (Point2f& inlierPoints : inlierPoints1)
    {
        int x = cvRound(inlierPoints.x); // 关键点的x坐标
        int y = cvRound(inlierPoints.y); // 关键点的y坐标
        Vec3b color = img1.at<Vec3b>(y, x);
        colors1.push_back(color);
    }

    for (Point2f& inlierPoints : inlierPoints2)
    {
        int x = cvRound(inlierPoints.x); // 关键点的x坐标
        int y = cvRound(inlierPoints.y); // 关键点的y坐标
        Vec3b color = img2.at<Vec3b>(y, x);
        colors2.push_back(color);
    }

    // 创建带颜色的点云数据结构
    vector<Point3f> pointCloud;
    vector<Vec3b> pointColors;

    // 关联颜色信息到点云
    for (int i = 0; i < point3D.rows; ++i)
    {
        Point3f point = point3D.at<Point3f>(i);
        Vec3b color1 = colors1[i];
        Vec3b color2 = colors2[i];

        // 在这里可以根据需要选择使用哪个颜色，或者进行颜色插值等处理
        Vec3b finalColor = color1; // 这里示例使用第一个相机的颜色

        pointCloud.push_back(point);
        pointColors.push_back(finalColor);
    }

    // 手动输出点云ply文件
    ofstream plyFile(PLY_SAVE_PATH);

    // ply的头部信息
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

    // 写入点云数据
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
