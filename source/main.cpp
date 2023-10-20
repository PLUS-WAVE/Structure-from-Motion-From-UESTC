// 定义图像文件路径和保存结果的路径
//#define INIT_IMG_PATH1 "test_img\\images\\100_7103.jpg"
//#define INIT_IMG_PATH2 "test_img\\images\\100_7104.jpg"
#define INIT_IMG_PATH1 "test_img\\First stage\\B25.jpg"
#define INIT_IMG_PATH2 "test_img\\First stage\\B24.jpg"
#define PLY_SAVE_PATH "test_img\\results\\output.ply"

#include "Includes.h"
#include "Images.h"
#include "Constructor.h"

//const Mat K = (Mat_<double>(3, 3) << 2905.88, 0, 1416, 0, 2905.88, 1064, 0, 0, 1);
const Mat K = (Mat_<double>(3, 3) << 719.5459, 0, 0, 0, 719.5459, 0, 0, 0, 1);

//const vector<string> sub_image_paths = { /*"test_img\\images\\100_7100.jpg", "test_img\\images\\100_7101.jpg", "test_img\\images\\100_7102.jpg",*/ /*"test_img\\images\\100_7103.jpg", "test_img\\images\\100_7104.jpg",*/ "test_img\\images\\100_7105.jpg", "test_img\\images\\100_7106.jpg", "test_img\\images\\100_7107.jpg", "test_img\\images\\100_7108.jpg", "test_img\\images\\100_7109.jpg"/*, "test_img\\images\\100_7110.jpg"*/ };


const vector<string> sub_image_paths = { "test_img\\First stage\\B23.jpg", "test_img\\First stage\\B22.jpg", "test_img\\First stage\\B21.jpg" };



struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d& observation)
        : observation(observation)
    {
    }

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }
};


void bundle_adjustment(
    Mat& intrinsic,
    vector<Mat>& extrinsics,
    vector<Images>& subImageBag,
    //vector<vector<int>>& correspond_struct_idx,
    //vector<vector<KeyPoint>>& key_points_for_all,
    vector<Point3d>& structure
)
{
    ceres::Problem problem;

    // load extrinsics (rotations and motions)
    for (size_t i = 0; i < extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
    }
    // fix the first camera.
    problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
    for (size_t img_idx = 0; img_idx < subImageBag.size(); ++img_idx)
    {
        vector<int>& point3d_ids = subImageBag[img_idx].correspond_struct_idx;
        vector<KeyPoint>& key_points = subImageBag[img_idx].keyPoints;
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;

            Point2d observed = key_points[point_idx].pt;
            // 模板參數中，第一個爲代價函數的類型，第二個爲代價的維度，剩下三個分別爲代價函數第一第二還有第三個參數的維度
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

            problem.AddResidualBlock(
                cost_function,
                loss_function,
                intrinsic.ptr<double>(),            // Intrinsic
                extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
                &(structure[point3d_id].x)          // Point in 3D space
            );
        }
    }

    // Solve BA
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }
}

void initConstruction(vector<Images>& initImages, vector<Point3d>& all_reconstructed_points, vector<Vec3b>& all_points_colors)
{
    
    initImages.push_back(*(new Images(INIT_IMG_PATH1)));
    initImages.push_back(*(new Images(INIT_IMG_PATH2)));

    vector<DMatch> matches;
    initImages[0].matchFeatures(initImages[1], matches);

    vector<uchar> mask;
    Constructor::findCamera(K, initImages[0].matchedPoints, initImages[1].matchedPoints, initImages[1].R, initImages[1].T, mask);
    initImages[0].R = Mat::eye(3, 3, CV_64FC1);
    initImages[0].T = Mat::zeros(3, 1, CV_64FC1);
    all_reconstructed_points = Constructor::pointsReconstruct(K, initImages[0].R, initImages[0].T, initImages[1].R, initImages[1].T, initImages[0].matchedPoints, initImages[1].matchedPoints);
    
    initImages[1].findColor();
    for (int i = 0; i < initImages[1].colors.size(); i++)
    {
        all_points_colors.push_back(initImages[1].colors[i]);
    }


    // 根据mask来记录初始两张图各个点和点云的关系
    int idx = 0;
    for (int i = 0; i < matches.size(); i++)
    {
        if (mask[i])
        {
            initImages[0].correspond_struct_idx[matches[i].queryIdx] = idx;
            initImages[1].correspond_struct_idx[matches[i].trainIdx] = idx;
            idx++;
        }
    }


    //Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
    //vector<Mat> extrinsics;
    //for (size_t i = 0; i < initImages.size(); ++i)
    //{
    //    Mat extrinsic(6, 1, CV_64FC1);
    //    Mat r;
    //    Rodrigues(initImages[i].R, r);

    //    r.copyTo(extrinsic.rowRange(0, 3));
    //    initImages[i].T.copyTo(extrinsic.rowRange(3, 6));

    //    extrinsics.push_back(extrinsic);
    //}

    //bundle_adjustment(intrinsic, extrinsics, initImages, all_reconstructed_points);
}

void addImageConstruction(vector<Images>& subImageBag, vector<Point3d>& all_reconstructed_points, vector<Vec3b>& all_points_colors)
{
    for (int i = 1; i < subImageBag.size(); i++)
    {
        cout << i << endl;
        vector<DMatch> matches;
        subImageBag[i - 1].matchFeatures(subImageBag[i], matches);

        subImageBag[i].getObjPointsAndImgPoints(matches, all_reconstructed_points, subImageBag[i - 1]);

        // 只是为了进行RANSAC筛选匹配点和获取mask
        vector<uchar> mask;
        Mat discardR, discardT;
        Constructor::findCamera(K, subImageBag[i - 1].matchedPoints, subImageBag[i].matchedPoints, discardR, discardT, mask);


        solvePnPRansac(subImageBag[i].object_points, subImageBag[i].image_points, K, noArray(), subImageBag[i].R, subImageBag[i].T);
        Rodrigues(subImageBag[i].R, subImageBag[i].R);

        vector<Point3d> new_restructure_points;
        new_restructure_points = Constructor::pointsReconstruct(K, subImageBag[i - 1].R, subImageBag[i - 1].T, subImageBag[i].R, subImageBag[i].T, subImageBag[i - 1].matchedPoints, subImageBag[i].matchedPoints);

        subImageBag[i].findColor();

        // 记录初始两张图各个点和点云的关系
        int idx = 0;
        for (int k = 0; k < matches.size(); k++)
        {
            if (mask[k])
            {
                subImageBag[i - 1].correspond_struct_idx[matches[k].queryIdx] = all_reconstructed_points.size() + idx;
                subImageBag[i].correspond_struct_idx[matches[k].trainIdx] = all_reconstructed_points.size() + idx;
                idx++;
            }

        }


       

        for (int k = 0; k < new_restructure_points.size(); k++)
        {
            all_reconstructed_points.push_back(new_restructure_points[k]);
            all_points_colors.push_back(subImageBag[i].colors[k]);
        }
    }

}

int main()
{
    try
    {
        vector<Images> initImages;
        vector<Point3d> all_reconstructed_points;
        vector<Vec3b> all_points_colors;

        initConstruction(initImages, all_reconstructed_points, all_points_colors);



        vector<Images> subImageBag;
        subImageBag.push_back(initImages[1]);
        for (auto& image_path : sub_image_paths)
        {
            subImageBag.push_back(Images(image_path));
        }

        addImageConstruction(subImageBag, all_reconstructed_points, all_points_colors);


        //--------------------------------------------------------------------------------------------------------------------
        //Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
        //vector<Mat> extrinsics;
        //for (size_t i = 0; i < subImageBag.size(); ++i)
        //{
        //    Mat extrinsic(6, 1, CV_64FC1);
        //    Mat r;
        //    Rodrigues(subImageBag[i].R, r);

        //    r.copyTo(extrinsic.rowRange(0, 3));
        //    subImageBag[i].T.copyTo(extrinsic.rowRange(3, 6));

        //    extrinsics.push_back(extrinsic);
        //}

        //bundle_adjustment(intrinsic, extrinsics, subImageBag, all_reconstructed_points);
        //--------------------------------------------------------------------------------------------------------------------
        
        
        
        // 手动输出点云ply文件
        std::ofstream plyFile(PLY_SAVE_PATH);

        // ply的头部信息
        plyFile << "ply\n";
        plyFile << "format ascii 1.0\n";
        plyFile << "element vertex " << all_reconstructed_points.size() << "\n";
        plyFile << "property float x\n";
        plyFile << "property float y\n";
        plyFile << "property float z\n";
        plyFile << "property uchar blue\n";
        plyFile << "property uchar green\n";
        plyFile << "property uchar red\n";
        plyFile << "end_header\n";

        // 写入点云数据
        for (int i = 0; i < all_reconstructed_points.size(); ++i)
        {
            cv::Vec3b color = all_points_colors[i];
            cv::Point3f point = all_reconstructed_points[i];
            plyFile << point.x << " " << point.y << " " << point.z << " "
                << static_cast<int>(color[0]) << " "
                << static_cast<int>(color[1]) << " "
                << static_cast<int>(color[2]) << std::endl;
        }

        plyFile.close();
        return 0;
    }
    catch (Exception e)
    {
        cout << e.msg << endl;
    }
    
}
