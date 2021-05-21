#include <opencv2/opencv.hpp>
#include <cmath>

int main() {
    cv::Mat rgb = cv::imread("../data/cross_0256x0256.png");
    cv::imwrite("lab03_rgb.png", rgb);
    cv::Mat gre;
    cv::cvtColor(rgb, gre, cv::COLOR_BGR2GRAY);
    cv::imwrite("lab03_gre.png", gre);
    cv::Mat look_up_table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        look_up_table.at<uchar>(0, i) = cv::saturate_cast<uchar>(cos(i / 255.0) * 255);
    }
    cv::Mat rgb_res, gre_res;
    cv::LUT(rgb, look_up_table, rgb_res);
    cv::LUT(gre, look_up_table, gre_res);
    cv::imwrite("lab03_gre_res.png", gre_res);
    cv::imwrite("lab03_rgb_res.png", rgb_res);
    cv::Mat func_img(512, 512, CV_8UC1, 255);
    for (int i = 0; i < 256; i++) {
        func_img.at<uchar>((256 - look_up_table.at<uchar>(0, i))*2, i*2) = 0;
    }
    cv::imwrite("lab03_viz_func.png", func_img);
    return 0;
}
