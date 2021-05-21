#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // пункт 1
    cv::Mat img(60, 768, CV_8UC1);
    img = 0;
    cv::Rect2d rc = {0, 0, 3, 60 };
    for (int i = 0; i < 256; i++) {
        cv::rectangle(img, rc, { static_cast<double>(i) }, cv::FILLED);
        rc.x += rc.width;
    }
    
    // пункт 2
    float gamma = 2.3;
    cv::TickMeter tm;
    
    tm.start();
    cv::Mat src;
    img.convertTo(src, CV_64FC1, 1.0f / 255.0f);
    cv::Mat dst;
    cv::pow(src, gamma, dst);
    cv::Mat res1;
    dst.convertTo(res1, CV_8UC1, 255);
    tm.stop();
    // пункт 5
    std::cout << "Method #1: " << tm.getTimeSec() << std::endl;
        
    // пункт 3
    tm.start();
    cv::Mat lookUpTable(1, 256, CV_8UC1);
    uchar* p = lookUpTable.ptr();
    for(int i = 0; i < 256; i++) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat res2 = img.clone();
    cv::LUT(img, lookUpTable, res2);
    tm.stop();
    // пункт 5
    std::cout << "Method #2: " << tm.getTimeSec() << std::endl;
        
    // пункт 4
    cv::Mat output1;
    cv::vconcat(img, res1, output1);
    cv::Mat output2;
    cv::vconcat(output1, res2, output2);
    cv::imshow("result", output2);
    cv::waitKey();
    
    cv::imwrite("lab01.png", output2);
    
    return 0;
}
