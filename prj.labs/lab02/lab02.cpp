#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

cv::Mat show_channel(cv::Mat channel_one, cv::Mat channel_two, cv::Mat channel_three) {
    cv::Mat result;
    std::vector<cv::Mat> channels;
    channels.push_back(channel_one);
    channels.push_back(channel_two);
    channels.push_back(channel_three);
    cv::merge(channels, result);
    return result;
}

cv::Mat create_collage(cv::Mat img, std::vector<cv::Mat> channels) {
    cv::Mat black, red, green, blue, top_row, bottom_row, output;
    black = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);
    red = show_channel(black, black, channels[2]);
    green = show_channel(black, channels[1], black);
    blue = show_channel(channels[0], black, black);
    cv::hconcat(img, red, top_row);
    cv::hconcat(green, blue, bottom_row);
    cv::vconcat(top_row, bottom_row, output);
    return output;
}

cv::Mat create_hist(std::vector<cv::Mat> channels) {
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range = {range};
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, &hist_range, true, false);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, &hist_range, true, false);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, &hist_range, true, false);
    int hist_w = 400, hist_h = 200;
    int bin_w = cvRound((double) hist_w / hist_size);
    cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
    
    cv::normalize(b_hist, b_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for (int i = 1; i < hist_size; i++) {
        cv::line(hist_image, cv::Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))),
                  cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
                  cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(hist_image, cv::Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1))),
                  cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
                  cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(hist_image, cv::Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
                  cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
                  cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    return hist_image;
}


int main() {

    cv::Mat image = cv::imread("../data/cross_0256x0256.png");

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(25);

    cv::imwrite("cross_0256x0256_025.jpg", image, compression_params);
    
    cv::Mat image_jpg = cv::imread("cross_0256x0256_025.jpg");

    cv::Mat mosaic, mosaic_jpg, hist, hist_jpg, hists_img;
    std::vector<cv::Mat> channels(3), channels_jpg(3);
    
    cv::split(image, channels);
    mosaic = create_collage(image, channels);
    cv::imwrite("cross_0256x0256_png_channels.png", mosaic);
    
    cv::split(image_jpg, channels_jpg);
    mosaic_jpg = create_collage(image_jpg, channels_jpg);
    cv::imwrite("cross_0256x0256_jpg_channels.png", mosaic_jpg);
    
    hist = create_hist(channels);
    hist_jpg = create_hist(channels_jpg);
    cv::vconcat(hist, hist_jpg, hists_img);
    cv::imwrite("cross_0256x0256_hists.png", hists_img);
    
    return 0;
}
