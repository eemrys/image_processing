#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


void borders_visualization(const std::vector<cv::Point2i>& points, 
	const cv::Mat& img, size_t idx) {
		cv::Mat res;
		img.copyTo(res);
		for (ptrdiff_t i_point = 0; i_point < points.size(); i_point += 1) {
			cv::line(res, points[i_point], points[(i_point + 1) % points.size()],
				cv::Scalar(0, 0, 255), 3);
		}
		cv::imwrite("lab05.b" + std::to_string(idx + 1) + ".png", res);
}

cv::Mat rotate_img(cv::Mat& img, const cv::Mat& scan, std::vector<cv::Point2i>& img_pts, const std::vector<cv::Point2i>& target_pts, size_t idx) {
	cv::Mat homography = cv::findHomography(img_pts, target_pts);
	cv::Mat img_homography;
	cv::warpPerspective(img, img_homography, homography, scan.size());
	borders_visualization(img_pts, img, idx);
	cv::imwrite("lab05.h" + std::to_string(idx + 1) + ".png", img_homography);
	return img_homography;
}


double connectivity_component_analysis(const cv::Mat1b& img, const cv::Mat1b& etalon) {
	cv::Mat1b inv_img;
	cv::bitwise_not(img, inv_img);
	cv::Mat labels, stats, centroids;
	uint32_t components_number = cv::connectedComponentsWithStats(inv_img, labels, stats, centroids);
	cv::Mat1b inv_et;
	cv::bitwise_not(etalon, inv_et);
	cv::Mat labels_et, stats_et, centroids_et;
	uint32_t components_number_et = cv::connectedComponentsWithStats(inv_et, labels_et, stats_et, centroids_et);
	int32_t fp = components_number, fn = 0;
	double tp = 0;
	for (uint32_t i_component = 1; i_component < components_number_et; i_component += 1) {
		cv::Mat1b component;
		int32_t comp_left = stats_et.at<int32_t>(i_component, cv::CC_STAT_LEFT);
		int32_t comp_top = stats_et.at<int32_t>(i_component, cv::CC_STAT_TOP);
		int32_t comp_width = stats_et.at<int32_t>(i_component, cv::CC_STAT_WIDTH);
		int32_t comp_height = stats_et.at<int32_t>(i_component, cv::CC_STAT_HEIGHT);
		cv::Rect comp_rect = cv::Rect(comp_left, comp_top, comp_width, comp_height);
		cv::compare(labels_et(comp_rect), i_component, component, cv::CMP_EQ);
		cv::Mat1b res_and;
		cv::bitwise_and(component, inv_img(comp_rect), res_and);
		int32_t res_area = cv::countNonZero(res_and);
		if (res_area == 0) {
			fn += 1;
			continue;
		}
		fp -= 1;
		tp += static_cast<double>(res_area) / stats_et.at<int32_t>(i_component, cv::CC_STAT_AREA);
	}
	fp = abs(fp);
	double recall = tp / (tp + fn);
	double prec = tp / (tp + fp);
	return 2 * prec * recall / (prec + recall);
}

double score(const cv::Mat1b& img, const cv::Mat1b& etalon) {
	return (connectivity_component_analysis(img, etalon) + connectivity_component_analysis(etalon, img)) / 2;
}

void diviation_illustration(const cv::Mat1b& img, const cv::Mat1b& standard, const size_t num, const size_t idx) {
	cv::Mat3b dev_img;
	cv::cvtColor(img, dev_img, cv::COLOR_GRAY2RGB);
	dev_img.setTo(cv::Vec3b(0, 0, 255), img > standard);
	dev_img.setTo(cv::Vec3b(0, 255, 0), img < standard);
	cv::imwrite("lab05.e" + std::to_string(idx+1) + ".version" + std::to_string(num) + ".png", dev_img);
}


void filtration(cv::Mat& img) {
	cv::Mat1b inv_img;
	cv::bitwise_not(img, inv_img);
	cv::Mat labels, stats, centroids;
	uint32_t components_number = cv::connectedComponentsWithStats(inv_img, labels, stats, centroids);
	int32_t margin = 100, small_comp_size = 50;
	std::set<int32_t> bad_components;
	for (uint32_t i_component = 0; i_component < components_number; i_component += 1) {
		if (stats.at<int32_t>(i_component, cv::CC_STAT_AREA) < small_comp_size
			|| stats.at<int32_t>(i_component, cv::CC_STAT_LEFT) < margin
			|| stats.at<int32_t>(i_component, cv::CC_STAT_TOP) < margin
			|| stats.at<int32_t>(i_component, cv::CC_STAT_LEFT)
			+ stats.at<int32_t>(i_component, cv::CC_STAT_WIDTH) > img.cols - margin
			|| stats.at<int32_t>(i_component, cv::CC_STAT_TOP)
			+ stats.at<int32_t>(i_component, cv::CC_STAT_HEIGHT) > img.rows - margin) {
			bad_components.insert(i_component);
		}
	}

	for (uint32_t i_row = 0; i_row < img.rows; i_row += 1) {
		for (uint32_t i_col = 0; i_col < img.cols; i_col += 1) {
			if (bad_components.count(labels.at<int32_t>(i_row, i_col))) {
				img.at<uint8_t>(i_row, i_col) = 255;
			}
		}
	}
}

double process(cv::Mat& img, const cv::Mat1b& etalon, const size_t idx, const size_t num, 
	const uint8_t param1, const uint8_t param2, const uint8_t window_size) {

	cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, param1, param2);
	cv::GaussianBlur(img, img, cv::Size(window_size, window_size), 0, 0);
	cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, param1, param2);
	filtration(img);

	double score0 = score(img, etalon);
	cv::imwrite("lab05.v" + std::to_string(idx + 1) + ".version" + std::to_string(num) + ".png", img);

	diviation_illustration(img, etalon, num, idx);
	std::cout << "Result for the image number " + std::to_string(idx+1) + ": " <<  score0 << std::endl;

	return score0;
}


int main() {
	const uint32_t num_img = 5;
	cv::Mat scan_img = cv::imread("../data/lab05.scan.png");
	cvtColor(scan_img, scan_img, cv::COLOR_BGR2GRAY);
	cv::Mat photos[num_img];
	photos[0] = cv::imread("../data/lab05.photo1.jpg");
	cvtColor(photos[0], photos[0], cv::COLOR_BGR2GRAY);

	photos[1] = cv::imread("../data/lab05.photo2.jpg");
	cvtColor(photos[1], photos[1], cv::COLOR_BGR2GRAY);

	photos[2] = cv::imread("../data/lab05.photo3.jpg");
	cvtColor(photos[2], photos[2], cv::COLOR_BGR2GRAY);

	photos[3] = cv::imread("../data/lab05.photo4.jpg");
	cvtColor(photos[3], photos[3], cv::COLOR_BGR2GRAY);

	photos[4] = cv::imread("../data/lab05.photo5.jpg");
	cvtColor(photos[4], photos[4], cv::COLOR_BGR2GRAY);


	std::vector<cv::Point2i> scan_special_pts = { {0, 0}, {scan_img.cols - 1, 0},{scan_img.cols - 1, scan_img.rows - 1},{0, scan_img.rows - 1}};
	std::vector<std::vector<cv::Point2i>> special_pts = {{{380,484},{2209,511},{2306,3215},{305,3232}},{{400,611}, {2009,609},{2031,2938} ,{379, 2918}}, {{534,1966},{449,294}, {3175,61},{3088,2165}},{{612,1996},{694,488}, {3020,203},{3127,2168}},{{2839,450}, {2786,2169},{472,2060}, {469,454}}};

	cv::Mat rotated_img[num_img];

	cv::Mat1b etalon;
	cv::adaptiveThreshold(scan_img, etalon, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 19, 10);
	filtration(etalon);

	for (ptrdiff_t i = 0; i < num_img; i = i + 1) {
		rotated_img[i] = rotate_img(photos[i], scan_img, special_pts[i], scan_special_pts, i);
	}

	double average_score1 = 0.0;
	std::cout << "Scores before changing:" << std::endl;
	for (ptrdiff_t i = 0; i < num_img; i = i + 1) {
		average_score1 += process(rotated_img[i], etalon, i, 1, 19, 10, 3);
	}
	std::cout << "Average score: " << average_score1/ num_img << std::endl << std::endl;

	double average_score2 = 0.0;
	std::cout << "Scores after changing:" << std::endl;
	for (ptrdiff_t i = 0; i < num_img; i = i + 1) {
		average_score2 += process(rotated_img[i], etalon, i, 2, 29, 6, 5);
	}
	std::cout << "Average score: " << average_score2/ num_img << std::endl;

	return 0;
}
