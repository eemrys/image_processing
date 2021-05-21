#include <opencv2/opencv.hpp>



void fast_gaussian_binarization(const cv::Mat1b& input, cv::Mat1b& output) {
	cv::Mat1b gaussian_img;
	int32_t window_size = 27;
	cv::GaussianBlur(input, gaussian_img, cv::Size(window_size, window_size), 0, 0);
	cv::Mat1b d;
	cv::absdiff(gaussian_img, input, d);
	cv::GaussianBlur(d, d, cv::Size(window_size, window_size), 0, 0);
	cv::Mat res1, res2;
	int32_t d0 = 10;
	cv::subtract(gaussian_img, input, res1);
	cv::add(d, d0, res2);
	cv::divide(res1, res2, res1);
	output = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
	int thres = 1;
	output.setTo(255, res1 < thres);
}


double connectivity_component_analysis(const cv::Mat1b& img, const cv::Mat1b& etalon) {
	cv::Mat1b inv_img;
	cv::bitwise_not(img, inv_img);
	cv::Mat labels, stats, centroids;
	uint32_t components_number = cv::connectedComponentsWithStats(
		inv_img, labels, stats, centroids);

	cv::Mat1b inv_et;
	cv::bitwise_not(etalon, inv_et);
	cv::Mat labels_et, stats_et, centroids_et;
	uint32_t components_number_et = cv::connectedComponentsWithStats(
		inv_et, labels_et, stats_et, centroids_et);

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
		tp += static_cast<double>(res_area)
			/ stats_et.at<int32_t>(i_component, cv::CC_STAT_AREA);
	}
	fp = abs(fp);
	double recall = tp / (tp + fn);
	double prec = tp / (tp + fp);
	return 2 * prec * recall / (prec + recall);
}


double score(const cv::Mat1b& img, const cv::Mat1b& etalon) {
	return (connectivity_component_analysis(img, etalon) + connectivity_component_analysis(etalon, img)) / 2;
}


void error_illustration(const cv::Mat1b& img, const cv::Mat1b& standard, const std::string method) {
	cv::Mat3b dev_img;
	cv::cvtColor(img, dev_img, cv::COLOR_GRAY2RGB);
	dev_img.setTo(cv::Vec3b(0, 0, 255), img > standard);
	dev_img.setTo(cv::Vec3b(0, 255, 0), img < standard);
	cv::imwrite("lab04.e" + method + ".png", dev_img);
}

void process(cv::Mat1b& img, const cv::Mat1b& etalon, const std::string method) {
	double score1 = score(img, etalon);

	cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0);
	double score2 = score(img, etalon);
	cv::imwrite("lab04.f" + method + ".png", img);

	cv::Mat1b inv_img;
	cv::bitwise_not(img, inv_img);

	cv::Mat labels, stats, centroids;
	uint32_t components_number = cv::connectedComponentsWithStats(
		inv_img, labels, stats, centroids);
	int32_t margin = 200, small_comp_size = 50;
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
	double score3 = score(img, etalon);
	cv::imwrite("lab04.v" + method + ".png", img);

	error_illustration(img, etalon, method);

	std::cout << "Results of the method " + method + ": " << std::endl;
	std::cout << "binarization - " << score1 << std::endl;
	std::cout << "filtering - " << score2 << std::endl;
	std::cout << "filtering components - " << score3 << std::endl;
}

int main() {
	cv::Mat img = cv::imread("lab04.src.jpg");
	cv::Mat etalon = cv::imread("etalon.png");
	resize(etalon, etalon, cv::Size(img.cols, img.rows));

	cvtColor(etalon, etalon, cv::COLOR_BGR2GRAY);

	cv::Mat g1;
	cvtColor(img, g1, cv::COLOR_BGR2GRAY);
	imwrite("lab04.g1.png", g1);

	cv::Mat1b b1;
	cv::adaptiveThreshold(g1, b1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 19, 10);
	imwrite("lab04.b1.png", b1);
	process(b1, etalon, "1");

	cv::Mat1b b2;
	fast_gaussian_binarization(g1, b2);
	cv::imwrite("lab04.b2.png", b2);
	process(b2, etalon, "2");

	return 0;
}
