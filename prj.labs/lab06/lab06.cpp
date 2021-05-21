#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


double euclidean_distance(cv::Point a, cv::Point b) {
	return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double points_estimation(const std::vector<cv::Point>& found, const std::vector<cv::Point>& etalon) {
	double score = 0.0;
	double max_len = std::max(euclidean_distance(etalon[0], etalon[2]), euclidean_distance(etalon[1], etalon[3]));
	for (ptrdiff_t i = 0; i < 4; i = i + 1) {
		score += (1.0 - euclidean_distance(found[i], etalon[i]) / max_len);
	}
	return score / 4;
}

double intersection_over_union(const cv::Mat& orig, const std::vector<cv::Point>& found, 
	const std::vector<cv::Point>& etalon_pts) {
	cv::Mat etalon = cv::Mat::zeros(cv::Size(orig.cols, orig.rows), CV_8U);
	cv::Mat my_result = cv::Mat::zeros(cv::Size(orig.cols, orig.rows), CV_8U);
	cv::fillPoly(etalon, etalon_pts, 255, 150, 0);
	cv::fillPoly(my_result, found, 255, 150, 0);
	std::vector<cv::Point> FP;
	std::vector<cv::Point> FN;
	double intersection = 0.0;
	double union_ = 0.0;
	for (ptrdiff_t i = 0; i < orig.cols; i++) {
		for (ptrdiff_t j = 0; j < orig.rows; j++) {
			if (etalon.at<uchar>(j, i) == 255 || my_result.at<uchar>(j, i) == 255) {
				union_ += 1.0;
			}
			if (etalon.at<uchar>(j, i) == 255 && my_result.at<uchar>(j, i) == 255) {
				intersection += 1.0;
			}
		}
	}
	double res = intersection / union_;
	cv::polylines(orig, etalon_pts, true, cv::Scalar(255, 0, 8, 0), 2, 150, 0);
	cv::polylines(orig, found, true, cv::Scalar(0, 0, 255, 0), 2, 150, 0);
	return res;
}



double process_image(cv::Mat& orig, const std::vector<cv::Point>& etalon_pts, const size_t idx, double& point_res) {
	std::string photo_name = std::to_string(idx+1);
	cv::Mat processed_img;
	cv::GaussianBlur(orig, processed_img, cv::Size(15, 15), 0, 0);
	cv::medianBlur(processed_img, processed_img, 51);
	cv::adaptiveThreshold(processed_img, processed_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);
	cv::GaussianBlur(processed_img, processed_img, cv::Size(7, 7), 0, 0);
	cv::GaussianBlur(processed_img, processed_img, cv::Size(15, 15), 0, 0);
	cv::imwrite("lab06.processed" + photo_name + ".png", processed_img);
	cv::Mat edges;
	cv::Canny(processed_img, edges, 100, 200, 5, true);
	cv::imwrite("lab06.canny_edges" + photo_name + ".png", edges);
	std::vector<cv::Vec4i> lines;
	std::vector<cv::Point> points;
	cv::HoughLinesP(edges, lines, 2, CV_PI / 180, 200, 200, 300);
	cv::cvtColor(orig, orig, cv::COLOR_GRAY2BGR);
	cv::Point RU(lines[0][0], lines[0][1]);
	cv::Point LU(lines[0][0], lines[0][1]);
	cv::Point LL(lines[0][0], lines[0][1]);
	cv::Point RL(lines[0][0], lines[0][1]);
	cv::Point left_top(0, 0);
	cv::Point right_top(orig.cols, 0);
	cv::Point right_bot(orig.cols, orig.rows);
	cv::Point left_bot(0, orig.rows);
	cv::Mat orig_hough;
	orig.copyTo(orig_hough);
	for (ptrdiff_t i = 0; i < lines.size(); i++) {
		cv::Vec4i curr = lines[i];
		cv::line(orig_hough, cv::Point(curr[0], curr[1]), cv::Point(curr[2], curr[3]), cv::Scalar(0, 0, 255, 0), 1, 8);
		if (euclidean_distance(left_top, LU) 
		> euclidean_distance(left_top, cv::Point(curr[0], curr[1]))) {
			LU = cv::Point(curr[0], curr[1]);
		}
		if (euclidean_distance(left_top, LU) 
		> euclidean_distance(left_top, cv::Point(curr[2], curr[3]))) {
			LU = cv::Point(curr[2], curr[3]);
		}
		if (euclidean_distance(right_top, RU) 
		> euclidean_distance(right_top, cv::Point(curr[0], curr[1]))) {
			RU = cv::Point(curr[0], curr[1]);
		}
		if (euclidean_distance(right_top, RU) 
		> euclidean_distance(right_top, cv::Point(curr[2], curr[3]))) {
			RU = cv::Point(curr[2], curr[3]);
		}
		if (euclidean_distance(right_bot, RL)	
		> euclidean_distance(right_bot, cv::Point(curr[0], curr[1]))) {
			RL = cv::Point(curr[0], curr[1]);
		}
		if (euclidean_distance(right_bot, RL) 
		> euclidean_distance(right_bot, cv::Point(curr[2], curr[3]))) {
			RL = cv::Point(curr[2], curr[3]);
		}
		if (euclidean_distance(left_bot, LL) 
		> euclidean_distance(left_bot, cv::Point(curr[0], curr[1]))) {
			LL = cv::Point(curr[0], curr[1]);
		}
		if (euclidean_distance(left_bot, LL) 
		> euclidean_distance(left_bot, cv::Point(curr[2], curr[3]))) {
			LL = cv::Point(curr[2], curr[3]);
		}
	}
	cv::imwrite("lab06.hough" + photo_name + ".png", orig_hough);
	std::vector<cv::Point> pts = { LU, RU, RL, LL };
	point_res = points_estimation(pts, etalon_pts);
	double IoU = intersection_over_union(orig, pts, etalon_pts);
	cv::imwrite("lab06.result" + photo_name + ".png", orig);
	std::cout << "Results for photo number " << photo_name << std::endl;
	std::cout << "Score for found corners = " << point_res << std::endl;
	std::cout << "Intesection over union = " << IoU << std::endl;
	std::cout << std::endl;
	return IoU;
}


int main() {
	cv::Mat scan_img = cv::imread("../data/lab05.scan.png");
	const uint32_t num_img = 5;
	cv::Mat photos[num_img];
	photos[0] = cv::imread("../data/lab05.photo1.jpg", cv::IMREAD_GRAYSCALE);
	photos[1] = cv::imread("../data/lab05.photo2.jpg", cv::IMREAD_GRAYSCALE);
	photos[2] = cv::imread("../data/lab05.photo3.jpg", cv::IMREAD_GRAYSCALE);
	photos[3] = cv::imread("../data/lab05.photo4.jpg", cv::IMREAD_GRAYSCALE);
	photos[4] = cv::imread("../data/lab05.photo5.jpg", cv::IMREAD_GRAYSCALE);

	std::vector<std::vector<cv::Point2i>> special_pts = {{{380,484},{2209,511},{2306,3215},{305,3232}},
									{{400,611}, {2009,609},{2031,2938} ,{379, 2918}}, 
									{{534,1966},{449,294}, {3175,61},{3088,2165}}, 
									{{612,1996},{694,488}, {3020,203},{3127,2168}},
									{{2839,450}, {2786,2169},{472,2060}, {469,454}}};

	cv::Mat edges[num_img];
	double average_score_iou = 0.0;
	double average_score_pts = 0.0;
	for (ptrdiff_t i = 0; i < num_img; i = i + 1) {
		double pts_score = 0.0;
		average_score_iou += process_image(photos[i], special_pts[i], i, pts_score);
		average_score_pts += pts_score;
	}

	std::cout << "Average score for points: " << average_score_pts / num_img << std::endl;
	std::cout << "Average intersiction over union score: " << average_score_iou / num_img;
	

	return 0;
}