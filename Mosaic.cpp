#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/types.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

int main() {
	vector<cv::String> fn;
	glob("C:\\Users\\jun17\\source\\repos\\Project3\\Project3\\image_db\\image_db\\*.jpg", fn, false);
	Mat src_before = imread("C:/folder/lena.png", IMREAD_COLOR);
	Mat images;
	float B = 0, G = 0, R = 0;
	float B_m = 0, G_m = 0, R_m = 0;
	size_t count = fn.size();

	Mat src;
	resize(src_before, src, Size(832, 832), 0, 0);
	if (src.empty()) { return -1; }

	Mat samples(src.rows * src.cols, 3, CV_32F);

	for (int y = 0; y < src.cols; y++)
		for (int x = 0; x < src.rows; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];

	int clusterCount = 15;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT |
		TermCriteria::EPS, 10000, 0.0001),attempts, KMEANS_PP_CENTERS, centers);

	Mat dst(src.size(), src.type());
	for(int y=0; y<src.rows; y++)
		for (int x = 0; x < src.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src.rows, 0);
			dst.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			dst.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			dst.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}

	for (int k = 0; k < dst.cols; k+=32) {
		for (int h = 0; h < dst.rows; h+=32) {
			//inside rect
			//printf("k:%d ,h:%d\n", k, h);
			for (int j = k; j < k + 32; j++) {
				for (int i = h; i < h + 32; i++) {
					B += dst.at<Vec3b>(j, i)[0];
					G += dst.at<Vec3b>(j, i)[1];
					R += dst.at<Vec3b>(j, i)[2];
				}
			}
			B = B / 1024;
			G = G / 1024;
			R = R / 1024;

			//printf("B: %lf G: %lf R:%lf \n ", B, G, R);

			for (size_t i = 0; i < 25000; i++)
			{
				images = imread(fn[i]);
				for (int y = k; y < k+32; y++) {
					for (int x = h; x < h+32; x++) {
						B_m += images.at<Vec3b>(y-k, x-h)[0];
						G_m += images.at<Vec3b>(y-k, x-h)[1];
						R_m += images.at<Vec3b>(y-k, x-h)[2];
					}
				}
				B_m = B_m / 1024;
				G_m = G_m / 1024;
				R_m = R_m / 1024;
				//printf("B_: %lf G_: %lf R_:%lf B: % lf G : % lf R : % lf \n ", B, G, R, abs(B_m-B), abs(G_m-G), abs(R_m-R));
				if (abs(B_m - B )< 10 && abs(G_m - G )< 10 && abs(R_m - R )< 10) {
					for (int y = k; y < k + 32; y++) {
						for (int x = h; x < h + 32; x++) {
							dst.at<Vec3b>(y, x)[0] = images.at<Vec3b>(y - k, x - h)[0];
							dst.at<Vec3b>(y, x)[1] = images.at<Vec3b>(y - k, x - h)[1];
							dst.at<Vec3b>(y, x)[2] = images.at<Vec3b>(y - k, x - h)[2];
						}
					}
					break;
				}
			}
			B = 0, G = 0, R = 0;
						
			//cv::putText(src, "3", Point(k + 5, h + 5), 0.5, 0.2, Scalar(255, 255, 255));
		}
	}

	imshow("image", dst);
	
	waitKey(0);
	return 0;

}