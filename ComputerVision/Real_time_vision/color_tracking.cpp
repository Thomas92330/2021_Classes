#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int  width = 320, height = 240;
Mat frame, proc_img;
VideoCapture capture;

Mat struct_element;
vector<vector<Point>> contours, hull;

string windowName1 = "1- Color Tracking", windowName2 = "2- Red Color", windowName3 = "3- Morphology (Close)",
windowName4 = "4- Canny", windowName5 = "5- Contours", windowName6 = "6- Large Contour",
windowName7 = "7- Convex Hull";

int slider_pos = 0, slider_pos_old = 0;
int num_frames = 0;

vector<vector<Point>> findLargeContours(vector<vector<Point> > contours);
Mat getColorRangeImage(Mat);

Mat getColorRangeImage(Mat hsv_img)
{
	Mat range_img = Mat(hsv_img.size(), CV_8UC1);

	inRange(hsv_img, Scalar(170, 160, 60), Scalar(180, 255, 255), range_img);

	return range_img;
}

vector<vector<Point>> findLargeContours(vector<vector<Point>> contours)
{
	vector<vector<Point>> currentContour = contours;
	vector<vector<Point>> largeContours;
	vector<Point> approx;

	double area;

	for (int i = 0; i < currentContour.size(); i++)
	{
		area = fabs(contourArea(currentContour.at(i)));

		if (area > 500.0)	
			largeContours.push_back(currentContour.at(i));
	}

	return largeContours;
}

void on_trackbar(int, void*)
{
	if (abs(slider_pos - slider_pos_old) > 1)
	{
		capture.set(CAP_PROP_POS_FRAMES, slider_pos);
	}

	if (slider_pos == num_frames || frame.empty())
	{
		destroyWindow(windowName2); destroyWindow(windowName3); destroyWindow(windowName4);
		destroyWindow(windowName5); destroyWindow(windowName6); destroyWindow(windowName7);

		slider_pos = 0;
		createTrackbar("frames", windowName1, &slider_pos, 0, on_trackbar);
		setTrackbarPos("frames", windowName1, slider_pos);
		capture.set(CAP_PROP_POS_FRAMES, slider_pos);
		capture >> frame;
		resize(frame, frame, Size(width, height));

		string text = "hit any key (except ESC) to restart...";
		Size textSize = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.45, 1, 0);
		Point textOrigin((frame.cols - textSize.width) / 2, (frame.rows - textSize.height) / 2);
		rectangle(frame, textOrigin + Point(0, 3), textOrigin + Point(textSize.width, -textSize.height), Scalar::all(255), FILLED);
		putText(frame, text, textOrigin, FONT_HERSHEY_COMPLEX, 0.45, Scalar(0, 0, 255), 1, LINE_AA);

		imshow(windowName1, frame);

		if (waitKey() == 27) exit(0);

		createTrackbar("frames", windowName1, &slider_pos, num_frames, on_trackbar);
		setTrackbarPos("frames", windowName1, slider_pos);
	}
}

int main()
{
	capture.open("videoTuto2.MOV");
	if (!capture.isOpened())
	{
		cerr << "Couldn't open the video" << endl;
		return 1;
	}

	namedWindow(windowName1, WINDOW_AUTOSIZE);

	num_frames = (int)capture.get(CAP_PROP_FRAME_COUNT);

	if (num_frames)
		createTrackbar("frames", windowName1, &slider_pos, num_frames, on_trackbar);

	struct_element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(3, 3));

	while (true)
	{
		capture >> frame;

		setTrackbarPos("frames", windowName1, slider_pos);
		slider_pos_old = slider_pos;
		slider_pos++;
		
		resize(frame, frame, Size(width, height));

		imshow(windowName1, frame);

		cvtColor(frame, proc_img, COLOR_BGR2HSV);

		proc_img = getColorRangeImage(proc_img);
		imshow(windowName2, proc_img);

		morphologyEx(proc_img, proc_img, MORPH_CLOSE, struct_element);
		imshow(windowName3, proc_img);

		Canny(proc_img, proc_img, 100, 200, 3);
		imshow(windowName4, proc_img);

		dilate(proc_img, proc_img, struct_element);

		findContours(proc_img, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
		Mat cont_img = Mat::zeros(Size(width, height), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
			drawContours(cont_img, contours, i, CV_RGB(255, 100, 0), 2);
		imshow(windowName5, cont_img);

		contours = findLargeContours(contours);
		Mat largContImg = Mat::zeros(Size(width, height), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
			drawContours(largContImg, contours, i, CV_RGB(255, 50, 0), 2);
		imshow(windowName6, largContImg);

		hull.resize(contours.size());
		for (int i = 0; i < contours.size(); i++)
			convexHull(contours[i], hull[i]);
		Mat hull_img = Mat::zeros(Size(width, height), CV_8UC3);
		for (int i = 0; i < hull.size(); i++)
			drawContours(hull_img, hull, i, CV_RGB(0, 255, 0), 2);
		imshow(windowName7, hull_img);

		for (int i = 0; i < hull.size(); i++)
			drawContours(frame, hull, i, CV_RGB(255, 0, 0), -1);
		imshow(windowName1, frame);

		if (waitKey(20) == 27)	break;
	}

	return 0;
}