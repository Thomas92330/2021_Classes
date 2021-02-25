#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

VideoCapture capture, overlayVideo;
Mat frame, trainImage;

Ptr<AKAZE> akaze;
Ptr<DescriptorMatcher> matcher;
vector<KeyPoint> kp_train, kp_query;
Mat desc_train, desc_query;

Mat akazeHomography;

vector<Point2f> sceneCorners; 
vector<Point2f> objectCorners;

vector<Point2f> previousSceneCorners; 
Mat previousFrame;
vector<uchar> status;
vector<float> err;
bool TRACKING = false;

void overlay() 
{
	Mat overlayFrame, warpedOverlayFrame;

	overlayVideo >> overlayFrame;
	resize(overlayFrame, overlayFrame, trainImage.size());

	if (overlayVideo.get(CAP_PROP_POS_FRAMES) == overlayVideo.get(CAP_PROP_FRAME_COUNT))
		overlayVideo.set(CAP_PROP_POS_FRAMES, 0);

	Mat trackingHomography = findHomography(objectCorners, sceneCorners);

	warpPerspective(overlayFrame, warpedOverlayFrame, trackingHomography, frame.size());

	vector<Point> vec_sceneCorners(sceneCorners.begin(), sceneCorners.end()); 
	vector<vector<Point>> poly_rec_object; 
	poly_rec_object.push_back(vec_sceneCorners);

	fillPoly(frame, poly_rec_object, Scalar(0));

	frame = warpedOverlayFrame + frame; 
}

int akazeTracker()
{
	double elapsedTime = getTickCount();

	akaze->detectAndCompute(frame, noArray(), kp_query, desc_query);

	vector<vector<DMatch>> matches;
	matcher->knnMatch(desc_query, desc_train, matches, 2);

	vector<KeyPoint> kp_query_good_match, kp_train_good_match;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.8 * matches[i][1].distance)
		{
			kp_query_good_match.push_back(kp_query[matches[i][0].queryIdx]);
			kp_train_good_match.push_back(kp_train[matches[i][0].trainIdx]);
		}
	}

	int numberOfGoodMatches;	
	numberOfGoodMatches = kp_query_good_match.size();
	if (numberOfGoodMatches < 4) return 0;

	vector<Point2f> pts_query_good_match, pts_train_good_match;
	for (int i = 0; i < numberOfGoodMatches; i++)
	{
		pts_query_good_match.push_back(kp_query_good_match[i].pt);
		pts_train_good_match.push_back(kp_train_good_match[i].pt);
	}

	Mat inlierMask = Mat(1, kp_query_good_match.size(), CV_8U);

	akazeHomography = findHomography(pts_train_good_match, pts_query_good_match, RANSAC, 2, inlierMask);
	
	vector<KeyPoint>  kp_query_inliers, kp_train_inliers;	
	vector<DMatch> inlierMatches;

	int numberOfInlierMatches = 0;
	if (!akazeHomography.empty())
	{
		for (int i = 0, j = 0; i < numberOfGoodMatches; i++)
		{
			if (inlierMask.at<uchar>(i))
			{
				kp_query_inliers.push_back(kp_query_good_match[i]);
				kp_train_inliers.push_back(kp_train_good_match[i]);
				inlierMatches.push_back(DMatch(j, j, 0));
				j++;
				numberOfInlierMatches++;
			}
		}

		Mat corres_frame;
		drawMatches(trainImage, kp_train_inliers, frame, kp_query_inliers, inlierMatches, corres_frame);

		elapsedTime = getTickCount() - elapsedTime;

		putText(corres_frame, format("Good Matches : %d", numberOfGoodMatches), Point(5, corres_frame.rows - 80),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
		putText(corres_frame, format("Inliers : %d", numberOfInlierMatches), Point(5, corres_frame.rows - 50),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
		putText(corres_frame, format("FPS : %.1f", getTickFrequency() / elapsedTime), Point(5, corres_frame.rows - 20),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);

		imshow("AKAZE Recognition", corres_frame);
	}

	return numberOfInlierMatches;
}

int main()
{
	capture.open("data/videoTuto5.avi");

	if (!capture.isOpened())
	{
		cerr << "Couldn't open the source video ..." << endl;
		return 1;
	}

	overlayVideo.open("data/Mercedes-C-Class.mp4");
	if (!overlayVideo.isOpened())
	{
		cerr << "Couldn't open the video ..." << endl;
		return 1;
	}

	akaze = AKAZE::create();
	matcher = DescriptorMatcher::create("BruteForce-Hamming");

	trainImage = imread("data/Mercedes-C-Class.png", IMREAD_UNCHANGED);
	if (trainImage.empty())
	{
		cerr << "Couldn't load the image ..." << endl;
		return 1;
	}
	resize(trainImage, trainImage, trainImage.size() / 2);
	objectCorners = { Point2f(0,0), Point2f(trainImage.cols, 0),
				  Point2f(trainImage.cols, trainImage.rows), Point2f(0, trainImage.rows) };

	akaze->detectAndCompute(trainImage, noArray(), kp_train, desc_train);

	while (true)
	{
		double elapsedTime = getTickCount();

		capture >> frame;
		if (frame.empty()) break;

		resize(frame, frame, Size(640, 480));
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

		if (TRACKING == true)
		{
			calcOpticalFlowPyrLK(previousFrame, grayFrame, previousSceneCorners, sceneCorners, status, err);
			
			overlay();

			if (norm(previousSceneCorners, sceneCorners, NORM_L2) > 3) TRACKING = false;

			swap(previousSceneCorners, sceneCorners);
			swap(previousFrame, grayFrame);
		}
		else
		{
			if (akazeTracker() > 20)
			{
				perspectiveTransform(objectCorners, sceneCorners, akazeHomography);

				overlay();

				swap(previousSceneCorners, sceneCorners);
				swap(previousFrame, grayFrame);

				TRACKING = true;
			}
		}

		elapsedTime = getTickCount() - elapsedTime;
		putText(frame, format("FPS %.1f", getTickFrequency() / elapsedTime), Point(10, 20),
			FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 255, 0), 1, LINE_AA);
		imshow("AR 2D Overlay", frame);

		if (waitKey(1) == 27) break;
	}

	return 0;
}