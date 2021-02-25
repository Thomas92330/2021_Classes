#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

VideoCapture capture;
Mat quaryFrame, outputFrame;
Mat corres_img;
int slider_pos = 0, slider_pos_old = 0;
int num_frames = 0;
string windowName = "Markerless Tracking";

#define MAX_OBJECTS 1024
vector<Mat> trainImages;
vector<Mat> desc_train(MAX_OBJECTS); Mat desc_query;
vector<vector<KeyPoint>> kp_train(MAX_OBJECTS);  vector<KeyPoint> kp_query(MAX_OBJECTS);

Ptr<ORB> orb;
Ptr<DescriptorMatcher> descriptorMatcher;
vector<KeyPoint> kp_query_inliers, kp_train_inliers;
vector<DMatch> inlierMatches;

void loadImages()
{
	vector<string> files;
	glob("data/images", files);
	for (int i = 0; i < files.size(); i++)
	{
		Mat img = imread(files[i]); 
		if (img.empty())
		{
			cerr << files[i] << " is invalid !" << endl;
			continue;
		}
		trainImages.push_back(img);
		cout << "<" << files[i] << "> loaded.\n";
	}
}

int orbTracker(int indexTrainImage)
{
	vector<vector<DMatch>> matches;
	descriptorMatcher->knnMatch(desc_query, desc_train.at(indexTrainImage), matches, 2);

	vector<KeyPoint> kp_query_good_match, kp_train_good_match;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.7 * matches[i][1].distance)
		{
			kp_query_good_match.push_back(kp_query[matches[i][0].queryIdx]);
			kp_train_good_match.push_back(kp_train.at(indexTrainImage)[matches[i][0].trainIdx]);
		}
	}

	int num_good_matches = kp_query_good_match.size();
	if (num_good_matches < 4) return 0;

	inlierMatches.clear();
	Mat inlierMask=Mat(1, kp_query_good_match.size(), CV_8U);

	vector<Point2f> pts_query_good_match, pts_train_good_match;
	for (int i = 0; i < num_good_matches; i++)
	{
		pts_query_good_match.push_back(kp_query_good_match[i].pt);
		pts_train_good_match.push_back(kp_train_good_match[i].pt);
	}

	Mat orbHomography = findHomography(pts_query_good_match, pts_train_good_match, RANSAC, 2, inlierMask);

	int num_inlier_matches = 0;

	if (!orbHomography.empty())
	{
		kp_query_inliers.clear(); 
		kp_train_inliers.clear();

		for (int i = 0, j = 0; i < num_good_matches; i++)
		{
			if (inlierMask.at<uchar>(i))
			{
				kp_query_inliers.push_back(kp_query_good_match[i]);
				kp_train_inliers.push_back(kp_train_good_match[i]);
				inlierMatches.push_back(DMatch(j, j, 0));
				j++;
				num_inlier_matches++;
			}
		}
	}

	return num_inlier_matches;
}

void on_trackbar(int, void*)
{
	if (abs(slider_pos - slider_pos_old) > 1)
		capture.set(CAP_PROP_POS_FRAMES, slider_pos);

	if (slider_pos == num_frames || quaryFrame.empty())
	{
		destroyWindow("Correspondences");

		createTrackbar("frames", windowName, &slider_pos, 0, on_trackbar);
		setTrackbarPos("frames", windowName, 0);
		capture.set(CAP_PROP_POS_FRAMES, 0);
		capture >> quaryFrame;
		resize(quaryFrame, quaryFrame, Size(640, 480));

		string text = "hit any key (except ESC) to restart...";
		Size textSize = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.5, 1.2, 0);
		Point textOrigin((quaryFrame.cols - textSize.width) / 2, (outputFrame.rows - textSize.height) / 2);
		rectangle(quaryFrame, textOrigin + Point(0, 3), textOrigin + Point(textSize.width, -textSize.height), Scalar::all(255), FILLED);
		putText(quaryFrame, text, textOrigin, FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1.2, LINE_AA);

		imshow(windowName, quaryFrame);
		if (waitKey(0) == 27) exit(0);

		createTrackbar("frames", windowName, &slider_pos, num_frames, on_trackbar);

	}
}

int main()
{
	capture.open("data/video/videoTuto3.MOV");

	orb = ORB::create();
	descriptorMatcher = DescriptorMatcher::create( "BruteForce-Hamming" );

	loadImages();
	if (!trainImages.size())
	{
		cerr << "Couldn't load images ..." << endl;
		return 1;
	}

	cout << endl << "< Total objects collected : " << trainImages.size() << endl;
	
	for (int i = 0; i < trainImages.size(); i++)
		orb->detectAndCompute(trainImages.at(i), noArray(), kp_train[i], desc_train[i]);	

	cout << "< Detection and description done" << endl;

	if (!capture.isOpened()) 
	{
		cerr << "Couldn't open the video file ..." << endl;
		return 1;
	}

	namedWindow(windowName, WINDOW_AUTOSIZE);

	num_frames = capture.get(CAP_PROP_FRAME_COUNT);

	if (num_frames)
		createTrackbar("frames", windowName, &slider_pos, num_frames, on_trackbar);

	while (true)
	{
		double elapsedTime = getTickCount();
		
		capture >> quaryFrame;
		if (quaryFrame.empty()) break;

		setTrackbarPos("frames", windowName, slider_pos);
		slider_pos_old = slider_pos;
		slider_pos++;

		resize(quaryFrame, quaryFrame, Size(640, 480));

		outputFrame = quaryFrame.clone();

		orb->detectAndCompute(quaryFrame, noArray(), kp_query, desc_query);

		for (int indexTrainImage = 0; indexTrainImage < trainImages.size(); indexTrainImage++)
		{
			int numberOfInlierMatches = orbTracker(indexTrainImage);
			
			if (numberOfInlierMatches > 10)
			{
				Mat closestImage;

				if (trainImages[indexTrainImage].rows > trainImages[indexTrainImage].cols)
				{
					resize(trainImages[indexTrainImage], closestImage, Size(100, 150));
					rectangle(closestImage, Point(0, 0), Point(closestImage.cols, closestImage.rows),
						Scalar(0, 255, 0), 3, 8, 0);
					closestImage.copyTo(outputFrame(Rect(0, 0, 100, 150)));
				}
				else
				{
					resize(trainImages[indexTrainImage], closestImage, Size(150, 100));
					rectangle(closestImage, Point(0, 0), Point(closestImage.cols, closestImage.rows),
						Scalar(0, 255, 0), 3, 8, 0);
					closestImage.copyTo(outputFrame(Rect(0, 0, 150, 100)));
				}

				drawMatches(trainImages[indexTrainImage], kp_train_inliers, quaryFrame, kp_query_inliers, inlierMatches, corres_img);

				resize(corres_img, corres_img, Size(), .7, .7);
				elapsedTime = getTickCount() - elapsedTime;

				putText(corres_img, format("Inliers : %d", numberOfInlierMatches), Point(5, corres_img.rows - 35),
					FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
				putText(corres_img, format("FPS : %.1f", getTickFrequency() / elapsedTime), Point(5, corres_img.rows - 5),
					FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
				imshow("Correspondences", corres_img);
			}
		}

		imshow(windowName, outputFrame);
		if (waitKey(1) == 27) break;
	}

	return 0;
}