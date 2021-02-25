#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Declare global variables
int slider_pos = 0, slider_pos_old = 0;
VideoCapture capture;

int menu()
{
	int choice;
	system("cls"); 

	cout << endl << "--- Computer Vision Basic Processings ---" << endl << endl;
	cout << "-----------------------------------------" << endl;
	cout << " 1 : Read and display an image" << endl;
	cout << " 2 : Linear filtering" << endl;
	cout << " 3 : Smoothing" << endl;
	cout << " 4 : Morphology" << endl;
	cout << " 5 : Thresholding" << endl;
	cout << " 6 : Edge detection" << endl;
	cout << " 7 : Histogram equalization" << endl;
	cout << " 8 : Template matching" << endl;
	cout << " 9 : Find contours" << endl;
	cout << " 10: Convex hull" << endl;
	cout << " 11: Matching descriptors" << endl;
	cout << " 12: SVM classification" << endl;
	cout << " 13: GUI video" << endl;
	cout << " 14: Quit the program..." << endl;
	cout << "-----------------------------------------" << endl << endl;
	cout << "Please enter the number of the function to run : ";
	
	cin >> choice;
	if ( choice > 0 && choice < 14 )
		cout << "Press any key to quit the function" << endl;

	return choice;
}

int readDisplayImage()
{
	// Declare the window name variable
	string windowName = "Read/Display Image";

	// Load the image
	Mat frame = imread( "images/landscape.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( frame.empty() )
	{
		cerr << endl << "Couldn't read the image..." << endl;
		return 1;
	}

	// Create a window
	namedWindow( windowName, WINDOW_AUTOSIZE );

	// Display the image
	imshow( windowName, frame );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the window
	destroyWindow(windowName);

	return 0;
}

int linearFiltering()
{
	// Declare variables
	Mat src, dst;
	
	string windowName_src = "Source Image";
	string windowName_dst = "Linear Filtering";

	// Initialize filter arguments
	Mat kernel = Mat::ones(3, 3, CV_32F )/(9.);

	// Load the image
	src = imread( "images/plane.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl << "Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Apply the filter
	filter2D( src, dst, -1 , kernel );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for a pressed key
	waitKey(0) ;
	
	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst);

	return 0;
}

int smoothing()
{
	// Declare variables
	int MAX_KERNEL_LENGTH = 31;
	Mat src; Mat dst1, dst2, dst3;
	string windowName_src = "Source Image";
	string windowName_dst1 = "Gaussian blur";
	string windowName_dst2 = "Median blur";
	string windowName_dst3 = "Bilateral Filter";

    // Load the source image
    src = imread( "images/monalisa.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst3, WINDOW_AUTOSIZE );

	// Apply the Gaussian blur
    GaussianBlur( src, dst1, Size( 5, 5 ), 0, 0 );
	
    // Apply the Median blur
    medianBlur( src, dst2, 5 );

	// Apply the Bilateral Filter
    bilateralFilter( src, dst3, 15, 15*2, 15/2 );

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dst1 );
	imshow( windowName_dst2, dst2 );
	imshow( windowName_dst3, dst3 );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);
	destroyWindow(windowName_dst3);

	return 0;
}

int morphology()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_dst1 = "Dilatation";
	string windowName_dst2 = "Erosion";
	Mat src, erosion_dst, dilation_dst;

	// Load the image
	src = imread( "images/apple.png", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );

	// Apply the dilation
	int dilation_size = 3;
	int dilation_type = MORPH_RECT;

	Mat structuring_element = getStructuringElement(MORPH_RECT, Size( 7, 7 ) );

    dilate( src, dilation_dst, structuring_element);

	// Apply the erosion
    erode( src, erosion_dst, structuring_element);

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dilation_dst );
	imshow( windowName_dst2, erosion_dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);

	return 0;
}

int thresholding()
{
	// Declare variables
	Mat src, src_gray, dst;
	string windowName_src = "Source Image";
	string windowName_dst = "Threshold";

	// Load the image
	src = imread( "images/tiger.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Threshold the image
	threshold( src_gray, dst, 100, 255, THRESH_BINARY );

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst);	
	
	return 0;
}

int edgeDetection()
{
	// Declare variables
	Mat src, src_gray, dst1, dst2;
	string windowName_src = "Source Image";
	string windowName_dst1 = "Laplace";
	string windowName_dst2 = "Canny";

	// Load the image
	src = imread( "images/building.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Apply Laplace
	Laplacian( src_gray, dst1, CV_16S, 3 );

	convertScaleAbs( dst1, dst1 );
	
	// Apply Canny
	Canny( src_gray, dst2, 20, 110 );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dst1 );
	imshow( windowName_dst2, dst2 );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);

	return 0;
}

int histogramEqualization()
{
	// Declare variables
	Mat src, src_gray, dst;
	string windowName_src = "Source Image";
	string windowName_dst = "Histogram Equalization";

	// Load the image
	src = imread( "images/bird.jpg", IMREAD_UNCHANGED );
	
	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Equalize the histogram
	equalizeHist( src_gray, dst );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_dst );

	return 0;
}

int templateMatching()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_templ = "Template Image";
	string windowName_dst = "Template Matching";
	Mat src, templ, dst;

	// Load the reference image and the template
	src = imread( "images/bus.jpg", IMREAD_UNCHANGED );
	templ = imread( "images/bus_template.png", IMREAD_UNCHANGED );

	// Check if images are empty
	if( src.empty() || templ.empty() )
	{
		cerr << endl << "Couldn't read one of the images..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_templ, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Match the template with the reference image
	matchTemplate( src, templ, dst, TM_SQDIFF_NORMED);

	// Localize the best match with minMaxLoc
	Point minLoc; 
	minMaxLoc( dst, 0, 0, &minLoc, 0 );

	dst = src.clone();
	
	// Draw the rectangle
	rectangle( dst, minLoc, Point(minLoc.x + templ.cols , minLoc.y + templ.rows ),
							CV_RGB(0,255,0), 5 );
		
	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_templ, templ );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_templ );
	destroyWindow( windowName_dst );

	return 0;
}

int findContours()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_dst = "Contours";
	Mat src, src_gray, dst;

	// Load the image
	src = imread( "images/porsche.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Find contours
	RNG rng(1);
	vector<vector<Point>> contours;

	// Detect edges using canny
	Canny( src_gray, src_gray, 100, 200 );

	// Find contours
	findContours( src_gray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );

	// Draw contours
	dst = Mat::zeros( src_gray.size(), CV_8UC3 );

	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( dst, contours, (int)i, color, 2 );
	}

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_dst );

return 0;
}

void convexHull()
{
	// Declare variables
    Mat img = Mat::zeros(Size(500, 500), CV_8UC3);	
	RNG &rng = theRNG();
	int n_points = rng.uniform(50, 100);
	cout << "n_points : " << n_points << endl;
	vector<Point> points;

	// Generate random points
    for( int i = 0; i < n_points; i++ )
    {
        Point pt;
		pt.x = rng.uniform(img.cols/4, img.cols*3/4);
		pt.y = rng.uniform(img.rows/4, img.rows*3/4);

        points.push_back(pt);
    }

	// Compute the convex hull
    vector<int> hull;
    convexHull( points, hull );

	// Draw points
    for( int i = 0; i < n_points; i++ )
        circle( img, points[i], 3, Scalar(0, 0, 255), FILLED, LINE_AA );

	// Draw the convex hull
    int hullcount = hull.size();
    Point pt0 = points[hull[hullcount-1]];

    for( int i = 0; i < hullcount; i++ )
    {
        Point pt = points[hull[i]];
        line( img, pt0, pt, Scalar(0, 255, 0) );
        pt0 = pt;
    }

	// Display the image
    imshow( "Convex hull", img );

	// Press escape to quit the loop
	waitKey(0);

	// Destroy the window
	destroyWindow( "Convex hull" );
}

int matchingDescriptors()
{
	// Load reference and query images
    Mat img1 = imread( "images/box.png", IMREAD_GRAYSCALE );	
    Mat img2 = imread( "images/box_in_scene.png", IMREAD_GRAYSCALE );

	// Check if images are empty
    if( img1.empty() || img2.empty() )
    {
		cerr << endl << "Couldn't read one of the images..." << endl;
		return 1;
    }

    // Detect keypoints
	Ptr<FeatureDetector> orb = ORB::create(100);
    vector<KeyPoint> keypoints1, keypoints2;
    orb->detect( img1, keypoints1 );
    orb->detect( img2, keypoints2 );

    // Compute descriptors    
    Mat descriptors1, descriptors2;
	orb->compute( img1, keypoints1, descriptors1 );
	orb->compute( img2, keypoints2, descriptors2 );

    // Match descriptors
    BFMatcher matcher( NORM_L2 );
    vector<DMatch> matches;
    matcher.match( descriptors1, descriptors2, matches );

    // Draw results
    string windowName = "Matches";
    namedWindow( windowName, WINDOW_AUTOSIZE );
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches );
   	imshow( windowName, img_matches );

	// Wait for pressed key
	waitKey(0);

	// Destroy the window
	destroyWindow( windowName );

	return 0;
}

void classificationSVM()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	int labels[4] = { 1, 2, 3, 4 };
	Mat labelsMat(4, 1, CV_32SC1, labels);

	// Generate random training data 
	Point data1 = Point(rand() % 512, rand() % 512);
	Point data2 = Point(rand() % 512, rand() % 512);
	Point data3 = Point(rand() % 512, rand() % 512);
	Point data4 = Point(rand() % 512, rand() % 512);
	float trainingData[4][2] = { {data1.x,data1.y }, {data2.x, data2.y }, {data3.x, data3.y }, {data4.x, data4.y } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	vector<Point2i> train_data;
	train_data.push_back(data1); train_data.push_back(data2); train_data.push_back(data3); train_data.push_back(data4);

	// Set up the SVM parameters
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	// Train the SVM
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), yellow(0, 255, 255);

	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = svm->predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(j, i) = green;
			else if (response == 2)
				image.at<Vec3b>(j, i) = blue;
			else if (response == 3)
				image.at<Vec3b>(j, i) = red;
			else if (response == 4)
				image.at<Vec3b>(j, i) = yellow;
		}

	// Show the training data
	circle(image, data1, 5, Scalar(0, 0, 0), -1);
	circle(image, data2, 5, Scalar(0, 0, 0), -1);
	circle(image, data3, 5, Scalar(0, 0, 0), -1);
	circle(image, data4, 5, Scalar(0, 0, 0), -1);

	// Draw results
	string windowName = "SVM classification";
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, image); // show it to the user

	// Wait for pressed key
	waitKey(0);

	// Destroy the window
	destroyWindow(windowName);
}

void onTrackbarSlide(int, void*)
{
    // Update the slider position only if an event has occured
	if( abs( slider_pos - slider_pos_old ) > 1 )
		capture.set( CAP_PROP_POS_FRAMES, slider_pos );
}

int guiVideo()
{
	// Declare variables
	string windowName = "Video Capture";
	const char *trackbar_name = "Slider";
	Mat frame;

	// Open the video
    capture.open( "videos/video.mp4" );

	// Check if the video is opened
    if( !capture.isOpened() )
    {
		cerr << endl << "Error: Could not initialize capturing..." << endl;
        return 1;
    }

	// Create the window
    namedWindow( windowName, WINDOW_AUTOSIZE );

	// Get the number of frames of the video
	int count_frames = (int) capture.get( CAP_PROP_FRAME_COUNT );

	// Create the trackbar
	if( count_frames )
		createTrackbar( trackbar_name, windowName, &slider_pos, count_frames, onTrackbarSlide );

	// Create the infinite loop
	while(true)
	    {
			// Grab the frame from the video stream
			capture >> frame;

			// Check if the frame is empty
			if( frame.empty() )	break;

			// Display the frame
			imshow( windowName, frame );

			// Press any key to quit
			if( waitKey(40) > 0 ) break;

			// Update the position of trackbar
			setTrackbarPos( trackbar_name, windowName, slider_pos );
			slider_pos_old = slider_pos;
			slider_pos++;

	    }
		// Destroy the window
		destroyWindow( windowName );

	return 0;
}
int main ()
{
	while ( true )
	{ 
		int fct_selected = menu();

		switch ( fct_selected )
		{
			case 1 : readDisplayImage();
					 break;
			case 2 : linearFiltering();
					 break;
			case 3 : smoothing();
					 break;
			case 4 : morphology();
					 break;
			case 5 : thresholding();
					 break;
			case 6 : edgeDetection();
					 break;
			case 7 : histogramEqualization();
					 break;
			case 8 : templateMatching();
					 break;
			case 9 : findContours();
					 break;
			case 10: convexHull();
					 break;
			case 11: matchingDescriptors();
					 break;
			case 12: classificationSVM();
					 break;
			case 13: guiVideo();
					 break;
			case 14: cout << endl << endl << "The program is exited... ";
					 return 0;				
			default: cout << endl << "The selected function is not valid! Please retry..." << endl;		
					 system("pause");
					 break;
		}
	} 

	return 0;
}