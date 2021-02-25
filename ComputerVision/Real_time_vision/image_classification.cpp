#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace dnn;

VideoCapture capture;
Mat frame;
string windowName = "Deep Learning image classification";
int num_frames = 0;
int slider_pos = 0, slider_pos_old = 0;

void on_trackbar(int, void*)
{
    if (abs(slider_pos - slider_pos_old) > 1)
        capture.set(CAP_PROP_POS_FRAMES, slider_pos);

    if (slider_pos == num_frames || frame.empty())
    {
        createTrackbar("frames", windowName, &slider_pos, 0, on_trackbar);
        setTrackbarPos("frames", windowName, 0);
        capture.set(CAP_PROP_POS_FRAMES, 0);
        capture >> frame;

        string text = "hit any key (except ESC) to restart...";
        Size textSize = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.5, 1.2, 0);
        Point textOrigin((frame.cols - textSize.width) / 2, (frame.rows - textSize.height) / 2);
        rectangle(frame, textOrigin + Point(0, 3), textOrigin + Point(textSize.width, -textSize.height), Scalar::all(255), FILLED);
        putText(frame, text, textOrigin, FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1.2, LINE_AA);

        imshow(windowName, frame);
        if (waitKey(0) == 27) exit(0);

        createTrackbar("frames", windowName, &slider_pos, num_frames, on_trackbar);
    }
}

int main(int argc, char** argv)
{
    string file = "googlenet/classes_names_googlenet.txt";
    ifstream ifs(file.c_str());
    if (!ifs.is_open())
        cerr<< "File " << file << " not found" << endl;
 
    vector<string> classes;
    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    string cfg_file = "googlenet/bvlc_googlenet.prototxt";
    string model_file = "googlenet/bvlc_googlenet.caffemodel";

    Net net = readNet(model_file, cfg_file);

    namedWindow(windowName, WINDOW_AUTOSIZE);

    capture.open("vidTuto41.wmv");

    Mat blob;
 
    num_frames = (int)capture.get(CAP_PROP_FRAME_COUNT);
    if (num_frames)
        createTrackbar("frames", windowName, &slider_pos, num_frames, on_trackbar);

    bool START = true;

    while (true)
    {
        double time = getTickCount();

        char key = waitKey(1);
        if (key == 32) START = !START;
        if (key == 27) break;        
        if (START == false) continue;
           
        capture >> frame;
        setTrackbarPos("frames", windowName, slider_pos);
        slider_pos_old = slider_pos;
        slider_pos++;

        blobFromImage(frame, blob, 1., Size(224, 224), Scalar(104, 117, 123), true);  

        net.setInput(blob);

        Mat prob = net.forward();

        Point classIdPoint;
        double confidence;
        minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
        
        int classId = classIdPoint.x;
        
        time = getTickCount() - time;
        time = 1000. / (getTickFrequency() / time);
        string label = format("%.2f ms", time);
        putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(0, 255, 0), 2, LINE_AA);

        label = format("%s: %.2f", classes[classId].c_str(), confidence);
        rectangle(frame, Point(0, frame.rows - 30), Point(frame.cols, frame.rows), Scalar::all(0), FILLED);
        putText(frame, label, Point(0, frame.rows-7), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 2, LINE_AA); 
       
        imshow(windowName, frame);
        if (key == 27) break;       
    }

    return 0;
}