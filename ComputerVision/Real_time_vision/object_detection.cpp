#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;

Mat frame;
string windowName = "Deep Learning object detection";
VideoCapture capture;
int num_frames = 0;
int slider_pos = 0, slider_pos_old = 0;
vector<string> classes;

void postProcess(vector<Mat> out, Net net);

void drawPredictions(int classId, float conf, Rect box);

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

int main()
{
    string file = "yolo/classes_names_yolo.txt";
    ifstream ifs(file.c_str());
    if (!ifs.is_open())
        cerr << "File " << file << " not found" << endl;

    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    string cfg_file = "yolo/yolov4-tiny.cfg";
    string weight_file = "yolo/yolov4-tiny.weights";
    
    Net net = readNet(weight_file, cfg_file); 
   
    if (net.empty()) cerr << "Error loading NET"<< endl;
 
    namedWindow(windowName, WINDOW_AUTOSIZE);

    capture.open("vidTuto42.wmv");

    if (!capture.isOpened())
    {
        cerr << "Couldn't open the video..." << endl;
        return 1;
    }
  
    num_frames = (int)capture.get(CAP_PROP_FRAME_COUNT);
    if (num_frames)
        createTrackbar("frames", windowName, &slider_pos, num_frames, on_trackbar);
        
    Mat blob;
    
    bool START = true;

    while(true)
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
        
        blobFromImage(frame, blob, 1., Size(416, 416), Scalar(), true);

        net.setInput(blob, "", 0.00392, Scalar(0, 0, 0));
                
        vector<String> outNames = net.getUnconnectedOutLayersNames();  

        vector<Mat> outs;
        net.forward(outs, outNames);

        postProcess(outs, net);

        time = getTickCount() - time;
        time  = 1000. / (getTickFrequency() / time);
        string label = format("%.2f ms", time);
        putText(frame, label, Point(0, 25), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0), 2, LINE_AA);
            
        imshow(windowName, frame);
        if (key == 27) break; 
    }

    return 0;
}

void postProcess(vector<Mat> outs, Net net)
{
    float confidenceThreshold = 0.35; 

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;   

    for (int i = 0; i < outs.size(); i++)
    {
        Mat outBlob = Mat(outs[i].size(), outs[i].depth(), outs[i].data);

        for (int j = 0; j < outBlob.rows; j++)
        {         
            Mat scores = outBlob.row(j).colRange(5, outBlob.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if(confidence > confidenceThreshold)
            {
                int centerX = (int)(outBlob.row(j).at<float>(0) * frame.cols);
                int centerY = (int)(outBlob.row(j).at<float>(1) * frame.rows);
                int width = (int)(outBlob.row(j).at<float>(2) * frame.cols);
                int height = (int)(outBlob.row(j).at<float>(3) * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x); 
                confidences.push_back(confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }  

    vector<int> indices;   
    float nmsThreshold=0.5;
  
    NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPredictions(classIds[idx], confidences[idx], box);
    }
}

void drawPredictions(int classId, float confidence, Rect box)
{
    rectangle(frame, box, CV_RGB(0, 255, 0));

    string label = format("%s: %.2f", classes[classId].c_str(), confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    rectangle(frame, Point(box.x, box.y - labelSize.height), Point(box.x + labelSize.width, box.y + baseLine), CV_RGB(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,0), 1, LINE_AA);
}