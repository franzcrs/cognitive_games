/*
#include <opencv2/opencv.hpp>

// Callback function for the trackbar
void onTrackbar(int value, void* userData) {
    cv::Mat* frame = static_cast<cv::Mat*>(userData);
    cv::Mat gray;
    cv::cvtColor(*frame, gray, cv::COLOR_BGR2GRAY); // Convert to grayscale
    cv::Mat binary;
    cv::threshold(gray, binary, value, 255, cv::THRESH_BINARY); // Apply threshold

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours to find the most likely rectangle
    double maxArea = 0;
    std::vector<cv::Point> maxContour;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            maxContour = contour;
        }
    }

    // If a contour with a significant area is found
    if (!maxContour.empty()) {
        // Approximate the contour to a polygon
        std::vector<cv::Point> approx;
        cv::approxPolyDP(maxContour, approx, 0.04 * cv::arcLength(maxContour, true), true);

        // If the polygon has four corners, it is likely a rectangle
        if (approx.size() == 4) {
            // Draw the rectangle and its sides
            cv::Mat coloredBinary;
            cv::cvtColor(binary, coloredBinary, cv::COLOR_GRAY2BGR); // Convert binary to color
            cv::drawContours(coloredBinary, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);
            for (int i = 0; i < 4; ++i) {
                cv::line(coloredBinary, approx[i], approx[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            // Resize the perfect rectangle before displaying it
            // Calculate the scaling factor
            double scale = 300.0 / coloredBinary.cols;
            cv::resize(coloredBinary, coloredBinary, cv::Size(), scale, scale);
            cv::imshow("Camera", coloredBinary);

            // Transform the rectangle into a perfect rectangle
            cv::Point2f srcPoints[4];
            float rect_width = 1100.0;
            float rect_height = 900.0;
            cv::Point2f dstPoints[4] = {{0, 0}, {0, rect_height}, {rect_width, rect_height}, {rect_width, 0}}; // Define the destination points for the perfect rectangle
            for (int i = 0; i < 4; ++i) {
                srcPoints[i] = approx[i];
            }
            cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
            cv::Mat perfectRect;
            cv::warpPerspective(binary, perfectRect, transformMatrix, cv::Size(rect_width, rect_height));


            // Crop the perfect rectangle to cut the border
            int borderSize = 10;
            cv::Rect roi(borderSize, borderSize, perfectRect.cols - 2 * borderSize, perfectRect.rows - 2 * borderSize);
            cv::Mat croppedRect = perfectRect(roi);

            // Dilate the cropped rectangle to expand black areas
            cv::Mat dil_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)); // Define dilation kernel
            cv::Mat dilatedRect;
            cv::dilate(croppedRect, dilatedRect, dil_element); // 'element' is the erosion kernel from the previous code snippet


            // Erode the black areas
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)); // Define erosion kernel
            cv::Mat erodedRect;
            cv::erode(dilatedRect, erodedRect, element);
            cv::erode(erodedRect, erodedRect, element);

            // Convert erodedRect to BGR for displaying circles in color
            cv::Mat erodedRectBGR;
            cv::cvtColor(erodedRect, erodedRectBGR, cv::COLOR_GRAY2BGR);

            // Find circles in the erodedRect
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(erodedRect, circles, cv::HOUGH_GRADIENT, 1,
                             100.0,  // change this value to detect circles with different distances to each other
                             100, 30, 80, 700 // change the last two parameters
                                            // (min_radius & max_radius) to detect larger circles
                             );

            // Draw the circles
            for (size_t i = 0; i < circles.size(); i++) {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                // Draw the circle center
                cv::circle(erodedRectBGR, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                // Draw the circle outline
                cv::circle(erodedRectBGR, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
            }

            cv::imshow("Perfect Rectangle", erodedRectBGR);

            return;
        }
    }

    // If no rectangle is found, show the binary image without modifications
    // Resize the perfect rectangle before displaying it
    // Calculate the scaling factor
    double scale = 300.0 / binary.cols;
    cv::resize(binary, binary, cv::Size(), scale, scale);
    cv::imshow("Camera", binary);
}

int main() {
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    cv::Mat frame; // Placeholder for the current frame
    cap.read(frame); // Read a frame from the camera

    cv::namedWindow("Camera");

    int initialThreshold = 184; // Initial threshold value
    int maxThreshold = 255; // Maximum threshold value
    cv::createTrackbar("Threshold", "Camera", &initialThreshold, maxThreshold, onTrackbar, &frame);

    onTrackbar(initialThreshold, &frame); // Initial binary conversion

    while (true) {
        cap.read(frame); // Read a frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Unable to read frame from the camera" << std::endl;
            break;
        }

        int key = cv::waitKey(30);
        if (key == 27) // Break the loop if 'Esc' is pressed
            break;

        onTrackbar(initialThreshold, &frame);
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows
    return 0;
}
*/

/*
#include <opencv2/opencv.hpp>
#include <deque>

// Structure to store the detected circle
struct DetectedCircle {
    cv::Point center;
    int radius;
};

// Callback function for the trackbar
void onTrackbar(int value, void* userData) {
    cv::Mat* frame = static_cast<cv::Mat*>(userData);
    cv::Mat gray;
    cv::cvtColor(*frame, gray, cv::COLOR_BGR2GRAY); // Convert to grayscale
    cv::Mat binary;
    cv::threshold(gray, binary, value, 255, cv::THRESH_BINARY); // Apply threshold

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours to find the most likely rectangle
    double maxArea = 0;
    std::vector<cv::Point> maxContour;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            maxContour = contour;
        }
    }

    // If a contour with a significant area is found
    if (!maxContour.empty()) {
        // Approximate the contour to a polygon
        std::vector<cv::Point> approx;
        cv::approxPolyDP(maxContour, approx, 0.04 * cv::arcLength(maxContour, true), true);

        // If the polygon has four corners, it is likely a rectangle
        if (approx.size() == 4) {
            // Draw the rectangle and its sides
            cv::Mat coloredBinary;
            cv::cvtColor(binary, coloredBinary, cv::COLOR_GRAY2BGR); // Convert binary to color
            cv::drawContours(coloredBinary, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);
            for (int i = 0; i < 4; ++i) {
                cv::line(coloredBinary, approx[i], approx[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            // Resize the perfect rectangle before displaying it
            // Calculate the scaling factor
            double scale = 300.0 / coloredBinary.cols;
            cv::resize(coloredBinary, coloredBinary, cv::Size(), scale, scale);
            cv::imshow("Camera", coloredBinary);

            // Transform the rectangle into a perfect rectangle
            cv::Point2f srcPoints[4];
            float rect_width = 1100.0;
            float rect_height = 900.0;
            cv::Point2f dstPoints[4] = {{0, 0}, {0, rect_height}, {rect_width, rect_height}, {rect_width, 0}}; // Define the destination points for the perfect rectangle
            for (int i = 0; i < 4; ++i) {
                srcPoints[i] = approx[i];
            }
            cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
            cv::Mat perfectRect;
            cv::warpPerspective(binary, perfectRect, transformMatrix, cv::Size(rect_width, rect_height));


            // Crop the perfect rectangle to cut the border
            int borderSize = 10;
            cv::Rect roi(borderSize, borderSize, perfectRect.cols - 2 * borderSize, perfectRect.rows - 2 * borderSize);
            cv::Mat croppedRect = perfectRect(roi);

            // Dilate the cropped rectangle to expand black areas
            cv::Mat dil_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)); // Define dilation kernel
            cv::Mat dilatedRect;
            cv::dilate(croppedRect, dilatedRect, dil_element); // 'element' is the erosion kernel from the previous code snippet


            // Erode the black areas
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)); // Define erosion kernel
            cv::Mat erodedRect;
            cv::erode(dilatedRect, erodedRect, element);
            cv::erode(erodedRect, erodedRect, element);

            // Convert erodedRect to BGR for displaying circles in color
            cv::Mat erodedRectBGR;
            cv::cvtColor(erodedRect, erodedRectBGR, cv::COLOR_GRAY2BGR);

            // Find circles in the erodedRect
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(erodedRect, circles, cv::HOUGH_GRADIENT, 1,
                             100.0,  // change this value to detect circles with different distances to each other
                             100, 30, 80, 700 // change the last two parameters
                                            // (min_radius & max_radius) to detect larger circles
                             );

            // Store the detected circles for this frame
            std::deque<DetectedCircle> prevCircles;
            for (const auto& circle : circles) {
                prevCircles.push_back({cv::Point(cvRound(circle[0]), cvRound(circle[1])), cvRound(circle[2])});
            }

            // Draw the average circle from the previous frames
            cv::Point avgCenter(0, 0);
            int avgRadius = 0;
            int count = 0;
            for (const auto& circle : prevCircles) {
                if (circle.radius > 0) {
                    avgCenter += circle.center;
                    avgRadius += circle.radius;
                    count++;
                }
            }
            if (count > 0) {
                avgCenter.x /= count;
                avgCenter.y /= count;
                avgRadius /= count;
                cv::circle(erodedRectBGR, avgCenter, avgRadius, cv::Scalar(255, 0, 0), 2);
            }

            // Draw the circles detected in the current frame
            for (const auto& circle : circles) {
                cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
                int radius = cvRound(circle[2]);
                // Draw the circle center
                // cv::circle(erodedRectBGR, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                // // Draw the circle outline
                // cv::circle(erodedRectBGR, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
            }

            cv::imshow("Perfect Rectangle", erodedRectBGR);
        }
    }

    // If no rectangle is found, show the binary image without modifications
    // Resize the perfect rectangle before displaying it
    // Calculate the scaling factor
    double scale = 300.0 / binary.cols;
    cv::resize(binary, binary, cv::Size(), scale, scale);
    cv::imshow("Camera", binary);
}

int main() {
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    cv::Mat frame; // Placeholder for the current frame
    cap.read(frame); // Read a frame from the camera

    cv::namedWindow("Camera");

    int initialThreshold = 200; // Initial threshold value
    int maxThreshold = 255; // Maximum threshold value
    cv::createTrackbar("Threshold", "Camera", &initialThreshold, maxThreshold, onTrackbar, &frame);

    onTrackbar(initialThreshold, &frame); // Initial binary conversion

    while (true) {
        cap.read(frame); // Read a frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Unable to read frame from the camera" << std::endl;
            break;
        }

        int key = cv::waitKey(30);
        if (key == 27) // Break the loop if 'Esc' is pressed
            break;

        onTrackbar(initialThreshold, &frame);
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows
    return 0;
}

*/

/*
#include <cstdio>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  return 0;
}
*/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <filesystem>
#include <cmath>
#include <sstream>

#include <signal.h>
#include <sys/time.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define DID     22           //  Device ID (fixed)

#define DEV     100         // The Number of Devices
#define SCE     100         // The number of Scenarios
#define GAM     100         // The number of Games


int DeviceValid[DEV+1],     //  available devices
    GameValid[GAM+1],     //  available games
    ScenarioValid[SCE+1];         //  available scenarios

// UDP  to get the Server IP Address - - - - - -

#define BUFSIZE         8192
#define ISS_UDP_PORT    9000    //  Port ID for UDP/IP ISS Database communication
#define ISS_TCP_PORT    9866    //  ISS Database Server
#define ISS_UDPR_PORT   9868    //  Port ID for UDP/IP ISS Database Receiver communication

#define TIMEOUT_MS      5000    //  milliSeconds between retransmits

char    ISS_ServerIP[30]="192.168.13.100",
        buf[BUFSIZE];

int     fin2=0,             //  1: game end (ready to start)
        GameScore[GAM+1],
        DeviceID=DID,       //  Device ID (000-999)
        ISSstate=-1,      //  0: not connected, 1: connected
        DeviceStage=0,      //  state to send to the ISS server
        DeviceCommand=0,    // DeviceCommand to send to the ISS server
        DeviceData=0,       //  state or value to send to the ISS server
        DataSent=1,         //  0: ready to send, 1: sent
        ScenarioStage=0,    //  Scenario Stage
        ScenarioValue,      //  Scenario Value (data)
        ScenarioStartTime;  //   Secnario Starting Time in Sec (00-59)

int ISS_UDP_receive()    //  receive server (host) IP through UDP/IP ISS Database communication
{
    
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    
    int i=0,sock;
    struct sockaddr_in addr;
    
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ISS_UDP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_len = sizeof(addr);
    
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    memset(ISS_ServerIP, 0, sizeof(ISS_ServerIP));
    recv(sock, ISS_ServerIP, sizeof(ISS_ServerIP), 0);
    
    printf("ISS Database Server IP %s\n", ISS_ServerIP);
    close(sock);
    return(i);
}

void init()     //  init for each scenario
{
    int i;
    
    fin2=1;     //  game end (ready to start)
    DataSent=1;
    DeviceID=DID;
    DeviceStage=ScenarioStage;
    DeviceCommand=0;
    DeviceData=0;
    
    for (i=0; i<DEV; i++)
        DeviceValid[i]=-1;     //  available devices
    for (i=0; i<GAM; i++)
        GameValid[i]=-1;     //  available games
    for (i=0; i<SCE; i++)
        ScenarioValid[i]=-1;         //  available scenarios
    // printf("\n\n Data init is done. \n\n");
}

void senddata(){    // send state / DeviceCommand and data to the server
    
    char    sentdata[BUFSIZE],
            codeno[BUFSIZE];      //   text to int
    
    struct sockaddr_in server;
    unsigned long  dst_ip = inet_addr(ISS_ServerIP);
    int port   = ISS_TCP_PORT;  //  database server
    
    int     h,  //  ID
            i,j,k,s;
    char    numberName[20]="01234567890";
    
    memset(sentdata, 0, sizeof(sentdata));
    
    if (DeviceID>=100){
        k=(int)(DeviceID/100)%10;
        sentdata[0]=numberName[k];   //  Device ID (000-999)
    }
    else
        sentdata[0]=numberName[0];
    if (DeviceID>=10){
        k=(int)(DeviceID/10)%10;
        sentdata[1]=numberName[k];
    }
    else
        sentdata[1]=numberName[0];
    k=DeviceID%10;
    sentdata[2]=numberName[k];
    
    if (DeviceStage>=1000) {
        k=(int)(DeviceStage/1000)%10;
        sentdata[3]=numberName[k]; // State / stage ID (0000 - 9999)
    }
    else
        sentdata[3]=numberName[0];
    if (DeviceStage>=100) {
        k=(int)(DeviceStage/100)%10;
        sentdata[4]=numberName[k];
    }
    else
        sentdata[4]=numberName[0];
    if (DeviceStage>=10) {
        k=(int)(DeviceStage/10)%10;
        sentdata[5]=numberName[k];
    }
    else
        sentdata[5]=numberName[0];
    k=DeviceStage%10;
    sentdata[6]=numberName[k];
    
    if (DeviceCommand>=10) {    //  send DeviceCommand data (00 - 99)
        k=(int)(DeviceCommand/10)%10;
        sentdata[7]=numberName[k]; // value ID
    }
    else
        sentdata[7]=numberName[0]; // value ID
    k=DeviceCommand%10;
    sentdata[8]=numberName[k]; // value ID
    
    if (DeviceData>=10) {           //  send value  (00 - 99)
        k=(int)(DeviceData/10)%10;
        sentdata[9]=numberName[k]; // value ID
    }
    else
        sentdata[9]=numberName[0]; // value ID
    k=DeviceData%10;
    sentdata[10]=numberName[k]; // value ID

    
    if ((s = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Fail to ISS Server\n");
    }
    else{
        memset((char *) &server, 0, sizeof(server));
        server.sin_family      = AF_INET;
        server.sin_addr.s_addr = dst_ip;
        server.sin_port        = htons(port);
        if (connect(s, (struct sockaddr *) &server, sizeof server) < 0) {
            printf(" Device [%d] No ISS Server - State:%d\n",
                   DeviceID, DeviceStage);
            ISSstate=0;
        }
        else{
            write(s, sentdata, strlen(sentdata));   //  sending data
            memset(buf, 0, sizeof(buf));
            read(s, buf, sizeof(buf));  //  receiving data
            close(s);
            ISSstate=1;
            
            memset(codeno, 0, sizeof(codeno));
            codeno[0]=buf[0];   //  Device State
            codeno[1]=buf[1];   //  Device State
            codeno[2]=buf[2];   //  Device State
            codeno[3]=buf[3];   //  Device State
            ScenarioStage=atoi(codeno);     //  Scenario State:
            
            memset(codeno, 0, sizeof(codeno));
            codeno[0]=buf[4];   //  Device Value
            codeno[1]=buf[5];   //  Device Value
            codeno[2]=buf[6];   //  Device Value
            codeno[3]=buf[7];  //  Device Value
            
            if ((ScenarioStage == 2)&&(ScenarioStage != DeviceStage))
                init();
            else if (ScenarioStage == 3){
                ScenarioStartTime=atoi(codeno); // Starting Time in Sec (00-59)
            }
            else if (DeviceCommand == 15){
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[4];   //  Scenario ID
                codeno[1]=buf[5];
                h=atoi(codeno);     //  Scenario ID
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[6];   //  Scenario Valid
                codeno[1]=buf[7];
                ScenarioValid[h]=atoi(codeno);     //  Scenario Valid
                printf("- - The previous Game [%d] Score: %d\n",
                       h,ScenarioValid[h]);
            }
            else if (DeviceCommand == 16){
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[4];   //  Scenario ID
                codeno[1]=buf[5];
                h=atoi(codeno);     //  Scenario ID
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[6];   //  Scenario Valid
                codeno[1]=buf[7];
                ScenarioValid[h]=atoi(codeno);     //  Scenario Valid
                printf("- - Total Scenario [%d] Score: %d\n",
                       h,ScenarioValid[h]);
            }
            else if (DeviceCommand == 20){
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[4];   //  Scenario ID
                codeno[1]=buf[5];
                h=atoi(codeno);     //  Scenario ID
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[6];   //  Scenario Valid
                codeno[1]=buf[7];
                ScenarioValid[h]=atoi(codeno);     //  Scenario Valid
                printf("Scenario[%d] avaialble: %d\n",
                       h,ScenarioValid[h]);
            }
            else if (DeviceCommand == 21){
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[4];   //  Game ID
                codeno[1]=buf[5];
                h=atoi(codeno);     //  Game ID
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[6];   //  Game Valid
                codeno[1]=buf[7];
                GameValid[h]=atoi(codeno);     //  Scenario Valid
                printf("Game[%d] avaialble: %d\n",
                       h,GameValid[h]);
            }
            else if (DeviceCommand == 22){
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[4];   //  Device ID
                codeno[1]=buf[5];
                h=atoi(codeno);     //  Device ID
                memset(codeno, 0, sizeof(codeno));
                codeno[0]=buf[6];   //  Device Valid
                codeno[1]=buf[7];
                DeviceValid[h]=atoi(codeno);     //  Scenario Valid
                printf("Device[%d] avaialble: %d\n",
                       h,DeviceValid[h]);
            }
            else
                ScenarioValue=atoi(codeno);          //  Scenario Value (data)
        }
    }
    if (DataSent==0){
        printf("sent: %s\n", sentdata);
        DeviceCommand=0;    //  reset
        DeviceData=0;       //  reset
        DataSent=1;
    }
}


void* iss( void* args ) // Sensor Node
{
    int fin=0, l=0, m=0, n=0,
        i,j,k,t;

    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 400000000;     // 0.4 msec Coomunication Time Interval
        
    k=0;
    while(fin<2){
//        if (DataSent==0){
//            printf("[%d] ID:%d, State:%d Data:%d \n",
//                   k, DeviceID, DeviceStage, DeviceData);
//            k++;
//        }
        senddata();
        nanosleep(&ts, NULL);
    }
    return NULL;
}

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::vector<double> split(const std::string& str, char delimiter) {
    std::vector<double> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(std::stod(token));
    }
    return tokens;
}

double compute_new_node_attribute(const std::vector<double>& node_data, const std::vector<double>& centroid_data, const std::string& similarity_type = "cosine") {
    double new_node_attribute;
    if (similarity_type == "cosine") {
        double dot_product = 0.0;
        double norm_node_data = 0.0;
        double norm_centroid_data = 0.0;
        for (size_t i = 0; i < node_data.size(); i++) {
            dot_product += node_data[i] * centroid_data[i];
            norm_node_data += node_data[i] * node_data[i];
            norm_centroid_data += centroid_data[i] * centroid_data[i];
        }
        norm_node_data = std::sqrt(norm_node_data);
        norm_centroid_data = std::sqrt(norm_centroid_data);
        new_node_attribute = 1 - (dot_product / (norm_node_data * norm_centroid_data));
    } else {
        double sum_of_squares = 0.0;
        for (size_t i = 0; i < node_data.size(); i++) {
            double diff = node_data[i] - centroid_data[i];
            sum_of_squares += diff * diff;
        }
        new_node_attribute = std::sqrt(sum_of_squares);
    }
    return new_node_attribute;
}

std::vector<double> compute_nodes_attribute(const std::string& cluster_file, std::vector<double>& centroid_data, const std::string& centroid_type = "mean", const std::string& similarity_type = "cosine") {
    std::ifstream file(cluster_file);
    std::vector<double> nodes_attribute;
    std::vector<std::vector<double>> nodes_data;
    if (file.is_open()) {
        std::string line;
        int n_rows = 0;
        if (std::getline(file, line)) {
            std::vector<double> node_data = split(line, ' ');
            for (size_t i = 0; i < node_data.size(); i++) {
                centroid_data.push_back(node_data[i]);
            }
            nodes_data.push_back(node_data);
            n_rows += 1;
        }
        while (std::getline(file, line)) {
            std::vector<double> node_data = split(line, ' ');
            for (size_t i = 0; i < node_data.size(); i++) {
                centroid_data[i] += node_data[i];
            }
            nodes_data.push_back(node_data);
            n_rows += 1;
        }
        for (size_t i = 0; i < centroid_data.size(); i++) {
            centroid_data[i] /= n_rows;
        }
    }
    for (const std::vector<double>& node_data : nodes_data) {
        double new_node_attribute = compute_new_node_attribute(node_data, centroid_data, similarity_type);
        nodes_attribute.push_back(new_node_attribute);
    }

    return nodes_attribute;
}

std::vector<std::vector<double>> calculate_centroids(const std::vector<std::string>& file_paths, const std::string& centroid_type = "mean", const std::string& similarity_type = "cosine") {
    printf("Similarity type: %s\n", similarity_type.c_str());
    std::vector<std::vector<double>> nodes_attributes_per_cluster;
    std::vector<std::vector<double>> centroids_per_cluster;
    printf("Nodes attributes per cluster initialized\n");
    for (const std::string& file_path : file_paths) {
        std::vector<double> centroid_vector;
        std::vector<double> nodes_attribute = compute_nodes_attribute(file_path, centroid_vector, centroid_type, similarity_type);
        nodes_attributes_per_cluster.push_back(nodes_attribute);
        centroids_per_cluster.push_back(centroid_vector);
        printf("Calculating nodes attributes and centroid for cluster: %s\n", file_path.c_str());
    }
    return centroids_per_cluster;
}

std::pair<int, std::string> find_similar_class(const std::vector<double>& new_feature_vector, std::vector<std::vector<double>>& centroids_per_cluster, const std::vector<std::string>& class_labels, const std::string& similarity_type = "cosine") {
    std::vector<double> new_node_attributes;
    int index = 0;
    // printf("Calculating current frame similarity values\n");
    for (const std::vector<double>& centroid_vector : centroids_per_cluster) {
        double new_node_attribute = compute_new_node_attribute(new_feature_vector, centroid_vector, similarity_type);
        new_node_attributes.push_back(new_node_attribute);
        // printf("Calculating new node's attribute w.r.t. cluster: %s\n", class_labels[index].c_str());
        index += 1;
    }
    // printf("Classifying ...\n");
    int class_index = std::distance(new_node_attributes.begin(), std::min_element(new_node_attributes.begin(), new_node_attributes.end()));
    // printf("Class index: %d\n", class_index);
    printf("Current frame class: %s\n", class_labels[class_index].c_str());
    return std::make_pair(class_index, class_labels[class_index]);
}

std::vector<std::string> load_model_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    TFLITE_MINIMAL_CHECK(file.is_open())
    printf("Label file loaded from %s\n", labels_file.c_str());
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str))
    {
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }
    file.close();
    return labels;
}

// Define a callback function for the trackbar
void onTrackbarChange(int sliderValue, void* userdata) {
    int* sliderStoredValue = static_cast<int*>(userdata);
    *sliderStoredValue = sliderValue;
}

int main(int argc, char **argv)
{
    // Get Model, label and input image
    const char *modelFileName = "/Users/kubotamacmini/Documents/cognitive_games/mobilenet_v3small-075-224-feature-vector.tflite";
    const char *labelFile = "/Users/kubotamacmini/Documents/cognitive_games/efficientnet_labels.txt";
    const char *imageFile = "camera";

    // Clusters data files
    std::vector<std::string> clusters_files_path = {
        "/Users/kubotamacmini/Documents/cognitive_games/L_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/L_2_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/L_3_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/C_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/C_2_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/C_3_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/None_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/None_letter_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/None_letter_2_vectors.txt",
        "/Users/kubotamacmini/Documents/cognitive_games/None_bg_vectors.txt"
    };
    std::vector<std::string> class_labels = {
        "L",
        "L",
        "L",
        "C",
        "C",
        "C",
        "None",
        "None",
        "None",
        "None"
    };
    std::string centroid_type = "mean";// In the meantime only mean is supported
    std::string similarity_type = "cosine";//"euclidean";
    std::vector<std::vector<double>> centroid_per_cluster = calculate_centroids(clusters_files_path, centroid_type, similarity_type);
    printf("Clusters Centroids calculated\n");

    // Camera activation and image loading
    std::vector<std::string> paths;
    bool readFromCamera = false;
    paths.push_back("/Users/kubotamacmini/Documents/cognitive_games/apple_above.jpg");
    readFromCamera = true;
    // Open the default camera
    cv::VideoCapture cap(0);
    if (readFromCamera && !cap.isOpened()) { // Check if the camera opened successfully
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }
    printf("Camera opened\n");

    // Classificationn model set up
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    printf("Model loaded \n");
    // Initiate Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter); //op: tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    printf("Interpreter initiated \n");
    // Configure the interpreter
    // interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(-1);
    // Choose a tensor from model by a tensor index
    /* Using mobilenet_v3small-075 class
    lite0-uint8
    Node  61 Operator Builtin Code   9 FULLY_CONNECTED (delegated by node 65)
    3 Input Tensors:[161,18,1] -> 0B (0.00MB)
    1 Output Tensors:[162] -> 0B (0.00MB)
    Node  62 Operator Builtin Code  25 SOFTMAX (not delegated)
    1 Input Tensors:[162] -> 1000B (0.00MB)
    1 Output Tensors:[163] -> 1000B (0.00MB)

    lite1-uint8
    Node  81 Operator Builtin Code   9 FULLY_CONNECTED (delegated by node 85)
    3 Input Tensors:[211,23,1] -> 0B (0.00MB)
    1 Output Tensors:[212] -> 0B (0.00MB)
    Node  82 Operator Builtin Code  25 SOFTMAX (not delegated)
    1 Input Tensors:[212] -> 1000B (0.00MB)
    1 Output Tensors:[213] -> 1000B (0.00MB)
    Tensor 212 efficientnet-lite1/mod... kTfLiteInt8     kTfLiteArenaRw     1000     / 0.00 [1,1000] [174080, 175080) <--- 
    Tensor 213 Softmax_int8              kTfLiteInt8     kTfLiteArenaRw     1000     / 0.00 [1,1000] [174080, 175080)
    Tensor 214 images                    kTfLiteUInt8    kTfLiteArenaRw     172800   / 0.16 [1,240,240,3] [0, 172800)
    Tensor 215 Softmax                   kTfLiteUInt8    kTfLiteArenaRw     1000     / 0.00 [1,1000] [172800, 173800)
    L_coin
    tensor_data_ptr[707]: -128
    tensor_data_ptr[708]: -128
    tensor_data_ptr[709]: -128
    tensor_data_ptr[710]: -128
    tensor_data_ptr[711]: -126
    tensor_data_ptr[712]: -128
    tensor_data_ptr[713]: -126
    tensor_data_ptr[714]: -128
    tensor_data_ptr[715]: -128
    tensor_data_ptr[716]: -128
    tensor_data_ptr[717]: -128
    tensor_data_ptr[718]: -128
    tensor_data_ptr[719]: -128
    apple
    tensor_data_ptr[707]: -126
    tensor_data_ptr[708]: -128
    tensor_data_ptr[709]: -128
    tensor_data_ptr[710]: -97
    tensor_data_ptr[711]: -128
    tensor_data_ptr[712]: -128
    tensor_data_ptr[713]: -128
    tensor_data_ptr[714]: -128
    tensor_data_ptr[715]: -128
    tensor_data_ptr[716]: -128
    tensor_data_ptr[717]: -128
    tensor_data_ptr[718]: -128
    tensor_data_ptr[719]: -1

    lite2-uint8
    Node  81 Operator Builtin Code   9 FULLY_CONNECTED (delegated by node 85)
    3 Input Tensors:[211,23,1] -> 0B (0.00MB)
    1 Output Tensors:[212] -> 0B (0.00MB)
    Node  82 Operator Builtin Code  25 SOFTMAX (not delegated)
    1 Input Tensors:[212] -> 1000B (0.00MB)
    1 Output Tensors:[213] -> 1000B (0.00MB)
    Tensor 212 efficientnet-lite2/mod... kTfLiteInt8     kTfLiteArenaRw     1000     / 0.00 [1,1000] [204096, 205096) <--- 
    Tensor 213 Softmax_int8              kTfLiteInt8     kTfLiteArenaRw     1000     / 0.00 [1,1000] [204096, 205096)
    Tensor 214 images                    kTfLiteUInt8    kTfLiteArenaRw     202800   / 0.19 [1,260,260,3] [0, 202800)
    Tensor 215 Softmax                   kTfLiteUInt8    kTfLiteArenaRw     1000     / 0.00 [1,1000] [202816, 203816)
    L_coin
    tensor_data_ptr[673]: -126
    tensor_data_ptr[674]: -128
    tensor_data_ptr[675]: -128
    tensor_data_ptr[676]: -128
    tensor_data_ptr[677]: -127
    tensor_data_ptr[678]: -128
    tensor_data_ptr[679]: -127
    tensor_data_ptr[680]: -128
    tensor_data_ptr[681]: -120
    tensor_data_ptr[682]: -127
    tensor_data_ptr[683]: -128
    apple
    tensor_data_ptr[673]: -126
    tensor_data_ptr[674]: -127
    tensor_data_ptr[675]: -128
    tensor_data_ptr[676]: -128
    tensor_data_ptr[677]: -128
    tensor_data_ptr[678]: -128
    tensor_data_ptr[679]: -128
    tensor_data_ptr[680]: -127
    tensor_data_ptr[681]: -128
    tensor_data_ptr[682]: -128
    tensor_data_ptr[683]: -128

    mobilenet_v1
    Node  29 Operator Builtin Code  22 RESHAPE (delegated by node 32)
    2 Input Tensors:[1,5] -> 0B (0.00MB)
    1 Output Tensors:[4] -> 0B (0.00MB)
    Node  30 Operator Builtin Code  25 SOFTMAX (not delegated)
    1 Input Tensors:[4] -> 1001B (0.00MB)
    1 Output Tensors:[87] -> 1001B (0.00MB)
    Tensor   4 MobilenetV1/Logits/Spa... kTfLiteUInt8    kTfLiteArenaRw     1001     / 0.00 [1,1001] [150528, 151529) <--- 
    L_coin
    tensor_data_ptr[500]: 102
    tensor_data_ptr[501]: 50
    tensor_data_ptr[502]: 104
    tensor_data_ptr[503]: 82
    tensor_data_ptr[504]: 96
    tensor_data_ptr[505]: 97
    tensor_data_ptr[506]: 105
    tensor_data_ptr[507]: 87
    tensor_data_ptr[508]: 111
    tensor_data_ptr[509]: 102
    tensor_data_ptr[510]: 70
    appple
    tensor_data_ptr[500]: 118
    tensor_data_ptr[501]: 28
    tensor_data_ptr[502]: 92
    tensor_data_ptr[503]: 97
    tensor_data_ptr[504]: 91
    tensor_data_ptr[505]: 100
    tensor_data_ptr[506]: 98
    tensor_data_ptr[507]: 111
    tensor_data_ptr[508]: 106
    tensor_data_ptr[509]: 94
    tensor_data_ptr[510]: 95
    
    mobilenet_v3small class
    Node 106 Operator Builtin Code   3 CONV_2D (delegated by node 111)
    3 Input Tensors:[219,77,111] -> 0B (0.00MB)
    1 Output Tensors:[220] -> 0B (0.00MB)
    1 Temporary Tensors:[302] -> 0B (0.00MB)
    Node 107 Operator Builtin Code 117 HARD_SWISH (delegated by node 111)
    1 Input Tensors:[220] -> 0B (0.00MB)
    1 Output Tensors:[221] -> 0B (0.00MB)
    Node 108 Operator Builtin Code   1 AVERAGE_POOL_2D (delegated by node 111)
    1 Input Tensors:[221] -> 0B (0.00MB)
    1 Output Tensors:[222] -> 0B (0.00MB)
    Node 109 Operator Builtin Code   3 CONV_2D (delegated by node 111)
    3 Input Tensors:[222,78,112] -> 0B (0.00MB)
    1 Output Tensors:[223] -> 0B (0.00MB)
    1 Temporary Tensors:[303] -> 0B (0.00MB)
    Node 110 Operator Builtin Code  22 RESHAPE (delegated by node 111)
    2 Input Tensors:[223,113] -> 0B (0.00MB)
    1 Output Tensors:[224] -> 0B (0.00MB)
    Node 111 Operator Custom Name TfLiteXNNPackDelegate 
    114 Input Tensors:[0-113] -> 8754368B (8.35MB) [075] || 10753440B (10.26MB) [100]
    1 Output Tensors:[224] -> 4004B (0.00MB)
    Tensor 219 predict/MobilenetV3/Av... kTfLiteFloat32  kTfLiteArenaRw     1728     / 0.00 [1,1,1,432] [-1, -1)
    Tensor 220 predict/MobilenetV3/Co... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1)
    Tensor 221 predict/MobilenetV3/Co... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1)
    Tensor 222 predict/MobilenetV3/Lo... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1) <--- 
    Tensor 223 predict/MobilenetV3/Lo... kTfLiteFloat32  kTfLiteArenaRw     4004     / 0.00 [1,1,1,1001] [-1, -1)
    Tensor 224 StatefulPartitionedCall:0 kTfLiteFloat32  kTfLiteArenaRw     4004     / 0.00 [1,1001] [602112, 606116)
        mobilenet_v3small feats
    Node 106 Operator Builtin Code   3 CONV_2D (delegated by node 110)
    3 Input Tensors:[217,77,110] -> 0B (0.00MB)
    1 Output Tensors:[218] -> 0B (0.00MB)
    1 Temporary Tensors:[299] -> 0B (0.00MB)
    Node 107 Operator Builtin Code 117 HARD_SWISH (delegated by node 110)
    1 Input Tensors:[218] -> 0B (0.00MB)
    1 Output Tensors:[219] -> 0B (0.00MB)
    Node 108 Operator Builtin Code   1 AVERAGE_POOL_2D (delegated by node 110)
    1 Input Tensors:[219] -> 0B (0.00MB)
    1 Output Tensors:[220] -> 0B (0.00MB)
    Node 109 Operator Builtin Code  22 RESHAPE (delegated by node 110)
    2 Input Tensors:[220,111] -> 0B (0.00MB)
    1 Output Tensors:[221] -> 0B (0.00MB)
    Node 110 Operator Custom Name TfLiteXNNPackDelegate 
    112 Input Tensors:[0-111] -> 4650268B (4.43MB) [075] || 6649340B (6.34MB) [100]
    1 Output Tensors:[221] -> 4096B (0.00MB)
    Tensor 217 predict/MobilenetV3/Av... kTfLiteFloat32  kTfLiteArenaRw     1728     / 0.00 [1,1,1,432] [-1, -1)
    Tensor 218 predict/MobilenetV3/Co... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1)
    Tensor 219 predict/MobilenetV3/Co... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1)
    Tensor 220 predict/MobilenetV3/Lo... kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1,1,1024] [-1, -1) <--- 
    Tensor 221 StatefulPartitionedCall:0 kTfLiteFloat32  kTfLiteArenaRw     4096     / 0.00 [1,1024] [602112, 606208)
    L_coin [075]
    tensor_data_ptr[990]: 0.184914
    tensor_data_ptr[991]: -0.337867
    tensor_data_ptr[992]: -0.374779
    tensor_data_ptr[993]: -0.066690
    tensor_data_ptr[994]: -0.374751
    tensor_data_ptr[995]: 0.897607
    tensor_data_ptr[996]: 0.044904
    tensor_data_ptr[997]: -0.160572
    tensor_data_ptr[998]: -0.280160
    tensor_data_ptr[999]: 0.325584
    apple [075]
    tensor_data_ptr[990]: 0.265381
    tensor_data_ptr[991]: -0.347447
    tensor_data_ptr[992]: -0.333938
    tensor_data_ptr[993]: -0.170890
    tensor_data_ptr[994]: -0.284572
    tensor_data_ptr[995]: -0.000000
    tensor_data_ptr[996]: 0.263300
    tensor_data_ptr[997]: -0.354430
    tensor_data_ptr[998]: -0.242586
    tensor_data_ptr[999]: -0.231432
    L_coin [100]
    tensor_data_ptr[990]: 0.296769
    tensor_data_ptr[991]: 0.285207
    tensor_data_ptr[992]: -0.325696
    tensor_data_ptr[993]: -0.357472
    tensor_data_ptr[994]: 0.729263
    tensor_data_ptr[995]: 0.615180
    tensor_data_ptr[996]: -0.099386
    tensor_data_ptr[997]: -0.301828
    tensor_data_ptr[998]: -0.316426
    tensor_data_ptr[999]: -0.303045
    appple [100]
    tensor_data_ptr[990]: -0.205722
    tensor_data_ptr[991]: 0.156332
    tensor_data_ptr[992]: 0.307186
    tensor_data_ptr[993]: -0.326140
    tensor_data_ptr[994]: -0.112252
    tensor_data_ptr[995]: -0.023291
    tensor_data_ptr[996]: -0.356689
    tensor_data_ptr[997]: -0.348391
    tensor_data_ptr[998]: -0.271443
    tensor_data_ptr[999]: 0.538711

    */
    int myTensorIndex = 221;
    // std::cout << "Enter the index of the Tensor before the Softmax layer [212 for lite1-uint8, lite2-uint8; 4 for mobilenet_v1, 221 for mobilenet_v3small]: ";
    // std::cin >> myTensorIndex;
    // Getting the tensor
    TfLiteTensor* myTensor = interpreter->tensor(myTensorIndex);
    // Get the number of dimensions in the tensor
    int numDims = myTensor->dims->size;
    // Get the number of elements in the tensor
    int numElements = 1;
    std::vector<int> tensorShape;    
    for (int i = 0; i < myTensor->dims->size; i++) {
        numElements *= myTensor->dims->data[i];
        tensorShape.push_back(myTensor->dims->data[i]);
        // printf("Dimension %d: %d\n", i, myTensor->dims->data[i]);
    }
    // Get the type of the tensor
    TfLiteType myTensorType = myTensor->type;
    int unit_memory = 1;
    switch (myTensorType) {
        case kTfLiteFloat32:
            unit_memory = sizeof(float32_t);
            break;
        case kTfLiteUInt8:
            unit_memory = sizeof(uint8_t);
            break;
        case kTfLiteInt8:
            unit_memory = sizeof(int8_t);
            break;
        default:
            fprintf(stderr, "cannot handle input type\n");
            exit(1);
    }

    // Inference set up
    std::vector<std::pair<float, int>> top_results; // Output tensor values
    float threshold = 0.1f; // Threshold for output tensor values
    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    // printf("Model input height, width, channels = %d, %d, %d \n", height, width, channels);

    // Placeholder for the current frame and properties
    cv::Mat frame; 
    frame = cv::imread(paths[0]);
    TFLITE_MINIMAL_CHECK(!frame.empty());
    if (!readFromCamera) printf("Placeholder image loaded from %s \n", paths[0].c_str());
    float cropProportionHeight = 0.7f;//0.9f; // Proportion of height to keep
    float cropProportionWidth = 0.5f;//0.6f; // Proportion of width to keep
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    int cropHeight = static_cast<int>(frameHeight * cropProportionHeight);
    int cropWidth = static_cast<int>(frameWidth * cropProportionWidth);

    // Interface setup
    int sliderOffsetX = 29;//35; // Slider value for X offset
    int storedOffsetX = sliderOffsetX;//35; // Slider value for X offset
    int sliderOffsetY = 22;//0; // Slider value for Y offset
    int storedOffsetY = sliderOffsetY;//0; // Slider value for Y offset
    int maxSliderValueX = int((1.0-cropProportionWidth)*100); // Maximum slider value for X offset
    int maxSliderValueY = int((1.0-cropProportionHeight)*100); // Maximum slider value for Y offset
    float addOffsetX = storedOffsetX / 100.0;
    float addOffsetY = storedOffsetY / 100.0;
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::String trackbarNameX = "Offset X:";
    cv::String trackbarNameY = "Offset Y:";
    cv::createTrackbar(trackbarNameX, "Frame", &sliderOffsetX, maxSliderValueX, onTrackbarChange, &storedOffsetX);
    cv::createTrackbar(trackbarNameY, "Frame", &sliderOffsetY, maxSliderValueY, onTrackbarChange, &storedOffsetY);
    int sliderValueShadow = 0; // Slider value for shadow reduction
    int storedValueShadow = sliderValueShadow; // Slider value for shadow reduction
    int sliderValueSaturation = 12; // Slider value for saturation
    int storedValueSaturation = sliderValueSaturation; // Slider value for saturation
    int maxSliderValueShadow = 200; // Maximum slider value for shadow reduction
    int maxSliderValueSaturation = 100; // Maximum slider value for saturation
    int sliderValueMultiply = 135;//148;//180;//260; // Slider value for multiply effect
    int storedValueMultiply = sliderValueMultiply; // Slider value for multiply effect
    int maxSliderValueMultiply = 800; // Maximum slider value for multiply effect
    cv::String trackbarNameShadow = "Shadows:";
    cv::String trackbarNameSaturation = "Saturation:";
    cv::createTrackbar(trackbarNameShadow, "Frame", &sliderValueShadow, maxSliderValueShadow, onTrackbarChange, &storedValueShadow);
    cv::createTrackbar(trackbarNameSaturation, "Frame", &sliderValueSaturation, maxSliderValueSaturation, onTrackbarChange, &storedValueSaturation);
    cv::String trackbarNameMultiply = "Multiply:";
    cv::createTrackbar(trackbarNameMultiply, "Frame", &sliderValueMultiply, maxSliderValueMultiply, onTrackbarChange, &storedValueMultiply);

    // Game variables set up
    auto user_time = 0;
    int user_score = 0;
    bool correct_ans = false;
    std::chrono::steady_clock::time_point challenge_start_time, challenge_end_time;
    bool game_start_flag = false;
    bool challenge_start_flag;
    ScenarioStage = 0000;
    DeviceCommand = 00;
    DeviceData = 00;
    DeviceStage = 0002;
    printf("Game variables set up\n");

    // Communication set up
    int fin=0;
    ISS_UDP_receive();
    ISSstate=-1;        //  not started
    pthread_t tid1;   // Thread ID
    pthread_create(&tid1, NULL, iss, (void *)NULL);
    printf("Communication set up\n");
    printf("First Connection attempted\n");

    bool app_start_flag = true;
    // Application loop
    while (app_start_flag){
        // Communication set up
        // ISS_UDP_receive();
        // Initialize game variables
        user_time = 0;
        user_score = 0;
        correct_ans = false;
        game_start_flag = false;
        challenge_start_flag = false;
        ScenarioStage = 0000;
        DeviceCommand = 00;
        DeviceData = 00;
        DeviceStage = 0002;
        // Open the default camera
        cv::VideoCapture cap(0);
        if (readFromCamera && !cap.isOpened()) { // Check if the camera opened successfully
            std::cerr << "Error: Unable to open camera" << std::endl;
            return -1;
        }
        if (ScenarioStage == 0002){
            DeviceCommand = 00;
            DeviceData = 00;
            DeviceStage = 0002;
            // Send
            printf("Received: %d. Returning same Scenario Stage\n", ScenarioStage);
        }
        else if (ScenarioStage == 3101){
            DeviceCommand = 06;
            DeviceData = 31;
            DeviceStage = 3102;
            game_start_flag = true;
            // Send
            printf("Received: %d. Starting Game\n", ScenarioStage);
        }
        while (game_start_flag){
            printf("Game running...\n");
            // Frame to use in loop
            cv::Mat frame_loop_init;
            // Get frame from camera
            if (readFromCamera) cap.read(frame_loop_init);
            // Rotate frame_loop
            cv::rotate(frame_loop_init, frame_loop_init, cv::ROTATE_180);
            // Zoom in frame_loop
            cv::Mat zoomed_frame_loop;
            cv::resize(frame_loop_init, zoomed_frame_loop, cv::Size(), 1.5, 1.5, cv::INTER_LINEAR);
            frame_loop_init = zoomed_frame_loop(cv::Rect((zoomed_frame_loop.cols - frame_loop_init.cols) / 2, (zoomed_frame_loop.rows - frame_loop_init.rows) / 2, frame_loop_init.cols, frame_loop_init.rows));
            if (readFromCamera && frame_loop_init.empty()) {
                printf("Failed to capture frame from camera\n");
                return -1;
            }
            // Apply shadow reduction
            cv::Mat frame_loop_shadow;
            cv::addWeighted(frame_loop_init, 1.0, cv::Scalar(storedValueShadow - maxSliderValueShadow/2), 0.0, 0.0, frame_loop_shadow);
            // Apply saturation
            cv::Mat frame_loop_saturation;
            cv::cvtColor(frame_loop_shadow, frame_loop_saturation, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(frame_loop_saturation, channels);
            channels[1] = channels[1] * ((storedValueSaturation) / 100.0);
            cv::merge(channels, frame_loop_saturation);
            // Apply multiply effect
            cv::Mat frame_loop_multiply;
            cv::cvtColor(frame_loop_saturation, frame_loop_multiply, cv::COLOR_HSV2BGR);
            cv::multiply(frame_loop_multiply, cv::Scalar(storedValueMultiply / 100.0, storedValueMultiply / 100.0, storedValueMultiply / 100.0), frame_loop_init);
            addOffsetX = (storedOffsetX - maxSliderValueX/2) / 100.0;
            addOffsetY = (storedOffsetY - maxSliderValueY/2) / 100.0;
            int cropOffsetX = int(((frameWidth - cropWidth) / 2) + addOffsetX*frameWidth);
            int cropOffsetY = int(((frameHeight - cropHeight) / 2) + addOffsetY*frameHeight);
            cv::Mat frame_loop_wrect = frame_loop_init.clone();
            cv::rectangle(frame_loop_wrect, cv::Point(cropOffsetX, cropOffsetY), cv::Point(cropOffsetX + cropWidth, cropOffsetY + cropHeight), cv::Scalar(0, 255, 0), 2);
            cv::imshow("Frame", frame_loop_wrect);

            if (ScenarioStage == 3103){
                challenge_start_flag = true;
                challenge_start_time = std::chrono::steady_clock::now();
                printf("Received: %d. Challenge starting\n", ScenarioStage);
            }
            else if(ScenarioStage == 3105){
                DeviceCommand = 8;
                DeviceData = 00;
                DeviceStage = 0004;
                // Send
                game_start_flag = false;
                printf("Received: %d. Game finishing\n", ScenarioStage);
            }
            // Inference Loop
            std::vector<std::string> class_verification_list;
            while (challenge_start_flag){
                // Allocate tensor buffers.
                TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

                // Clone the frame and draw the rectangle of the region to be cropped
                cv::Mat frame_loop;// = frame.clone();
                // Get frame from camera
                if (readFromCamera) cap.read(frame_loop);
                // Rotate frame_loop
                cv::rotate(frame_loop, frame_loop, cv::ROTATE_180);
                // Zoom in frame_loop
                cv::Mat zoomed_frame_loop;
                cv::resize(frame_loop, zoomed_frame_loop, cv::Size(), 1.5, 1.5, cv::INTER_LINEAR);
                frame_loop = zoomed_frame_loop(cv::Rect((zoomed_frame_loop.cols - frame_loop.cols) / 2, (zoomed_frame_loop.rows - frame_loop.rows) / 2, frame_loop.cols, frame_loop.rows));
                if (readFromCamera && frame_loop.empty()) {
                    printf("Failed to capture frame from camera\n");
                    return -1;
                }
                // Apply shadow reduction
                cv::Mat frame_loop_shadow;
                cv::addWeighted(frame_loop, 1.0, cv::Scalar(storedValueShadow - maxSliderValueShadow/2), 0.0, 0.0, frame_loop_shadow);
                // Apply saturation
                cv::Mat frame_loop_saturation;
                cv::cvtColor(frame_loop_shadow, frame_loop_saturation, cv::COLOR_BGR2HSV);
                std::vector<cv::Mat> channels;
                cv::split(frame_loop_saturation, channels);
                channels[1] = channels[1] * ((storedValueSaturation) / 100.0);
                cv::merge(channels, frame_loop_saturation);
                // Apply multiply effect
                cv::Mat frame_loop_multiply;
                cv::cvtColor(frame_loop_saturation, frame_loop_multiply, cv::COLOR_HSV2BGR);
                cv::multiply(frame_loop_multiply, cv::Scalar(storedValueMultiply / 100.0, storedValueMultiply / 100.0, storedValueMultiply / 100.0), frame_loop);
                addOffsetX = (storedOffsetX - maxSliderValueX/2) / 100.0;
                addOffsetY = (storedOffsetY - maxSliderValueY/2) / 100.0;
                int cropOffsetX = int(((frameWidth - cropWidth) / 2) + addOffsetX*frameWidth);
                int cropOffsetY = int(((frameHeight - cropHeight) / 2) + addOffsetY*frameHeight);
                cv::Mat frame_loop_wrect = frame_loop.clone();
                cv::rectangle(frame_loop_wrect, cv::Point(cropOffsetX, cropOffsetY), cv::Point(cropOffsetX + cropWidth, cropOffsetY + cropHeight), cv::Scalar(0, 255, 0), 2);

                frame = frame_loop;
                // Crop region
                cv::Rect cropRegion(cropOffsetX, cropOffsetY, cropWidth, cropHeight);
                // Crop frame
                frame = frame(cropRegion);
                
                // Crop the corners of the image in the shape of triangles
                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::Point pts[4][3];
                pts[0][0] = cv::Point(0, 0);
                pts[0][1] = cv::Point(0, int(cropHeight/5));
                pts[0][2] = cv::Point(cropWidth/6, 0);
                pts[1][0] = cv::Point(cropWidth, 0);
                pts[1][1] = cv::Point(cropWidth, cropHeight/5);
                pts[1][2] = cv::Point(cropWidth - cropWidth/6, 0);
                pts[2][0] = cv::Point(0, cropHeight);
                pts[2][1] = cv::Point(0, cropHeight - cropHeight/5);
                pts[2][2] = cv::Point(cropWidth/6, cropHeight);
                pts[3][0] = cv::Point(cropWidth, cropHeight);
                pts[3][1] = cv::Point(cropWidth, cropHeight - cropHeight/5);
                pts[3][2] = cv::Point(cropWidth - cropWidth/6, cropHeight);
                std::vector<cv::Point> poly1 = {pts[0][0], pts[0][1], pts[0][2]};
                std::vector<cv::Point> poly2 = {pts[1][0], pts[1][1], pts[1][2]};
                std::vector<cv::Point> poly3 = {pts[2][0], pts[2][1], pts[2][2]};
                std::vector<cv::Point> poly4 = {pts[3][0], pts[3][1], pts[3][2]};
                std::vector<std::vector<cv::Point>> polygons = {poly1, poly2, poly3, poly4};
                cv::fillPoly(mask, polygons, cv::Scalar(255));
                frame.setTo(cv::Scalar(0), mask);

                // Copy image to input tensor size
                cv::Mat image;
                cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
                // printf("Image resized to %dx%d \n", width, height);
                if (myTensorIndex>220){
                    // Normalize image from 0 to 1
                    image.convertTo(image, CV_32F, 1.0 / 255.0);
                }
                switch (interpreter->tensor(input)->type)
                {
                case kTfLiteFloat32:
                    memcpy(interpreter->typed_input_tensor<float32_t>(0), image.data, image.total() * image.elemSize());
                    break;
                case kTfLiteUInt8:
                    memcpy(interpreter->typed_input_tensor<uint8_t>(0), image.data, image.total() * image.elemSize());
                    break;
                case kTfLiteInt8:
                    memcpy(interpreter->typed_input_tensor<uint8_t>(0), image.data, image.total() * image.elemSize());
                    break;
                default:
                    fprintf(stderr, "cannot handle input type\n");
                    exit(1);
                }
                // printf("Image copied to first input tensor\n");

                // Inference
                // std::chrono::steady_clock::time_point start, end;
                // start = std::chrono::steady_clock::now();
                TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);//interpreter->Invoke();
                // end = std::chrono::steady_clock::now();
                // inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                // printf("End of inference. Inference time : %s \n", std::to_string(inference_time).c_str());

                // Get Output tensor index
                int output = interpreter->outputs()[0];
                TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
                auto output_size = output_dims->data[output_dims->size - 1];
                // Get the model classificaiton results
                // std::vector<std::pair<float, int>> top_results;
                // float threshold = 0.01f;
                switch (interpreter->tensor(output)->type)
                {
                case kTfLiteInt32:
                    tflite::label_image::get_top_n<float32_t>(interpreter->typed_output_tensor<float32_t>(0), output_size, 1, threshold, &top_results, kTfLiteFloat32);
                    break;
                case kTfLiteFloat32:
                    tflite::label_image::get_top_n<float32_t>(interpreter->typed_output_tensor<float32_t>(0), output_size, 1, threshold, &top_results, kTfLiteFloat32);
                    break;
                case kTfLiteUInt8:
                    tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
                    break;
                case kTfLiteInt8:
                    tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
                    break;
                default:
                    fprintf(stderr, "cannot handle output type\n");
                    exit(1);
                }

                // Vector to save the tensor values
                std::vector<double> tensor_vector;
                // Retrieve selected tensor values
                if (myTensorType == kTfLiteUInt8 && numDims == 2) {
                    printf("Type of the selected tensor [%d]: kTfLiteUInt8\n", myTensorIndex);
                    // Retrieve output tensor values
                    uint8_t* tensor_data_ptr = interpreter->typed_tensor<uint8_t>(myTensorIndex);
                    // Print the some elements of the tensor
                    printf("Some elements of the tensor [%d]:\n", myTensorIndex);
                    for (int i = 500; i < 1000; i++) {
                        printf("tensor_data_ptr[%d]: %d\n", i, tensor_data_ptr[i]);
                    }
                } else if (myTensorType == kTfLiteInt8 && numDims == 2) {
                    printf("Type of the selected tensor [%d]: kTfLiteInt8\n", myTensorIndex);
                    // Retrieve output tensor values
                    int8_t* tensor_data_ptr = interpreter->typed_tensor<int8_t>(myTensorIndex);
                    // Print the some elements of the tensor
                    printf("Some elements of the tensor [%d]:\n", myTensorIndex);
                    for (int i = 500; i < 1000; i++) {
                        printf("tensor_data_ptr[%d]: %d\n", i, tensor_data_ptr[i]);
                    }
                } else if (myTensorType == kTfLiteFloat32 && numDims == 2) {
                    // printf("Type of the selected tensor [%d]: kTfLiteFloat32\n", myTensorIndex);
                    // Retrieve output tensor values
                    float32_t* tensor_data_ptr = interpreter->typed_tensor<float32_t>(myTensorIndex);
                    for (int i = 0; i < numElements; i++) {
                        tensor_vector.push_back(tensor_data_ptr[i]);
                    }
                    // std::ofstream outputFile("/Users/kubotamacmini/Documents/cognitive_games/curr_vector.txt", std::ios::app);
                    // if (outputFile.is_open()) {
                    //     for (int i = 0; i < numElements; i++) {
                    //         if (myTensorType == kTfLiteFloat32) {
                    //             outputFile << std::fixed << std::setprecision(6) << tensor_data_ptr[i] << " ";
                    //         } else {
                    //             outputFile << tensor_data_ptr[i] << " ";
                    //         }
                    //     }
                    //     outputFile << "\n";
                    //     outputFile.close();
                    //     // printf("Tensor saved to file: curr_vector.txt\n");
                    // } else {
                    //     printf("Failed to open file: curr_vector.txt\n");
                    // }
                }
                // Clean up the interpreter
                interpreter->ResetVariableTensors();

                // Classify the image features vector
                std::pair<int, std::string> similar_class = find_similar_class(tensor_vector, centroid_per_cluster, class_labels);
                std::string current_class = similar_class.second;

                cv::putText(frame_loop_wrect, "Figure recognized as "+current_class, cv::Point(0.45*frameWidth , 0.9*frameHeight), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::imshow("Frame", frame_loop_wrect);

                if (class_verification_list.size() == 4) {
                    class_verification_list.erase(class_verification_list.begin());
                }
                class_verification_list.push_back(current_class);

                if (std::all_of(class_verification_list.begin(), class_verification_list.end(), [](const std::string& cls) { return cls == "L"; })) {
                    printf("Challenge completed\n");
                    correct_ans = true;
                    challenge_end_time = std::chrono::steady_clock::now();
                    user_time = std::chrono::duration_cast<std::chrono::milliseconds>(challenge_end_time - challenge_start_time).count();
                    challenge_start_flag = false;
                }
                
                int key = cv::waitKey(1);
                // Check if key is pressed to exit the loop
                if (key == 'q') {
                // frame = frame_loop;
                break;
                }
                if (key == 'e') {
                printf("Challenge completed\n");
                correct_ans = true;
                challenge_end_time = std::chrono::steady_clock::now();
                user_time = std::chrono::duration_cast<std::chrono::milliseconds>(challenge_end_time - challenge_start_time).count();
                challenge_start_flag = false;
                }
            }
            if (correct_ans){
                challenge_start_flag = false;
                if (user_time < 4000) {
                    // Time is less than 4000 ms
                    user_score = 20;
                } else if (user_time < 5000) {
                    // Time is between 4000 ms and 5000 ms
                    user_score = 18;
                } else if (user_time < 6000) {
                    // Time is between 5000 ms and 6000 ms
                    user_score = 15;
                } else if (user_time < 7000) {
                    // Time is between 6000 ms and 7000 ms
                    user_score = 10;
                } else {
                    // Time is greater than or equal to 7000 ms
                    user_score = 5;
                }
                DeviceCommand = 7;
                DeviceData = user_score;
                DeviceStage = 3104;
                // Send
                printf("User Score is: %d\n", user_score);
                while(ScenarioStage == 3103);
            }
            int key = cv::waitKey(1);
                // Check if key is pressed to exit the loop
            if (key == 'q') {
                // frame = frame_loop;
                app_start_flag = false;
                cap.release(); // Release the camera
                cv::destroyAllWindows();
                break;
            }
            if (key == 'w') {
                // frame = frame_loop;
                challenge_start_flag = true;
                challenge_start_time = std::chrono::steady_clock::now();
                printf("Challenge starting\n");
            }
            if (key == 'e') {
                // frame = frame_loop;
                game_start_flag = false;
                printf("Game finishing\n");
            }
        }
        if (correct_ans){
            cap.release(); // Release the camera
            cv::destroyAllWindows();
            printf("Game stopped\n");
            correct_ans = false;
            // ScenarioStage = "9999";
        }
        int key_0 = cv::waitKey(1);
        if (key_0 == 'q') {
            // frame = frame_loop;
            app_start_flag = false;
            cap.release();
        }
    }
    printf("Application finished\n");
    if (paths.size() == 1) {
        // Print inference ms in input image
        // cv::putText(frame, "Inference Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        // Load Labels
        auto labels = load_model_labels(labelFile);

        // Print labels with confidence in input image
        for (const auto &result : top_results)
        {
            const float confidence = result.first;
            const int index = result.second;
            std::string output_txt_1 = "Label :" + labels[index];
            std::string output_txt_2 = "Confidence : " + std::to_string(confidence);
            cv::putText(frame, output_txt_1, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, output_txt_2, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            break;
        }

        // Display image
        cv::imshow("Output", frame);
        cv::waitKey(0);
    }
    // pthread_join(tid1, NULL);
    pthread_cancel(tid1);

    return 0;
}