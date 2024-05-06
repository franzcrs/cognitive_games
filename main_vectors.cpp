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

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::vector<std::string> load_labels(std::string labels_file)
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

    // Get Model label and input image
    if (argc != 4)
    {
        fprintf(stderr, "Run as: ./main modelfile labels image\n");
        exit(1);
    }
    // const char *modelFileName = "/Users/kubotamacmini/Documents/cognitive_games/mobilenet_v3_large-075-224-feature-vector.tflite";//argv[1];
    // const char *labelFile = "/Users/kubotamacmini/Documents/cognitive_games/imagenet_labels.txt";//argv[2];
    // const char *imageFile = "/Users/kubotamacmini/Documents/camera/build/None/photo_1714112877.jpg";//argv[3];
    
    const char *modelFileName = argv[1];
    const char *labelFile = argv[2];
    const char *imageFile = argv[3];

    std::vector<std::string> paths;
    bool readFromCamera = false;
    // Check if imageFile is a file or a folder
    if (std::filesystem::is_regular_file(imageFile))
    {
        // imageFile is a file
        paths.push_back(imageFile);
        printf("Image file path loaded \n");
    }
    else if (std::filesystem::is_directory(imageFile))
    {
        // imageFile is a folder
        for (const auto &entry : std::filesystem::directory_iterator(imageFile))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg")
            {
                paths.push_back(entry.path().string());
            }
        }
        printf("Image file paths contained in folder loaded \n");
    }
    else
    {
        // imageFile does not exist or is neither a file nor a folder
        fprintf(stderr, "Invalid image file or folder. Changing to camera source.\n");
        paths.push_back("/Users/kubotamacmini/Documents/cognitive_games/apple_above.jpg");
        readFromCamera = true;
    }
    
    // Open the default camera
    cv::VideoCapture cap(0);
    if (readFromCamera && !cap.isOpened()) { // Check if the camera opened successfully
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Load Model
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
    int myTensorIndex;
    std::cout << "Enter the index of the Tensor before the Softmax layer [212 for lite1-uint8, lite2-uint8; 4 for mobilenet_v1, 221 for mobilenet_v3small]: ";
    std::cin >> myTensorIndex;
    // Getting the tensor
    TfLiteTensor* myTensor = interpreter->tensor(myTensorIndex);
    // Get the number of dimensions in the tensor
    int numDims = myTensor->dims->size;
    // Print the number of dimensions in the tensor
    printf("Number of dimensions in the tensor [%d]: %d\n", myTensorIndex, numDims);
    if (numDims > 2) {
        printf("Tensor is not 2D. Feature vector values will not be shown. \nMost probably you have inserted a classification mobilenet model instead of the features version\n");
    }
    // Get the number of elements in the tensor
    int numElements = 1;
    std::vector<int> tensorShape;    
    for (int i = 0; i < myTensor->dims->size; i++) {
        numElements *= myTensor->dims->data[i];
        tensorShape.push_back(myTensor->dims->data[i]);
        // printf("Dimension %d: %d\n", i, myTensor->dims->data[i]);
    }
    // Print the number of elements in the tensor
    printf("Number of elements in the tensor: [%d]: %d\n", myTensorIndex, numElements);
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
    // Set custom allocation for tensor of index tensorIndex
    // interpreter->SetTensorParametersReadWrite(myTensorIndex, myTensorType, "MyTensor", tensorShape, {607000, 607000+unit_memory*numElements});
    // printf("Tensor [%d] set to custom allocation \n", myTensorIndex);

    // Run inferences for all images in the paths vector
    auto inference_time = 0; // One interation inference time
    std::vector<std::pair<float, int>> top_results; // Output tensor values
    float threshold = 0.01f; // Threshold for output tensor values
    cv::Mat frame; // Placeholder for the current frame
    frame = cv::imread(paths[0]);
    float cropProportionHeight = 0.9f; // Proportion of height to keep
    float cropProportionWidth = 0.6f; // Proportion of width to keep
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    int cropHeight = static_cast<int>(frameHeight * cropProportionHeight);
    int cropWidth = static_cast<int>(frameWidth * cropProportionWidth);
    int numIters = 0; // Number of iterations
    for (const auto& imagePath : paths) {
        printf("********** Iteration start ********** \n");
        // Allocate tensor buffers.
        TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
        printf("Interpreter Tensors could be allocated \n");

        // Get Input Tensor Dimensions
        int input = interpreter->inputs()[0];
        auto height = interpreter->tensor(input)->dims->data[1];
        auto width = interpreter->tensor(input)->dims->data[2];
        auto channels = interpreter->tensor(input)->dims->data[3];
        // printf("Model input height, width, channels = %d, %d, %d \n", height, width, channels);
        
        // Load Input Image
        frame = cv::imread(imagePath);
        TFLITE_MINIMAL_CHECK(!frame.empty());
        if (!readFromCamera) printf("Image loaded from %s \n", imagePath.c_str());
        int sliderOffsetX = 13;//35; // Slider value for X offset
        int storedOffsetX = sliderOffsetX;//35; // Slider value for X offset
        int sliderOffsetY = 10;//0; // Slider value for Y offset
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
        int sliderValueMultiply = 148;//180;//260; // Slider value for multiply effect
        int storedValueMultiply = sliderValueMultiply; // Slider value for multiply effect
        int maxSliderValueMultiply = 800; // Maximum slider value for multiply effect
        cv::String trackbarNameShadow = "Shadows:";
        cv::String trackbarNameSaturation = "Saturation:";
        cv::createTrackbar(trackbarNameShadow, "Frame", &sliderValueShadow, maxSliderValueShadow, onTrackbarChange, &storedValueShadow);
        cv::createTrackbar(trackbarNameSaturation, "Frame", &sliderValueSaturation, maxSliderValueSaturation, onTrackbarChange, &storedValueSaturation);
        cv::String trackbarNameMultiply = "Multiply:";
        cv::createTrackbar(trackbarNameMultiply, "Frame", &sliderValueMultiply, maxSliderValueMultiply, onTrackbarChange, &storedValueMultiply);
        bool inferenceButtonPressed = false;
        while (true) {
            // Clone the frame and draw the rectangle of the region to be cropped
            cv::Mat frame_loop = frame.clone();
            // Get frame from camera
            if (readFromCamera) cap.read(frame_loop);
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
            
            // cv::imshow("Frame", frame_loop_wrect);
            frame = frame_loop;
            break;
            
            int key = cv::waitKey(1);
            // Check if key is pressed to exit the loop
            if (key == 'q') {
            break;
            } else if (key == 'i') {
            frame = frame_loop;
            inferenceButtonPressed = true;
            break;
            }
        }
        cap.release(); // Release the camera
        cv::destroyAllWindows();
        printf("Sliders stored values: \nShadow: %d, Saturation: %d, Multiply: %d \n", storedValueShadow, storedValueSaturation, storedValueMultiply);
        
        // // Apply shadow reduction
        // cv::Mat frame_loop_shadow;
        // cv::addWeighted(frame, 1.0, cv::Scalar(storedValueShadow - maxSliderValueShadow/2), 0.0, 0.0, frame_loop_shadow);
        // // Apply saturation
        // cv::Mat frame_loop_saturation;
        // cv::cvtColor(frame_loop_shadow, frame_loop_saturation, cv::COLOR_BGR2HSV);
        // std::vector<cv::Mat> channels_sat;
        // cv::split(frame_loop_saturation, channels_sat);
        // channels_sat[1] = channels_sat[1] * ((storedValueSaturation) / 100.0);
        // cv::merge(channels_sat, frame_loop_saturation);
        // // Apply multiply effect
        // cv::Mat frame_loop_multiply;
        // cv::cvtColor(frame_loop_saturation, frame_loop_multiply, cv::COLOR_HSV2BGR);
        // cv::multiply(frame_loop_multiply, cv::Scalar(storedValueMultiply / 100.0, storedValueMultiply / 100.0, storedValueMultiply / 100.0), frame);
        // frame_loop_shadow.release();
        // frame_loop_saturation.release();
        // frame_loop_multiply.release();
        addOffsetX = (storedOffsetX - maxSliderValueX/2) / 100.0;
        addOffsetY = (storedOffsetY - maxSliderValueY/2) / 100.0;
        int cropOffsetX = int(((frameWidth - cropWidth) / 2) + addOffsetX*frameWidth);
        int cropOffsetY = int(((frameHeight - cropHeight) / 2) + addOffsetY*frameHeight);
        cv::Rect cropRegion(cropOffsetX, cropOffsetY, cropWidth, cropHeight); // Crop region
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
        
        // // Load names of classes
        // std::string classesFile = "coco.names";
        // std::ifstream ifs(classesFile.c_str());
        // std::string line;
        // std::vector<std::string> classes;
        // while (getline(ifs, line)) classes.push_back(line);
        // // Load the object detection model
        // cv::dnn::Net model = cv::dnn::readNetFromDarknet("/Users/kubotamacmini/Documents/cognitive_games/yolov3.cfg", "/Users/kubotamacmini/Documents/cognitive_games/yolov3.weights");
        // printf("Yolo model loaded \n");
        // // Create a blob from the frame
        // cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        // // Set the input blob for the model
        // model.setInput(blob);
        // printf("Blob set as input \n");
        // // Get the output layer names
        // std::vector<cv::String> outputLayerNames = model.getUnconnectedOutLayersNames();
        // // Forward pass through the model
        // std::vector<cv::Mat> outputs;
        // model.forward(outputs, outputLayerNames);
        // // Process the outputs
        // for (const cv::Mat& output : outputs) {
        //     // Process each detection
        //     for (int i = 0; i < output.rows; i++) {
        //         cv::Mat detection = output.row(i);
        //         // Extract the class ID, confidence, and bounding box coordinates
        //         int classId;
        //         float confidence;
        //         cv::Rect bbox;
        //         float* data = (float*)detection.data;
        //         classId = static_cast<int>(data[1]);
        //         confidence = data[2];
        //         bbox.x = static_cast<int>(data[3] * frame.cols);
        //         bbox.y = static_cast<int>(data[4] * frame.rows);
        //         bbox.width = static_cast<int>(data[5] * frame.cols);
        //         bbox.height = static_cast<int>(data[6] * frame.rows);
        //         // Filter detections based on confidence threshold
        //         if (confidence > 0.0) {
        //             // Draw bounding box and class label on the frame
        //             cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
        //             cv::putText(frame, cv::format("%s: %.2f", classes[classId].c_str(), confidence), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        //         }
        //     }
        // }
        // // Show the frame with detections
        // cv::imshow("Frame with Detections", frame);
        // cv::waitKey(0);

        // // Convert image to grayscale
        // cv::Mat grayImage;
        // cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
        // // Apply binary thresholding
        // cv::Mat binaryImage;
        // cv::threshold(grayImage, binaryImage, 120, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        // // Find contours in the binary image
        // std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        // // Filter contours based on size
        // std::vector<std::vector<cv::Point>> filteredContours;
        // int frameArea = frame.cols * frame.rows;
        // int minContourArea = frameArea * 0.01; // Minimum contour area as a percentage of the frame area
        // int maxContourArea = frameArea * 0.9; // Maximum contour area as a percentage of the frame area
        // for (const auto& contour : contours) {
        //     double contourArea = cv::contourArea(contour);
        //     if (contourArea > minContourArea && contourArea < maxContourArea) {
        //     filteredContours.push_back(contour);
        //     }
        // }
        // // Find the contour with the largest area
        // double maxArea = 0;
        // int maxAreaIdx = -1;
        // for (int i = 0; i < filteredContours.size(); i++) {
        //     double area = cv::contourArea(filteredContours[i]);
        //     if (area > maxArea) {
        //         maxArea = area;
        //         maxAreaIdx = i;
        //     }
        // }
        // // Enclose the contour with a rectangle
        // if (maxAreaIdx != -1) {
        //     cv::Rect boundingRect = cv::boundingRect(contours[maxAreaIdx]);
        //     cv::Mat frame_wrect = frame.clone();
        //     cv::rectangle(frame_wrect, boundingRect, cv::Scalar(0, 0, 255), 2);
        //     cv::imshow("Frame with Rectangle", frame_wrect);
        //     cv::waitKey(0);
        // }

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
        std::chrono::steady_clock::time_point start, end;
        start = std::chrono::steady_clock::now();
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);//interpreter->Invoke();
        end = std::chrono::steady_clock::now();
        inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("End of inference. Inference time : %s \n", std::to_string(inference_time).c_str());

        // Get Output
        int output = interpreter->outputs()[0];
        TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
        auto output_size = output_dims->data[output_dims->size - 1];
        printf("Output size: %d \n", output_size);

        // Get the output tensor values
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
        // printf("Top results:\n");
        // for(const auto& result : top_results) {
        //     printf("Value: %g, Index: %d\n", result.first, result.second);
        // }

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
            printf("Type of the selected tensor [%d]: kTfLiteFloat32\n", myTensorIndex);
            // Retrieve output tensor values
            float32_t* tensor_data_ptr = interpreter->typed_tensor<float32_t>(myTensorIndex);
            // Save tensor to file
            std::ofstream outputFile("/Users/kubotamacmini/Documents/cognitive_games/vectors.txt", std::ios::app);
            if (outputFile.is_open()) {
                for (int i = 0; i < numElements; i++) {
                    if (myTensorType == kTfLiteFloat32) {
                        outputFile << std::fixed << std::setprecision(6) << tensor_data_ptr[i] << " ";
                    } else {
                        outputFile << tensor_data_ptr[i] << " ";
                    }
                }
                outputFile << "\n";
                outputFile.close();
                printf("Tensor saved to file: vectors.txt\n");
            } else {
                printf("Failed to open file: vectors.txt\n");
            }
            // Print the some elements of the tensor
            // printf("Some elements of the tensor [%d]:\n", myTensorIndex);
            // for (int i = 500; i < 1000; i++) {
            //     printf("tensor_data_ptr[%d]: %f\n", i, tensor_data_ptr[i]);
            // }
        }

        // Clean up the interpreter
        interpreter->ResetVariableTensors();
        numIters += 1;
        printf("********** Iteration end ********** \n");
    }
    printf("Number of iterations: %d\n", numIters);
    
    
    if (paths.size() == 1) {
        // Print inference ms in input image
        cv::putText(frame, "Inference Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        // Load Labels
        auto labels = load_labels(labelFile);

        // Print labels with confidence in input image
        for (const auto &result : top_results)
        {
            const float confidence = result.first;
            const int index = result.second;
            std::string output_txt_1 = "Label :" + labels[index];
            std::string output_txt_2 = "Confidence : " + std::to_string(confidence);
            cv::putText(frame, output_txt_1, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, output_txt_2, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        // Display image
        cv::imshow("Output", frame);
        cv::waitKey(0);
    }
    
    return 0;
}