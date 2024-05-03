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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
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
