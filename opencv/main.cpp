#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to rotate an image 90 degrees clockwise
Mat rotateImage90CW(const Mat& src) {
    double start = static_cast<double>(getTickCount());

    Mat dst;
    transpose(src, dst);
    flip(dst, dst, 1); // 1 means flipping around y-axis, 0 means flipping around x-axis

    double duration = (static_cast<double>(getTickCount()) - start) / getTickFrequency();
    cout << "Rotation Time: " << duration * 1000 << "milli seconds." << endl;

    return dst;
}

int main() {
    // Read the image file
    Mat image = imread("./testBig.jpg");
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Rotate the image manually
    Mat rotatedImage = rotateImage90CW(image);

    // Create windows to display images
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Rotated Image", WINDOW_AUTOSIZE);

    // Show images
    imshow("Original Image", image);
    imshow("Rotated Image", rotatedImage);

    // Wait for any key press
    waitKey(0);

    return 0;
}
