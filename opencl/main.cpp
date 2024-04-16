#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <CL/cl.h>

using namespace cv;
using namespace std;


const char* rotateKernelSource = R"(
__kernel void rotate90CW(
    __global const uchar* inputImage,
    __global uchar* outputImage,
    const int width,
    const int height)
{
    int globalIndex = get_global_id(0);
    int pixelIndex = globalIndex / 3;
    int channel = globalIndex % 3;  // Determines the color channel (R, G, B)

    int x = pixelIndex % width;
    int y = pixelIndex / width;
    int newY = width - 1 - x;  // New y-coordinate
    int newX = y;              // New x-coordinate

    int inputIdx = globalIndex;
    int outputIdx = (newX + width * newY) * 3 + channel; // Index for specific channel in the output image

    // Copy the color component
    outputImage[outputIdx] = inputImage[inputIdx];
}
)";

int main() {
    // Read the image file
    Mat src = imread("./testBig.jpg");
    if (src.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    size_t srcSize = src.total() * src.elemSize();
    size_t dstSize = srcSize;
    Mat dst(src.rows, src.cols, src.type()); // Swap rows and cols

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem srcBuffer, dstBuffer;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Create and build program
    program = clCreateProgramWithSource(context, 1, &rotateKernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    kernel = clCreateKernel(program, "rotate90CW", &err);


    // Create buffers
    srcBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, srcSize, src.data, &err);
    dstBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dstSize, NULL, &err);


    while(true)
    {

        double start = static_cast<double>(getTickCount());

        // Set kernel arguments
        clEnqueueWriteBuffer(queue, srcBuffer, CL_TRUE, 0, srcSize, src.data, 0, NULL, NULL);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcBuffer);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dstBuffer);
        err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
        err = clSetKernelArg(kernel, 3, sizeof(int), &src.rows);

        // Define an ND range
        size_t globalSize[1] = { (size_t)src.cols * src.rows * 3 };  // Now considering each byte as a separate item
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);
        clFinish(queue);
        // Read the output back to host memory
        clEnqueueReadBuffer(queue, dstBuffer, CL_TRUE, 0, dstSize, dst.data, 0, NULL, NULL);
        // Wait for the kernel to complete
        clFinish(queue);
        double duration = (static_cast<double>(getTickCount()) - start) / getTickFrequency();
        cout << "Rotation Time: " << duration * 1000 << "milli seconds." << endl;

        // Create windows to display images              
        // Create windows with normal behavior, which allows resizing
        namedWindow("Original Image", WINDOW_AUTOSIZE);
        namedWindow("Rotated Image", WINDOW_AUTOSIZE);

        // Show images
        imshow("Original Image", src);
        imshow("Rotated Image", dst);
        // Wait for any key press
        waitKey(0);

        src = dst.clone();
    }
    // Release OpenCL resources
    clReleaseMemObject(srcBuffer);
    clReleaseMemObject(dstBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
