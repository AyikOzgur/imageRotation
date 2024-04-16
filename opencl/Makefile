# Define the compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -I/usr/local/include/opencv4 -I/usr/include -I/usr/include/opencv4

# Linker flags. Link against OpenCV and OpenCL
LDFLAGS = -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lOpenCL

# Source files
SRCS = main.cpp

# Target executable
TARGET = rotateImageOpenCL

# Rule to link the program
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Rule for cleaning files
clean:
	rm -f $(TARGET) *.o

.PHONY: clean
