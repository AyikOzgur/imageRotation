# Define the compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -std=c++11 -Wall -I/usr/include/opencv4

# Define the linker flags
# Here, you need to link against the OpenCV libraries.
LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Define the source files
SRCS = main.cpp

# Define the target executable
TARGET = rotateImage

# Rule to link the program
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Rule for cleaning files
clean:
	rm -f $(TARGET) *.o
