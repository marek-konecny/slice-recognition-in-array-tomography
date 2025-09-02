CXX := g++
OPENCV_INC := /usr/include/opencv4
OPENCV_LIB := /usr/lib64

CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -Wall -I$(OPENCV_INC)
LDFLAGS  := -fopenmp -L$(OPENCV_LIB) -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

TARGET := slice_localize
SRC    := main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
