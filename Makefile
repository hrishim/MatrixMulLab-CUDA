# CUDA compiler
NVCC := nvcc

# C++ compiler for non-CUDA files
CXX := g++

# Architecture
ARCH := -arch=sm_86

# Output executable
TARGET := mat_mul

# Source files
CU_SRCS := mat_utils.cu mat_mul.cu
CPP_SRCS := matrix.cpp

# Header files
HEADERS := matrix.hpp mat_utils.h

# Object files
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
OBJS := $(CU_OBJS) $(CPP_OBJS)

# Compilation flags
NVCCFLAGS := -ccbin gcc-12 $(ARCH) -std=c++17 -Xcompiler="-fPIC" -lcudart
CXXFLAGS := -std=c++17 -Wall -Wextra -fPIC

# Linker flags
LDFLAGS := -lcudart -lstdc++

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

# CUDA object files with specific dependencies
mat_utils.o: mat_utils.cu matrix.hpp mat_utils.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

mat_mul.o: mat_mul.cu mat_utils.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# C++ object file with specific dependencies
matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
