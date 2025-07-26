# CUDA compiler
NVCC := nvcc

# C++ compiler for non-CUDA files
CXX := g++

# Architecture
ARCH := -arch=sm_86

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

# Source files
CU_SRCS := $(SRC_DIR)/mat_utils.cu $(SRC_DIR)/mat_mul.cu
CPP_SRCS := $(SRC_DIR)/matrix.cpp

# Object files
CU_OBJS := $(BUILD_DIR)/mat_utils.o $(BUILD_DIR)/mat_mul.o
CPP_OBJS := $(BUILD_DIR)/matrix.o
OBJS := $(CU_OBJS) $(CPP_OBJS)

# Output executable
TARGET := $(BUILD_DIR)/mat_mul

# Compilation flags
NVCCFLAGS := -ccbin gcc-12 $(ARCH) -std=c++17 -Xcompiler="-fPIC" -lcudart -I$(INC_DIR)
CXXFLAGS := -std=c++17 -Wall -Wextra -fPIC -I$(INC_DIR)
CXXFLAGS += -I/usr/local/cuda/include

# Linker flags
LDFLAGS := -lcudart -lstdc++

# Build rules
all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $(OBJS)

# CUDA object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/*.h $(INC_DIR)/*.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# C++ object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/*.h $(INC_DIR)/*.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean