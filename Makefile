# Name of the CUDA compiler
NVCC := nvcc

# Choose your compute architecture (update if needed for your GPU) sm_86:  RTX 30xx  sm_80: A100, sm_80: RTX 40xx, sm_90: H100, sm_100: B100, RTX 50xx
ARCH := -arch=sm_86

# Output executable name
TARGET := mat_mul

# Source files
SRCS := mat_utils.cu mat_mul.cu

# Object files
OBJS := $(SRCS:.cu=.o)

# Compilation flags (add -O2 for optimization, -g for debugging)
NVCCFLAGS := -ccbin gcc-12 $(ARCH) -std=c++11

# Default build rule
all: $(TARGET)

# Link executable from object files
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile each .cu file to .o
%.o: %.cu mat_utils.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule to remove all generated files
clean:
	rm -f $(OBJS) $(TARGET)	