# Check if the platform is Windows or Linux
ifeq ($(OS),Windows_NT)
    PLATFORM := windows
    EXT := dll
    OBJECT_EXT := obj
    RM := del /Q
    MKDIR := mkdir
else
    PLATFORM := linux
    EXT := so
    OBJECT_EXT := o
    RM := rm -f
    MKDIR := mkdir -p
endif

# Path to CUDA toolkit (adjust if necessary)
CUDA_PATH ?= /usr/local/cuda
NVCC       := $(CUDA_PATH)/bin/nvcc

# Directories
SRC_DIR    := src
INCLUDE_DIR:= include

# Compiler flags
# -O3 for optimization, -std=c++11 for C++11 standard, -I to include headers, -fPIC for position independent code
NVCCFLAGS  := -O3 -std=c++11 -I$(INCLUDE_DIR) -Xcompiler -fPIC -arch=sm_75

# Sources: all .cu and .cpp files in the src/ directory
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Create object files from source file names (.cu/.cpp -> .obj or .o)
CU_OBJS := $(CU_SRCS:$(SRC_DIR)/%.cu=$(SRC_DIR)/%.$(OBJECT_EXT))
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.$(OBJECT_EXT))
OBJS := $(CU_OBJS) $(CPP_OBJS)

# Target shared library name
TARGET := libcudaSLIC.$(EXT)

# Default target
all: $(TARGET)

# Link object files into a shared library
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $^

# Compile CUDA source files (.cu)
$(SRC_DIR)/%.$(OBJECT_EXT): $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Compile C++ source files (.cpp)
$(SRC_DIR)/%.$(OBJECT_EXT): $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up build files
clean:
	$(RM) $(SRC_DIR)/*.$(OBJECT_EXT) $(TARGET)

.PHONY: all clean
