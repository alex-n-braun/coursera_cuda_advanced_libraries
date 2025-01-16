################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

# Define the compiler and flags
CUDA_PATH ?= /usr/local/cuda

NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -g -I$(CUDA_PATH)/include -Iinclude
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcudnn -lfreeimage `pkg-config --cflags --libs opencv4`


# Define source files for CUDA kernels
SRC_KERNELS = $(SRC_DIR)/cudaKernels.cu
OBJ_KERNELS = $(BIN_DIR)/cudaKernels.o

# Rule to compile CUDA kernels to object file
$(OBJ_KERNELS): $(SRC_KERNELS)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -c $(SRC_KERNELS) -o $(OBJ_KERNELS)

# Define source files for imageManip
SRC_IMAGEMANIP = $(SRC_DIR)/imageManip.cu
OBJ_IMAGEMANIP = $(BIN_DIR)/imageManip.o

# Rule to compile imageManip to object file
$(OBJ_IMAGEMANIP): $(SRC_IMAGEMANIP)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -c $(SRC_IMAGEMANIP) -o $(OBJ_IMAGEMANIP)

# Define source files for GPU Blob
SRC_BLOB = $(SRC_DIR)/gpuBlob.cu
OBJ_BLOB = $(BIN_DIR)/gpuBlob.o

# Rule to compile GPU Blob to object file
$(OBJ_BLOB): $(SRC_BLOB)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -c $(SRC_BLOB) -o $(OBJ_BLOB)

# Define source files and target executable
SRC_EDGE = $(SRC_DIR)/edgeDetection.cpp
TARGET_EDGE = $(BIN_DIR)/edgeDetection

# Define the default rule
all: $(TARGET_EDGE)

$(TARGET_EDGE): $(SRC_EDGE) $(OBJ_KERNELS) $(OBJ_BLOB) $(OBJ_IMAGEMANIP)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC_EDGE) $(OBJ_KERNELS) $(OBJ_BLOB) $(OBJ_IMAGEMANIP) -o $(TARGET_EDGE) $(LDFLAGS)

# Rules for running the applications
run: $(TARGET_EDGE)
	./$(TARGET_EDGE) --input $(DATA_DIR)/Lena.png --output $(DATA_DIR)/Lena_filtered.png

# Clean up
clean:
	rm -rf $(BIN_DIR)

# Help command
help:
	@echo "Available make commands:"
	@echo "  make          - Build the project."
	@echo "  make run      - Run the project."
	@echo "  make clean    - Clean up the build files."
	@echo "  make install  - Install the project (if applicable)."
	@echo "  make help     - Display this help message."
