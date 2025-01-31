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

# Define source files and object files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%.o, $(SRC_FILES))

# Define source files and target executable
SRC_EDGE = $(SRC_DIR)/edgeDetection.cpp
TARGET_EDGE = $(BIN_DIR)/edgeDetection

# Define the default rule
all: $(TARGET_EDGE)

# Pattern rule to compile CUDA source files to object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(TARGET_EDGE): $(SRC_EDGE) $(OBJ_FILES)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_EDGE) $(OBJ_FILES) -o $(TARGET_EDGE) $(LDFLAGS)

# Rules for running the applications
run: $(TARGET_EDGE)
	./$(TARGET_EDGE) --input $(DATA_DIR)/Lena.png --output $(DATA_DIR)/Lena_edge.png

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
