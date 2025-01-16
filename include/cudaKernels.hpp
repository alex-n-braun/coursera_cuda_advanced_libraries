#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cuda_runtime.h>

#include <cstdint>

__global__ void kernelConvertUint8ToFloat(const std::uint8_t* input, float* output, int size);
__global__ void kernelConvertFloatToUint8(const float* input, std::uint8_t* output, int size);
__global__ void kernelSetChannel(float* data, int channel, float value, int numChannels, int size);
__global__ void kernelPointwiseAbs(const float* input, float* output, int size);
__global__ void kernelPointwiseMin(const float* input, float minValue, float* output, int size);
__global__ void kernelPointwiseHalo(const float* rgbInput, const float* haloInput, float* output,
                                    int size);

#endif  // KERNELS_HPP
