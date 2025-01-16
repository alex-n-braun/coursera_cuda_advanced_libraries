#include <cuda_runtime.h>

#include <cstdint>

#include "cudaKernels.hpp"

__global__ void kernelConvertUint8ToFloat(const std::uint8_t* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / 255.0f;  // Normalize to [0, 1]
    }
}

__global__ void kernelConvertFloatToUint8(const float* input, std::uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<std::uint8_t>(fminf(fmaxf(input[idx] * 255.0f, 0.0f), 255.0f));
    }
}

__global__ void kernelSetChannel(float* data, int channel, float value, int numChannels, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[numChannels * idx + channel] = value;
    }
}

__global__ void kernelPointwiseAbs(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fabsf(input[idx]);
    }
}

__global__ void kernelPointwiseMin(const float* input, float minValue, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fminf(input[idx], minValue);
    }
}

__global__ void kernelPointwiseHalo(const float* rgbInput, const float* haloInput, float* output,
                                    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float distToMax = 1.0f - rgbInput[idx];
        constexpr float cutoff = 0.1f;
        constexpr float factor = 4.0f;
        const float halo = fminf(1.0f, (fmaxf(cutoff, haloInput[idx]) - cutoff) * factor);
        output[idx] = 1.0f - distToMax * (1.0f - halo);
    }
}