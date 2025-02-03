#include <cuda_runtime.h>  // NOLINT(misc-include-cleaner)

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "cuda_kernels.hpp"

__global__ void kernelConvertUint8ToFloat(const std::uint8_t* input, float* output,
                                          std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        constexpr float kNormalizationFactor = 1.0F / 255.0F;
        output[idx] = static_cast<float>(input[idx]) * kNormalizationFactor;
    }
}

__global__ void kernelConvertFloatToUint8(const float* input, std::uint8_t* output,
                                          std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        constexpr float kMax = 255.0F;
        constexpr float kMin = 0.0F;
        output[idx] = static_cast<std::uint8_t>(fminf(fmaxf(input[idx] * kMax, kMin), kMax));
    }
}

__global__ void kernelSetChannel(float* data, int channel, float value, int num_channels,
                                 std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[num_channels * idx + channel] = value;
    }
}

__global__ void kernelPointwiseAbs(const float* input, float* output, std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fabsf(input[idx]);
    }
}

__global__ void kernelPointwiseMin(const float* input, float limit_value, float* output,
                                   std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fminf(input[idx], limit_value);
    }
}

__global__ void kernelPointwiseHalo(const float* rgb_input, const float* halo_input, float* output,
                                    std::size_t size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        constexpr float kMaxValue = 1.0F;
        const float dist_to_max = kMaxValue - rgb_input[idx];
        constexpr float kCutoff = 0.1F;
        constexpr float kFactor = 4.0F;
        const float halo = fminf(kMaxValue, (fmaxf(kCutoff, halo_input[idx]) - kCutoff) * kFactor);
        output[idx] = kMaxValue - dist_to_max * (kMaxValue - halo);
    }
}
