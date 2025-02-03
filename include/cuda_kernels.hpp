#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <device_types.h>

#include <cstddef>
#include <cstdint>

__global__ void kernelConvertUint8ToFloat(const std::uint8_t* input, float* output,
                                          std::size_t size);
__global__ void kernelConvertFloatToUint8(const float* input, std::uint8_t* output,
                                          std::size_t size);
__global__ void kernelSetChannel(float* data, int channel, float value, int num_channels,
                                 std::size_t size);
__global__ void kernelPointwiseAbs(const float* input, float* output, std::size_t size);
__global__ void kernelPointwiseMin(const float* input, float limit_value, float* output,
                                   std::size_t size);
__global__ void kernelPointwiseHalo(const float* rgb_input, const float* halo_input, float* output,
                                    std::size_t size);

#endif  // KERNELS_HPP
