#ifndef IMAGE_MANIP_HPP
#define IMAGE_MANIP_HPP

#include <driver_types.h>

#include <cstddef>
#include <cstdint>

#include "types.hpp"

void convertUint8ToFloat(ImageGPU<float, 4>& output, const ImageGPU<std::uint8_t, 4>& input,
                         const cudaStream_t& stream = nullptr);
void convertFloatToUint8(ImageGPU<std::uint8_t, 4>& output, const ImageGPU<float, 4>& input,
                         const cudaStream_t& stream = nullptr);
void setChannel(ImageGPU<float, 4>& data, int channel, float value,
                const cudaStream_t& stream = nullptr);
template <std::size_t kChannelsT>
void pointwiseAbs(ImageGPU<float, kChannelsT>& output, const ImageGPU<float, kChannelsT>& input,
                  const cudaStream_t& stream = nullptr);
template <std::size_t kChannelsT>
void pointwiseMin(ImageGPU<float, kChannelsT>& output, float limit_value,
                  const ImageGPU<float, kChannelsT>& input, const cudaStream_t& stream = nullptr);
template <std::size_t kChannelsT>
void pointwiseHalo(ImageGPU<float, kChannelsT>& output,
                   const ImageGPU<float, kChannelsT>& rgb_input,
                   const ImageGPU<float, kChannelsT>& halo_input,
                   const cudaStream_t& stream = nullptr);

#endif  // IMAGE_MANIP_HPP
