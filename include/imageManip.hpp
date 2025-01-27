#ifndef IMAGE_MANIP_HPP
#define IMAGE_MANIP_HPP

#include <cstdint>

#include "types.hpp"

void convertUint8ToFloat(ImageGPU<float, 4>& output, const ImageGPU<std::uint8_t, 4>& input,
                         const cudaStream_t& stream = 0);
void convertFloatToUint8(ImageGPU<std::uint8_t, 4>& output, const ImageGPU<float, 4>& input,
                         const cudaStream_t& stream = 0);
void setChannel(ImageGPU<float, 4>& data, int channel, float value, const cudaStream_t& stream = 0);
template <std::size_t Channels>
void pointwiseAbs(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& input,
                  const cudaStream_t& stream = 0);
template <std::size_t Channels>
void pointwiseMin(ImageGPU<float, Channels>& output, float minValue,
                  const ImageGPU<float, Channels>& input, const cudaStream_t& stream = 0);
template <std::size_t Channels>
void pointwiseHalo(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& rgbInput,
                   const ImageGPU<float, Channels>& haloInput, const cudaStream_t& stream = 0);

#endif  // IMAGE_MANIP_HPP
