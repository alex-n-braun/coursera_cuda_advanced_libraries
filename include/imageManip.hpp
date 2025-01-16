#ifndef IMAGE_MANIP_HPP
#define IMAGE_MANIP_HPP

#include <cstdint>

#include "types.hpp"

void convertUint8ToFloat(ImageGPU<float, 4>& output, const ImageGPU<std::uint8_t, 4>& input);
void convertFloatToUint8(ImageGPU<std::uint8_t, 4>& output, const ImageGPU<float, 4>& input);
void setChannel(ImageGPU<float, 4>& data, int channel, float value);
template <std::size_t Channels>
void pointwiseAbs(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& input);
template <std::size_t Channels>
void pointwiseMin(ImageGPU<float, Channels>& output, float minValue,
                  const ImageGPU<float, Channels>& input);
template <std::size_t Channels>
void pointwiseHalo(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& rgbInput,
                   const ImageGPU<float, Channels>& haloInput);

#endif  // IMAGE_MANIP_HPP
