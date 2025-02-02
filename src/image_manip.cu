#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_kernels.hpp"
#include "image_manip.hpp"
#include "types.hpp"

constexpr unsigned int kMaxThreadsPerBlock = 256;

void convertUint8ToFloat(ImageGPU<float, 4>& output, const ImageGPU<std::uint8_t, 4>& input,
                         const cudaStream_t& stream) {
    const unsigned int grid_dim = (input.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelConvertUint8ToFloat<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(
        input.data(), output.data(), input.size());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelConvertUint8ToFloat): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void convertFloatToUint8(ImageGPU<std::uint8_t, 4>& output, const ImageGPU<float, 4>& input,
                         const cudaStream_t& stream) {
    const unsigned int grid_dim = (input.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelConvertFloatToUint8<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(
        input.data(), output.data(), input.size());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelConvertFloatToUint8): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void setChannel(ImageGPU<float, 4>& data, int channel, float value, const cudaStream_t& stream) {
    const unsigned int grid_dim = (data.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelSetChannel<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(data.data(), channel, value, 4,
                                                                   data.numPixels());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelSetChannel): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

namespace {

template <std::size_t kChannelsT>
void pointwiseAbsImpl(ImageGPU<float, kChannelsT>& output, const ImageGPU<float, kChannelsT>& input,
                      const cudaStream_t& stream) {
    const unsigned int grid_dim = (input.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelPointwiseAbs<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(input.data(), output.data(),
                                                                     input.size());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseAbs): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

}  // namespace

template <>
void pointwiseAbs<2>(ImageGPU<float, 2>& output, const ImageGPU<float, 2>& input,
                     const cudaStream_t& stream) {
    pointwiseAbsImpl<2>(output, input, stream);
}

template <std::size_t kChannelsT>
void pointwiseMinImpl(ImageGPU<float, kChannelsT>& output, float limit_value,
                      const ImageGPU<float, kChannelsT>& input, const cudaStream_t& stream) {
    const unsigned int grid_dim = (input.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelPointwiseMin<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(input.data(), limit_value,
                                                                     output.data(), input.size());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseMin): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <>
void pointwiseMin<1>(ImageGPU<float, 1>& output, float limit_value, const ImageGPU<float, 1>& input,
                     const cudaStream_t& stream) {
    pointwiseMinImpl<1>(output, limit_value, input, stream);
}

namespace {

template <std::size_t kChannelsT>
void pointwiseHaloImpl(ImageGPU<float, kChannelsT>& output,
                       const ImageGPU<float, kChannelsT>& rgb_input,
                       const ImageGPU<float, kChannelsT>& halo_input, const cudaStream_t& stream) {
    if (rgb_input.size() != halo_input.size() || rgb_input.size() != output.size()) {
        throw std::runtime_error("Image sizes do not match");
    }

    const unsigned int grid_dim =
        (rgb_input.size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    kernelPointwiseHalo<<<grid_dim, kMaxThreadsPerBlock, 0, stream>>>(
        rgb_input.data(), halo_input.data(), output.data(), rgb_input.size());

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseHalo): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

}  // namespace

template <>
void pointwiseHalo<4>(ImageGPU<float, 4>& output, const ImageGPU<float, 4>& rgb_input,
                      const ImageGPU<float, 4>& halo_input, const cudaStream_t& stream) {
    pointwiseHaloImpl<4>(output, rgb_input, halo_input, stream);
}
