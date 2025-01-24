#include "cudaKernels.hpp"
#include "imageManip.hpp"

void convertUint8ToFloat(ImageGPU<float, 4>& output, const ImageGPU<std::uint8_t, 4>& input,
                         const cudaStream_t& stream) {
    kernelConvertUint8ToFloat<<<(input.size() + 255) / 256, 256, 0, stream>>>(
        input.data(), output.data(), input.size());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelConvertUint8ToFloat): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void convertFloatToUint8(ImageGPU<std::uint8_t, 4>& output, const ImageGPU<float, 4>& input,
                         const cudaStream_t& stream) {
    kernelConvertFloatToUint8<<<(input.size() + 255) / 256, 256, 0, stream>>>(
        input.data(), output.data(), input.size());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelConvertFloatToUint8): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void setChannel(ImageGPU<float, 4>& data, int channel, float value, const cudaStream_t& stream) {
    kernelSetChannel<<<(data.size() + 255) / 256, 256, 0, stream>>>(data.data(), channel, value, 4,
                                                                    data.numPixels());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelSetChannel): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <std::size_t Channels>
void pointwiseAbs_(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& input,
                   const cudaStream_t& stream) {
    kernelPointwiseAbs<<<(input.size() + 255) / 256, 256, 0, stream>>>(input.data(), output.data(),
                                                                       input.size());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseAbs): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <>
void pointwiseAbs<2>(ImageGPU<float, 2>& output, const ImageGPU<float, 2>& input,
                     const cudaStream_t& stream) {
    pointwiseAbs_<2>(output, input, stream);
}

template <std::size_t Channels>
void pointwiseMin_(ImageGPU<float, Channels>& output, float minValue,
                   const ImageGPU<float, Channels>& input, const cudaStream_t& stream) {
    kernelPointwiseMin<<<(input.size() + 255) / 256, 256, 0, stream>>>(input.data(), minValue,
                                                                       output.data(), input.size());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseMin): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <>
void pointwiseMin<1>(ImageGPU<float, 1>& output, float minValue, const ImageGPU<float, 1>& input,
                     const cudaStream_t& stream) {
    pointwiseMin_<1>(output, minValue, input, stream);
}

template <std::size_t Channels>
void pointwiseHalo_(ImageGPU<float, Channels>& output, const ImageGPU<float, Channels>& rgbInput,
                    const ImageGPU<float, Channels>& haloInput, const cudaStream_t& stream) {
    if (rgbInput.size() != haloInput.size() || rgbInput.size() != output.size()) {
        throw std::runtime_error("Image sizes do not match");
    }

    kernelPointwiseHalo<<<(rgbInput.size() + 255) / 256, 256, 0, stream>>>(
        rgbInput.data(), haloInput.data(), output.data(), rgbInput.size());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error (kernelPointwiseHalo): " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <>
void pointwiseHalo<4>(ImageGPU<float, 4>& output, const ImageGPU<float, 4>& rgbInput,
                      const ImageGPU<float, 4>& haloInput, const cudaStream_t& stream) {
    pointwiseHalo_<4>(output, rgbInput, haloInput, stream);
}
