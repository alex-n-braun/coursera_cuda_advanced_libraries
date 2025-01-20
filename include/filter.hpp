#include <cstdint>
#include <vector>

#include "convolution.hpp"
#include "imageManip.hpp"
#include "timer.hpp"
#include "types.hpp"

#define CUDA_CHECK(call)                                                                     \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                    \
    }

#define CUTENSOR_CHECK(call)                                                                    \
    {                                                                                           \
        cutensorStatus_t status = call;                                                         \
        if (status != CUTENSOR_STATUS_SUCCESS) {                                                \
            throw std::runtime_error(std::string("cuTENSOR Error: ") +                          \
                                     cutensorGetErrorString(status) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                 \
        }                                                                                       \
    }

class Filter {
   public:
    Filter(std::size_t width, std::size_t height)
        : oDeviceSrc(width, height),
          oDeviceTmp(width, height),
          edgesImage(width, height),
          oDeviceDstBroadcast(width, height),
          m_conv_to_grayscale({{0.299f, 0.587f, 0.114f, 0.0f}}),
          m_conv_broadcast_to_4_channels({{1.0f, 1.0f, 1.0f, 1.0f}}),
          m_conv_horz({{-0.25, 0, 0.25,  //
                        -0.5, 0, 0.5,    //
                        -0.25, 0, 0.25}}),
          m_conv_edges({{-0.25, 0, 0.25,      //
                         -0.5, 0, 0.5,        //
                         -0.25, 0, 0.25,      //
                         -0.25, -0.5, -0.25,  //
                         0, 0, 0,             //
                         0.25, 0.5, 0.25}}),
          m_conv_reduce_2D_to_1D({{1.0f, 1.0f}}),
          m_conv_smooth({{
              1.0f / 12.0f, 2.0f / 12.0f, 1.0f / 12.0f,  //
              2.0f / 12.0f, 4.0f / 12.0f, 2.0f / 12.0f,  //
              1.0f / 12.0f, 2.0f / 12.0f, 1.0f / 12.0f   //
          }}),
          m_conv_delete({{
                            -0.12f, -0.05f, -0.02f, -0.05f, -0.12f,  //
                            -0.05f, -0.01f, 0.0f,   -0.01,  -0.05f,  //
                            -0.02f, 0.0f,   1.0f,   0.0f,   -0.02f,  //
                            -0.05f, -0.01,  0.0f,   -0.01,  -0.05f,  //
                            -0.12f, -0.05f, -0.02f, -0.05f, -0.12f   //
                        }},
                        1.0f, 0.0f, 4)
    // cudnnCreate(&cudnnHandle);
    // cudnnCreateTensorDescriptor(&inputDesc);
    // cudnnCreateTensorDescriptor(&outputDesc);
    // cudnnCreateFilterDescriptor(&filterDesc);
    // cudnnCreateConvolutionDescriptor(&convDesc);
    {}

    ~Filter() {
        // cudnnDestroy(cudnnHandle);
        // cudnnDestroyTensorDescriptor(inputDesc);
        // cudnnDestroyTensorDescriptor(outputDesc);
        // cudnnDestroyFilterDescriptor(filterDesc);
        // cudnnDestroyConvolutionDescriptor(convDesc);
    }

    void filter(const ImageCPU<std::uint8_t, 4>& input, ImageCPU<std::uint8_t, 4>& output) const {
        ImageGPU<std::uint8_t, 4> d_input(input);
        ImageGPU<float, 4> d_image_float{d_input.width(), d_input.height()};

        if (m_gpu_timer) m_gpu_timer->start();

        convertUint8ToFloat(d_image_float, d_input);

        if (m_gpu_timer_wo_conversion) m_gpu_timer_wo_conversion->start();
        runFilterOnGpu(d_image_float);
        if (m_gpu_timer_wo_conversion) m_gpu_timer_wo_conversion->stop();

        ImageGPU<std::uint8_t, 4> d_output{d_image_float.width(), d_image_float.height()};
        convertFloatToUint8(d_output, d_image_float);

        if (m_gpu_timer) m_gpu_timer->stop();

        d_output.copy_to(output);
    }

    void setGpuTimers(std::shared_ptr<Timer> gpu_timer,
                      std::shared_ptr<Timer> gpu_timer_wo_conversion) {
        m_gpu_timer = gpu_timer;
        m_gpu_timer_wo_conversion = gpu_timer_wo_conversion;
    }

   private:
    void runFilterOnGpu(ImageGPU<float, 4>& d_image) const {
        ImageGPU<float, 1> d_image_gray{d_image.width(), d_image.height()};
        m_conv_to_grayscale.apply(d_image_gray, d_image);

        ImageGPU<float, 2> d_img_temp_2D{d_image.width(), d_image.height()};
        m_conv_edges.apply(d_img_temp_2D, d_image_gray);

        pointwiseAbs(d_img_temp_2D, d_img_temp_2D);

        ImageGPU<float, 1> d_img_temp_1D{d_image.width(), d_image.height()};
        m_conv_reduce_2D_to_1D.apply(d_img_temp_1D, d_img_temp_2D);
        m_conv_smooth.apply(edgesImage, d_img_temp_1D);
        for (std::size_t count = 0; count < 3; count++) {
            pointwiseMin(edgesImage, 0.6f, edgesImage);
            m_conv_smooth.apply(d_img_temp_1D, edgesImage);
            m_conv_smooth.apply(edgesImage, d_img_temp_1D);
        }
        m_conv_delete.apply(d_img_temp_1D, edgesImage);
        pointwiseMin(edgesImage, 1.0f, d_img_temp_1D);

        ImageGPU<float, 4> d_image_broadcast{d_image.width(), d_image.height()};
        m_conv_broadcast_to_4_channels.apply(d_image_broadcast, edgesImage);
        pointwiseHalo(d_image, d_image, d_image_broadcast);

        setChannel(d_image, 3, 1.0);
    }

    mutable ImageGPU<std::uint8_t, 4> oDeviceSrc;
    mutable ImageGPU<float, 1> oDeviceTmp;
    mutable ImageGPU<float, 1> edgesImage;
    mutable ImageGPU<std::uint8_t, 4> oDeviceDstBroadcast;

    Convolution<Kernel<float, 1, 1, 1, 4>, ImageGPU<float, 4>, ImageGPU<float, 1>>
        m_conv_to_grayscale;
    Convolution<Kernel<float, 4, 1, 1, 1>, ImageGPU<float, 1>, ImageGPU<float, 4>>
        m_conv_broadcast_to_4_channels;
    Convolution<Kernel<float, 1, 3, 3, 1>, ImageGPU<float, 1>, ImageGPU<float, 1>> m_conv_horz;
    Convolution<Kernel<float, 2, 3, 3, 1>, ImageGPU<float, 1>, ImageGPU<float, 2>> m_conv_edges;
    Convolution<Kernel<float, 1, 1, 1, 2>, ImageGPU<float, 2>, ImageGPU<float, 1>>
        m_conv_reduce_2D_to_1D;
    Convolution<Kernel<float, 1, 3, 3, 1>, ImageGPU<float, 1>, ImageGPU<float, 1>> m_conv_smooth;
    Convolution<Kernel<float, 1, 5, 5, 1>, ImageGPU<float, 1>, ImageGPU<float, 1>> m_conv_delete;

    std::shared_ptr<Timer> m_gpu_timer;
    std::shared_ptr<Timer> m_gpu_timer_wo_conversion;

    // cudnnHandle_t cudnnHandle;
    // cudnnTensorDescriptor_t inputDesc;
    // cudnnTensorDescriptor_t outputDesc;
    // cudnnFilterDescriptor_t filterDesc;
    // cudnnConvolutionDescriptor_t convDesc;
};
