#ifndef FILTER_HPP
#define FILTER_HPP

#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "convolution.hpp"
#include "gpu_session.hpp"
#include "image_manip.hpp"
#include "timer.hpp"
#include "types.hpp"

class Filter {
   public:
    // NOLINTBEGIN(*avoid-magic-numbers, readability-magic-numbers)
    Filter(GpuSession& gpu_session, std::size_t width, std::size_t height)
        : m_d_input{width, height},
          m_d_image_float{width, height},
          m_d_output{width, height},
          m_d_image_gray{width, height},
          m_d_img_temp_2d{width, height},
          m_d_img_temp_1d{width, height},
          m_d_image_broadcast{width, height},
          m_d_img_edges(width, height),
          m_gpu_session(gpu_session),
          m_conv_to_grayscale(m_gpu_session, width, height, {0.299F, 0.587F, 0.114F, 0.0F}),
          m_conv_broadcast_to_4_channels(m_gpu_session, width, height, {1.0F, 1.0F, 1.0F, 1.0F}),
          m_conv_edges(m_gpu_session, width, height,
                       {-0.25F, 0.0F, 0.25F,    //
                        -0.5F, 0.0F, 0.5F,      //
                        -0.25F, 0.0F, 0.25F,    //
                        -0.25F, -0.5F, -0.25F,  //
                        0.0F, 0.0F, 0.0F,       //
                        0.25F, 0.5F, 0.25F}),
          m_conv_reduce_2d_to_1d(m_gpu_session, width, height, {1.0F, 1.0F}),
          m_conv_smooth(m_gpu_session, width, height,
                        {
                            1.0F / 12.0F, 2.0F / 12.0F, 1.0F / 12.0F,  //
                            2.0F / 12.0F, 4.0F / 12.0F, 2.0F / 12.0F,  //
                            1.0F / 12.0F, 2.0F / 12.0F, 1.0F / 12.0F   //
                        }),
          m_conv_delete(m_gpu_session, width, height,
                        {
                            -0.12F, -0.05F, -0.02F, -0.05F, -0.12F,  //
                            -0.05F, -0.01F, 0.0F,   -0.01F, -0.05F,  //
                            -0.02F, 0.0F,   1.0F,   0.0F,   -0.02F,  //
                            -0.05F, -0.01F, 0.0F,   -0.01F, -0.05F,  //
                            -0.12F, -0.05F, -0.02F, -0.05F, -0.12F   //
                        },
                        1.0F, 0.0F, 4) {}
    // NOLINTEND(*avoid-magic-numbers,  readability-magic-numbers)

    void filter(const ImageCPU<std::uint8_t, 4>& input, ImageCPU<std::uint8_t, 4>& output) const {
        m_d_input.copyFrom(input);

        if (m_gpu_timer) {
            m_gpu_timer->start();
        }

        runFilterOnGpu();

        if (m_gpu_timer) {
            m_gpu_timer->stop();
        }

        m_d_output.copyTo(output);
    }

    void setGpuTimers(std::shared_ptr<Timer> gpu_timer) { m_gpu_timer = std::move(gpu_timer); }

    void prepareGraph(const cudaStream_t& stream) const { runFilterOnGpu(stream); }
    void setInput(const ImageCPU<std::uint8_t, 4>& input) { m_d_input.copyFrom(input); }
    void retrieveOutput(ImageCPU<std::uint8_t, 4>& output) const { m_d_output.copyTo(output); }

   private:
    void runFilterOnGpu(const cudaStream_t& stream = nullptr) const {
        convertUint8ToFloat(m_d_image_float, m_d_input, stream);

        m_conv_to_grayscale.apply(m_d_image_gray, m_d_image_float);
        m_conv_edges.apply(m_d_img_temp_2d, m_d_image_gray);
        pointwiseAbs(m_d_img_temp_2d, m_d_img_temp_2d, stream);
        m_conv_reduce_2d_to_1d.apply(m_d_img_temp_1d, m_d_img_temp_2d);
        m_conv_smooth.apply(m_d_img_edges, m_d_img_temp_1d);

        constexpr std::size_t kNumberOfSmoothingIterations = 3;
        constexpr float kMagicIntensityLimit = 0.6F;
        constexpr float kMaxIntensity = 1.0F;
        for (std::size_t count = 0; count < kNumberOfSmoothingIterations; count++) {
            pointwiseMin(m_d_img_edges, kMagicIntensityLimit, m_d_img_edges, stream);
            m_conv_smooth.apply(m_d_img_temp_1d, m_d_img_edges);
            m_conv_smooth.apply(m_d_img_edges, m_d_img_temp_1d);
        }
        m_conv_delete.apply(m_d_img_temp_1d, m_d_img_edges);
        pointwiseMin(m_d_img_edges, kMaxIntensity, m_d_img_temp_1d, stream);

        m_conv_broadcast_to_4_channels.apply(m_d_image_broadcast, m_d_img_edges);
        pointwiseHalo(m_d_image_float, m_d_image_float, m_d_image_broadcast, stream);

        constexpr int kAlphaChannel = 3;
        setChannel(m_d_image_float, kAlphaChannel, kMaxIntensity, stream);

        convertFloatToUint8(m_d_output, m_d_image_float, stream);
    }

    // NOLINTBEGIN(readability-magic-numbers, *-avoid-magic-numbers)
    mutable ImageGPU<std::uint8_t, 4> m_d_input;
    mutable ImageGPU<float, 4> m_d_image_float;
    mutable ImageGPU<std::uint8_t, 4> m_d_output;
    mutable ImageGPU<float, 1> m_d_image_gray;
    mutable ImageGPU<float, 2> m_d_img_temp_2d;
    mutable ImageGPU<float, 1> m_d_img_temp_1d;
    mutable ImageGPU<float, 1> m_d_img_edges;
    mutable ImageGPU<float, 4> m_d_image_broadcast;

    GpuSession& m_gpu_session;
    Convolution<Kernel<float, 1, 1, 1, 4>, ImageGPU<float, 4>, ImageGPU<float, 1>>
        m_conv_to_grayscale;
    Convolution<Kernel<float, 4, 1, 1, 1>, ImageGPU<float, 1>, ImageGPU<float, 4>>
        m_conv_broadcast_to_4_channels;
    Convolution<Kernel<float, 2, 3, 3, 1>, ImageGPU<float, 1>, ImageGPU<float, 2>> m_conv_edges;
    Convolution<Kernel<float, 1, 1, 1, 2>, ImageGPU<float, 2>, ImageGPU<float, 1>>
        m_conv_reduce_2d_to_1d;
    Convolution<Kernel<float, 1, 3, 3, 1>, ImageGPU<float, 1>, ImageGPU<float, 1>> m_conv_smooth;
    Convolution<Kernel<float, 1, 5, 5, 1>, ImageGPU<float, 1>, ImageGPU<float, 1>> m_conv_delete;
    // NOLINTEND(readability-magic-numbers, *-avoid-magic-numbers)

    std::shared_ptr<Timer> m_gpu_timer;
};

#endif  // FILTER_HPP
