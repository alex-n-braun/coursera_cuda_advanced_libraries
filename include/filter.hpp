#ifndef FILTER_HPP
#define FILTER_HPP

#include <cstdint>
#include <vector>

#include "convolution.hpp"
#include "gpu_session.hpp"
#include "imageManip.hpp"
#include "timer.hpp"
#include "types.hpp"

class Filter {
   public:
    Filter(GpuSession& gpu_session, std::size_t width, std::size_t height)
        : m_d_input{width, height},
          m_d_image_float{width, height},
          m_d_output{width, height},
          m_d_image_gray{width, height},
          m_d_img_temp_2D{width, height},
          m_d_img_temp_1D{width, height},
          m_d_image_broadcast{width, height},
          m_d_img_edges(width, height),
          m_gpu_session(gpu_session),
          m_conv_to_grayscale(m_gpu_session, width, height, {{0.299f, 0.587f, 0.114f, 0.0f}}),
          m_conv_broadcast_to_4_channels(m_gpu_session, width, height, {{1.0f, 1.0f, 1.0f, 1.0f}}),
          m_conv_horz(m_gpu_session, width, height,
                      {{-0.25, 0, 0.25,  //
                        -0.5, 0, 0.5,    //
                        -0.25, 0, 0.25}}),
          m_conv_edges(m_gpu_session, width, height,
                       {{-0.25, 0, 0.25,      //
                         -0.5, 0, 0.5,        //
                         -0.25, 0, 0.25,      //
                         -0.25, -0.5, -0.25,  //
                         0, 0, 0,             //
                         0.25, 0.5, 0.25}}),
          m_conv_reduce_2D_to_1D(m_gpu_session, width, height, {{1.0f, 1.0f}}),
          m_conv_smooth(m_gpu_session, width, height,
                        {{
                            1.0f / 12.0f, 2.0f / 12.0f, 1.0f / 12.0f,  //
                            2.0f / 12.0f, 4.0f / 12.0f, 2.0f / 12.0f,  //
                            1.0f / 12.0f, 2.0f / 12.0f, 1.0f / 12.0f   //
                        }}),
          m_conv_delete(m_gpu_session, width, height,
                        {{
                            -0.12f, -0.05f, -0.02f, -0.05f, -0.12f,  //
                            -0.05f, -0.01f, 0.0f,   -0.01,  -0.05f,  //
                            -0.02f, 0.0f,   1.0f,   0.0f,   -0.02f,  //
                            -0.05f, -0.01,  0.0f,   -0.01,  -0.05f,  //
                            -0.12f, -0.05f, -0.02f, -0.05f, -0.12f   //
                        }},
                        1.0f, 0.0f, 4) {}

    ~Filter() {}

    void filter(const ImageCPU<std::uint8_t, 4>& input, ImageCPU<std::uint8_t, 4>& output) const {
        m_d_input.copy_from(input);

        if (m_gpu_timer) m_gpu_timer->start();

        runFilterOnGpu();

        if (m_gpu_timer) m_gpu_timer->stop();

        m_d_output.copy_to(output);
    }

    void setGpuTimers(std::shared_ptr<Timer> gpu_timer) { m_gpu_timer = gpu_timer; }

    void prepareGraph(const cudaStream_t& stream) const { runFilterOnGpu(stream); }
    void setInput(const ImageCPU<std::uint8_t, 4>& input) { m_d_input.copy_from(input); }
    void retrieveOutput(ImageCPU<std::uint8_t, 4>& output) const { m_d_output.copy_to(output); }

   private:
    void runFilterOnGpu(const cudaStream_t& stream = 0) const {
        convertUint8ToFloat(m_d_image_float, m_d_input, stream);

        m_conv_to_grayscale.apply(m_d_image_gray, m_d_image_float);
        m_conv_edges.apply(m_d_img_temp_2D, m_d_image_gray);
        pointwiseAbs(m_d_img_temp_2D, m_d_img_temp_2D, stream);
        m_conv_reduce_2D_to_1D.apply(m_d_img_temp_1D, m_d_img_temp_2D);
        m_conv_smooth.apply(m_d_img_edges, m_d_img_temp_1D);
        for (std::size_t count = 0; count < 3; count++) {
            pointwiseMin(m_d_img_edges, 0.6f, m_d_img_edges, stream);
            m_conv_smooth.apply(m_d_img_temp_1D, m_d_img_edges);
            m_conv_smooth.apply(m_d_img_edges, m_d_img_temp_1D);
        }
        m_conv_delete.apply(m_d_img_temp_1D, m_d_img_edges);
        pointwiseMin(m_d_img_edges, 1.0f, m_d_img_temp_1D, stream);

        m_conv_broadcast_to_4_channels.apply(m_d_image_broadcast, m_d_img_edges);
        pointwiseHalo(m_d_image_float, m_d_image_float, m_d_image_broadcast, stream);

        setChannel(m_d_image_float, 3, 1.0, stream);

        convertFloatToUint8(m_d_output, m_d_image_float, stream);
    }

    mutable ImageGPU<std::uint8_t, 4> m_d_input;
    mutable ImageGPU<float, 4> m_d_image_float;
    mutable ImageGPU<std::uint8_t, 4> m_d_output;
    mutable ImageGPU<float, 1> m_d_image_gray;
    mutable ImageGPU<float, 2> m_d_img_temp_2D;
    mutable ImageGPU<float, 1> m_d_img_temp_1D;
    mutable ImageGPU<float, 1> m_d_img_edges;
    mutable ImageGPU<float, 4> m_d_image_broadcast;

    GpuSession& m_gpu_session;
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
};

#endif  // FILTER_HPP
