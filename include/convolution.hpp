#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <cuda_runtime_api.h>
#include <cudnn_cnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

#include "gpu_session.hpp"

template <typename KernelT, typename InputImageT, typename OutputImageT>
class Convolution {
    static_assert(KernelT::channels() == InputImageT::channels(),
                  "Kernel and input image must have the same number of channels");
    static_assert(KernelT::filters() == OutputImageT::channels(),
                  "Kernel filters must match the number of output image channels");

   public:
    Convolution(GpuSession& gpu_session, std::size_t width, std::size_t height,
                std::initializer_list<float> kernel_values,
                float alpha = 1.0F,  // NOLINT(readability-magic-numbers)
                float beta = 0.0F, int dilation = 1)
        : m_width(width),
          m_height(height),
          m_gpu_session(gpu_session),
          m_kernel{kernel_values},
          m_alpha(alpha),
          m_beta(beta),
          m_dilation(dilation) {
        if (m_kernel.width() % 2 == 0 || m_kernel.height() % 2 == 0) {
            throw std::runtime_error("Kernel width and height must be odd");
        }
        setup();
    }

    ~Convolution() {
        cudaFree(m_d_workspace);
        cudnnDestroyFilterDescriptor(m_kernel_desc);
        cudnnDestroyConvolutionDescriptor(m_conv_desc);
        cudnnDestroyTensorDescriptor(m_output_desc);
        cudnnDestroyTensorDescriptor(m_input_desc);
    }

    Convolution(const Convolution&) = delete;
    Convolution& operator=(const Convolution&) = delete;
    Convolution(Convolution&&) = delete;
    Convolution& operator=(Convolution&&) = delete;

    void setup() {
        // Define kernel descriptor
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_kernel_desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(m_kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               KernelT::filters(), KernelT::channels(),
                                               KernelT::height(), KernelT::width()));

        // Define input tensor descriptor
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1,
                                               InputImageT::channels(), m_height, m_width));

        // Define output tensor descriptor
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_output_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                                               1, OutputImageT::channels(), m_height, m_width));

        // Define convolution descriptor
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            m_conv_desc, m_dilation * (m_kernel.width() / 2), m_dilation * (m_kernel.height() / 2),
            1, 1, m_dilation, m_dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            m_gpu_session.handle(), m_input_desc, m_kernel_desc, m_conv_desc, m_output_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &m_workspace_size));
        CUDA_CHECK(cudaMalloc(&m_d_workspace, m_workspace_size));
    }

    void apply(OutputImageT& output, const InputImageT& input) const {
        assert(input.width() == m_width);
        assert(input.height() == m_height);
        assert(output.width() == m_width);
        assert(output.height() == m_height);

        // Perform the convolution
        CUDNN_CHECK(cudnnConvolutionForward(
            m_gpu_session.handle(), &m_alpha, m_input_desc, input.data(), m_kernel_desc,
            m_kernel.data(), m_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, m_d_workspace,
            m_workspace_size, &m_beta, m_output_desc, output.data()));
    }

   private:
    GpuSession& m_gpu_session;  // NOLINT(*-avoid-const-or-ref-data-members)
    KernelT m_kernel;
    float m_alpha;
    float m_beta;
    int m_dilation;
    cudnnTensorDescriptor_t m_input_desc{};
    cudnnTensorDescriptor_t m_output_desc{};
    cudnnConvolutionDescriptor_t m_conv_desc{};
    cudnnFilterDescriptor_t m_kernel_desc{};
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_workspace_size = 0;
    void* m_d_workspace = nullptr;
};

#endif  // CONVOLUTION_HPP
