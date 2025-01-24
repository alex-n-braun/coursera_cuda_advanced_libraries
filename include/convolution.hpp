#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cassert>
#include <iostream>

#define CHECK_CUDNN(status)                                                                      \
    {                                                                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                                    \
            fprintf(stderr, "cuDNN Error: %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, \
                    __LINE__);                                                                   \
            throw std::runtime_error("cuDNN Error");                                             \
        }                                                                                        \
    }

class GpuSession {
   public:
    GpuSession() { CHECK_CUDNN(cudnnCreate(&m_cudnn)); }
    ~GpuSession() { cudnnDestroy(m_cudnn); }

    cudnnHandle_t& handle() { return m_cudnn; }

   private:
    cudnnHandle_t m_cudnn;
};

template <typename Kernel_T, typename InputImage_T, typename OutputImage_T>
class Convolution {
    static_assert(Kernel_T::channels() == InputImage_T::channels(),
                  "Kernel and input image must have the same number of channels");
    static_assert(Kernel_T::filters() == OutputImage_T::channels(),
                  "Kernel filters must match the number of output image channels");

   public:
    Convolution(GpuSession& gpuSession, Kernel_T&& kernel, float alpha = 1.0f, float beta = 0.0f,
                int dilation = 1)
        : m_gpu_session(gpuSession),
          m_kernel(std::move(kernel)),
          m_alpha(alpha),
          m_beta(beta),
          m_dilation(dilation) {
        if (m_kernel.width() % 2 == 0 || m_kernel.height() % 2 == 0) {
            throw std::runtime_error("Kernel width and height must be odd");
        }
        // Define input tensor descriptor
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_inputDesc));
        // Define output tensor descriptor
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_desc));
        // Define convolution descriptor
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc));
        // Define kernel descriptor
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_kernel_desc));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(m_kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               Kernel_T::filters(), Kernel_T::channels(),
                                               Kernel_T::height(), Kernel_T::width()));
    }

    ~Convolution() {
        cudaFree(m_d_workspace);
        cudnnDestroyFilterDescriptor(m_kernel_desc);
        cudnnDestroyConvolutionDescriptor(m_convDesc);
        cudnnDestroyTensorDescriptor(m_output_desc);
        cudnnDestroyTensorDescriptor(m_inputDesc);
    }

    void apply(OutputImage_T& output, const InputImage_T& input) const {
        assert(input.width() == output.width());
        assert(input.height() == output.height());

        if (input.height() != m_height || input.width() != m_width) {
            m_height = input.height();
            m_width = input.width();

            CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                                                   1, InputImage_T::channels(), input.height(),
                                                   input.width()));

            CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_output_desc, CUDNN_TENSOR_NHWC,
                                                   CUDNN_DATA_FLOAT, 1, OutputImage_T::channels(),
                                                   output.height(), output.width()));

            CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
                m_convDesc, m_dilation * (m_kernel.width() / 2),
                m_dilation * (m_kernel.height() / 2), 1, 1, m_dilation, m_dilation,
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

            // Workspace and algorithm selection
            cudaFree(m_d_workspace);

            CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                m_gpu_session.handle(), m_inputDesc, m_kernel_desc, m_convDesc, m_output_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &m_workspace_size));
            cudaError_t cudaStatus = cudaMalloc(&m_d_workspace, m_workspace_size);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(cudaStatus),
                        __FILE__, __LINE__);
                throw std::runtime_error("CUDA Error");
            }
        }

        // Perform the convolution
        CHECK_CUDNN(cudnnConvolutionForward(
            m_gpu_session.handle(), &m_alpha, m_inputDesc, input.data(), m_kernel_desc,
            m_kernel.data(), m_convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, m_d_workspace,
            m_workspace_size, &m_beta, m_output_desc, output.data()));
    }

   private:
    GpuSession& m_gpu_session;
    Kernel_T m_kernel;
    float m_alpha;
    float m_beta;
    int m_dilation;
    cudnnTensorDescriptor_t m_inputDesc;
    cudnnTensorDescriptor_t m_output_desc;
    cudnnConvolutionDescriptor_t m_convDesc;
    cudnnFilterDescriptor_t m_kernel_desc;
    mutable std::int64_t m_width = -1;
    mutable std::int64_t m_height = -1;
    mutable std::size_t m_workspace_size = 0;
    mutable void* m_d_workspace = nullptr;
};

#endif  // CONVOLUTION_HPP
