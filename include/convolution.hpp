#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>

#define CHECK_CUDNN(status)                                                                      \
    {                                                                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                                    \
            fprintf(stderr, "cuDNN Error: %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, \
                    __LINE__);                                                                   \
            throw std::runtime_error("cuDNN Error");                                             \
        }                                                                                        \
    }

template <typename Kernel_T, typename InputImage_T, typename OutputImage_T>
class Convolution {
    static_assert(Kernel_T::channels() == InputImage_T::channels(),
                  "Kernel and input image must have the same number of channels");
    static_assert(Kernel_T::filters() == OutputImage_T::channels(),
                  "Kernel filters must match the number of output image channels");

   public:
    Convolution(Kernel_T&& kernel, float alpha = 1.0f, float beta = 0.0f, int dilation = 1)
        : m_kernel(std::move(kernel)), m_alpha(alpha), m_beta(beta), m_dilation(dilation) {
        if (m_kernel.width() % 2 == 0 || m_kernel.height() % 2 == 0) {
            throw std::runtime_error("Kernel width and height must be odd");
        }
    }

    void apply(OutputImage_T& output, const InputImage_T& input) const {
        assert(input.width() == output.width());
        assert(input.height() == output.height());

        cudnnHandle_t cudnn;
        CHECK_CUDNN(cudnnCreate(&cudnn));

        // Define input tensor descriptor
        cudnnTensorDescriptor_t inputDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1,
                                               InputImage_T::channels(), input.height(),
                                               input.width()));

        // Define output tensor descriptor
        cudnnTensorDescriptor_t outputDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1,
                                               OutputImage_T::channels(), output.height(),
                                               output.width()));

        // Define convolution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_dilation * (m_kernel.width() / 2),
                                                    m_dilation * (m_kernel.height() / 2), 1, 1, m_dilation, m_dilation,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // Define kernel descriptor
        cudnnFilterDescriptor_t kernelDesc;
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               Kernel_T::filters(), Kernel_T::channels(),
                                               Kernel_T::height(), Kernel_T::width()));

        // Workspace and algorithm selection
        size_t workspaceSize = 0;
        void* d_workspace = nullptr;

        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, inputDesc, kernelDesc, convDesc, outputDesc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspaceSize));
        cudaError_t cudaStatus = cudaMalloc(&d_workspace, workspaceSize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(cudaStatus), __FILE__,
                    __LINE__);
            throw std::runtime_error("CUDA Error");
        }

        // Perform the convolution
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &m_alpha, inputDesc, input.data(), kernelDesc,
                                            m_kernel.data(), convDesc,
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, d_workspace,
                                            workspaceSize, &m_beta, outputDesc, output.data()));

        // Cleanup
        cudaFree(d_workspace);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyFilterDescriptor(kernelDesc);
        cudnnDestroy(cudnn);
    }

   private:
    Kernel_T m_kernel;
    float m_alpha;
    float m_beta;
    int m_dilation;
};

#endif  // CONVOLUTION_HPP
