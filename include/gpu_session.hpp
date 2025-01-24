#ifndef GPU_SESSION_HPP
#define GPU_SESSION_HPP

#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>

#define CUDNN_CHECK(status)                                                                      \
    {                                                                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                                    \
            fprintf(stderr, "cuDNN Error: %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, \
                    __LINE__);                                                                   \
            throw std::runtime_error("cuDNN Error");                                             \
        }                                                                                        \
    }

#define CUDA_CHECK(call)                                                                     \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                    \
    }

class GpuSession {
   public:
    GpuSession() {
        CUDNN_CHECK(cudnnCreate(&m_cudnn));
        // add error handling
        cudaStreamCreate(&m_stream);

        CUDNN_CHECK(cudnnSetStream(m_cudnn, m_stream));
    }
    ~GpuSession() {
        cudnnDestroy(m_cudnn);
        cudaStreamDestroy(m_stream);
        // add error handling
    }

    cudnnHandle_t& handle() { return m_cudnn; }
    cudaStream_t& sessionStream() { return m_stream; }

   private:
    cudnnHandle_t m_cudnn;
    cudaStream_t m_stream;
};

#endif  // GPU_SESSION_HPP
