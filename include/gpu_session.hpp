#ifndef GPU_SESSION_HPP
#define GPU_SESSION_HPP

#include <cuda_runtime_api.h>
#include <cudnn_graph.h>
#include <driver_types.h>

#include <stdexcept>
#include <string>

constexpr void cudnnCheck(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuDNN Error: ") + cudnnGetErrorString(status) +
                                 " at " + file + ":" + std::to_string(line));
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CUDNN_CHECK(status) \
    { cudnnCheck(status, __FILE__, __LINE__); }

constexpr void cudaCheck(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(status) + " at " +
                                 file + ":" + std::to_string(line));
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CUDA_CHECK(status) \
    { cudaCheck(status, __FILE__, __LINE__); }

class GpuSession {
   public:
    GpuSession() {  // NOLINT(*-member-init)
        CUDNN_CHECK(cudnnCreate(&m_cudnn));
        CUDA_CHECK(cudaStreamCreate(&m_stream));
        CUDNN_CHECK(cudnnSetStream(m_cudnn, m_stream));
    }
    ~GpuSession() {
        cudnnDestroy(m_cudnn);
        cudaStreamDestroy(m_stream);
    }

    cudnnHandle_t& handle() { return m_cudnn; }
    cudaStream_t& sessionStream() { return m_stream; }

    GpuSession(const GpuSession&) = delete;
    GpuSession& operator=(const GpuSession&) = delete;
    GpuSession(GpuSession&&) = delete;
    GpuSession& operator=(GpuSession&&) = delete;

   private:
    cudnnHandle_t m_cudnn;
    cudaStream_t m_stream;
};

#endif