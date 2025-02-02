#ifndef CUDA_GRAPH_HPP
#define CUDA_GRAPH_HPP

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <functional>
#include <stdexcept>

#include "gpu_session.hpp"

class CudaGraph {
   public:
    CudaGraph(GpuSession& gpu_session, const std::function<void(const cudaStream_t&)>& func)
        : m_gpu_session(gpu_session) {
        setup(func);
    }

    ~CudaGraph() {
        if (m_instance != nullptr) {
            cudaGraphExecDestroy(m_instance);
        }
        if (m_graph != nullptr) {
            cudaGraphDestroy(m_graph);
        }
    }

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&&) = delete;
    CudaGraph& operator=(CudaGraph&&) = delete;

    void run() {
        if (m_instance != nullptr) {
            cudaGraphLaunch(m_instance, nullptr);
            cudaStreamSynchronize(nullptr);
        } else {
            throw std::runtime_error("Cuda graph instance is not initialized.");
        }
    }

   private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_instance{};
    GpuSession& m_gpu_session;

    void setup(const std::function<void(const cudaStream_t&)>& func) {
        cudaStream_t stream = m_gpu_session.sessionStream();

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        func(stream);
        cudaStreamEndCapture(stream, &m_graph);

        cudaGraphInstantiate(&m_instance, m_graph, nullptr, nullptr, 0);
    }
};

#endif  // CUDA_GRAPH_HPP
