#ifndef CUDA_GRAPH_HPP
#define CUDA_GRAPH_HPP

#include <cuda_runtime.h>

#include <functional>
#include <stdexcept>

#include "gpu_session.hpp"

class CudaGraph {
   public:
    CudaGraph(GpuSession& gpu_session, const std::function<void(const cudaStream_t&)>& func)
        : graph(nullptr), instance(nullptr), m_gpu_session(gpu_session) {
        setup(func);
    }

    ~CudaGraph() {
        if (instance) {
            cudaGraphExecDestroy(instance);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }

    void run() {
        if (instance) {
            cudaGraphLaunch(instance, 0);
            cudaStreamSynchronize(0);
        } else {
            throw std::runtime_error("Cuda graph instance is not initialized.");
        }
    }

   private:
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    GpuSession& m_gpu_session;

    void setup(const std::function<void(const cudaStream_t&)>& func) {
        cudaStream_t stream = m_gpu_session.sessionStream();

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        func(stream);
        cudaStreamEndCapture(stream, &graph);

        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    }
};

#endif  // CUDA_GRAPH_HPP
