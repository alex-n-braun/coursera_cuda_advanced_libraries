#ifndef CUDA_GRAPH_HPP
#define CUDA_GRAPH_HPP

#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>

class CudaGraph {
public:
    CudaGraph() : graph(nullptr), instance(nullptr) {}

    ~CudaGraph() {
        if (instance) {
            cudaGraphExecDestroy(instance);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }

    void setup(const std::function<void()>& func) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        func();
        cudaStreamEndCapture(stream, &graph);

        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

        cudaStreamDestroy(stream);
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
};

#endif // CUDA_GRAPH_HPP
