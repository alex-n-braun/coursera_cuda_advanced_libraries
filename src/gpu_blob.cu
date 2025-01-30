#include <cuda_runtime.h>

#include <iostream>

#include "gpu_blob.hpp"

GpuBlob::GpuBlob(std::size_t size) : m_size(size), m_data(nullptr) {
    cudaError_t err = cudaMalloc(&m_data, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory at " + std::string(__FILE__) +
                                 ":" + std::to_string(__LINE__));
    }
}
GpuBlob::~GpuBlob() {
    cudaError_t err = cudaFree(m_data);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device memory at " << __FILE__ << ":" << __LINE__ << '\n';
        std::terminate();
    }
}
void GpuBlob::copyFrom(const void* data) {
    cudaError_t err = cudaMemcpy(m_data, data, m_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to device at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }
}
void GpuBlob::copyTo(void* data) const {
    cudaError_t err = cudaMemcpy(data, m_data, m_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from device at " + std::string(__FILE__) +
                                 ":" + std::to_string(__LINE__));
    }
}
void* GpuBlob::data() { return m_data; }
const void* GpuBlob::data() const { return m_data; }
std::size_t GpuBlob::size() const { return m_size; }
