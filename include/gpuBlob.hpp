#ifndef GPU_BLOB_HPP
#define GPU_BLOB_HPP

#include <stdexcept>

class GpuBlob {
   public:
    GpuBlob(std::size_t size);
    GpuBlob(const GpuBlob&) = delete;
    GpuBlob& operator=(const GpuBlob&) = delete;
    GpuBlob(GpuBlob&& other) { m_data = other.m_data; m_size = other.m_size; other.m_data = nullptr; }
    GpuBlob& operator=(GpuBlob&& other) {
        if (this != &other) {
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
        }
        return *this;
    }
    
    ~GpuBlob();
    void copy_from(const void* data);
    void copy_to(void* data) const;
    void* data();
    const void* data() const;
    std::size_t size() const;

   private:
    void* m_data;
    std::size_t m_size;
};

#endif  // GPU_BLOB_HPP
