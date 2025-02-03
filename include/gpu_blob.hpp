#ifndef GPU_BLOB_HPP
#define GPU_BLOB_HPP

#include <cstddef>

class GpuBlob {
   public:
    explicit GpuBlob(std::size_t size);
    GpuBlob(const GpuBlob&) = delete;
    GpuBlob& operator=(const GpuBlob&) = delete;
    GpuBlob(GpuBlob&& other) noexcept : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
    }
    GpuBlob& operator=(GpuBlob&& other) noexcept {
        if (this != &other) {
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
        }
        return *this;
    }

    ~GpuBlob();
    void copyFrom(const void* data);
    void copyTo(void* data) const;
    void* data();
    [[nodiscard]] const void* data() const;
    [[nodiscard]] std::size_t size() const;

   private:
    void* m_data;
    std::size_t m_size;
};

#endif  // GPU_BLOB_HPP
