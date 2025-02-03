#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstddef>
#include <gpu_blob.hpp>
#include <stdexcept>
#include <vector>

template <typename DataT, std::size_t kFiltersT, std::size_t kWidthT, std::size_t kHeightT,
          std::size_t kChannelsT>
class Kernel {
   public:
    explicit Kernel(const std::vector<DataT>& numbers) : m_kernel(numbers.size() * sizeof(DataT)) {
        if (!(numbers.size() == kFiltersT * kWidthT * kHeightT * kChannelsT)) {
            throw std::runtime_error("Kernel size does not match the expected size");
        }
        m_kernel.copyFrom(numbers.data());
    }
    const DataT* data() const { return static_cast<const DataT*>(m_kernel.data()); }

    static constexpr std::size_t filters() { return kFiltersT; }
    static constexpr std::size_t width() { return kWidthT; }
    static constexpr std::size_t height() { return kHeightT; }
    static constexpr std::size_t channels() { return kChannelsT; }

   private:
    GpuBlob m_kernel;
};

template <typename DataT, std::size_t kChannelsT>
class ImageCPU {
   public:
    ImageCPU(std::size_t width, std::size_t height)
        : m_width(width), m_height(height), m_data(width * height * kChannelsT) {}
    [[nodiscard]] std::size_t width() const { return m_width; }
    [[nodiscard]] std::size_t height() const { return m_height; }
    [[nodiscard]] std::size_t pitch() const { return m_width * kChannelsT * sizeof(DataT); }
    [[nodiscard]] std::size_t numPixels() const { return width() * height(); }
    [[nodiscard]] std::size_t size() const { return m_data.size(); }
    [[nodiscard]] static constexpr std::size_t channels() { return kChannelsT; }
    [[nodiscard]] DataT* data() { return m_data.data(); }
    [[nodiscard]] const DataT* data() const { return m_data.data(); }

   private:
    std::size_t m_width;
    std::size_t m_height;
    std::vector<DataT> m_data;
};

template <typename DataT, std::size_t kChannelsT>
class ImageGPU {
   public:
    ImageGPU(std::size_t width, std::size_t height)
        : m_width(width), m_height(height), m_data(width * height * kChannelsT * sizeof(DataT)) {}
    // NOLINTNEXTLINE(*-member-init)
    explicit ImageGPU(const ImageCPU<DataT, kChannelsT>& image)
        : ImageGPU(image.width(), image.height()) {
        m_data.copyFrom(image.data());
    }
    [[nodiscard]] std::size_t width() const { return m_width; }
    [[nodiscard]] std::size_t height() const { return m_height; }
    [[nodiscard]] std::size_t pitch() const { return m_width * kChannelsT * sizeof(DataT); }
    [[nodiscard]] std::size_t numPixels() const { return width() * height(); }
    [[nodiscard]] std::size_t size() const { return width() * height() * kChannelsT; }
    [[nodiscard]] static constexpr std::size_t channels() { return kChannelsT; }
    void copyFrom(const ImageCPU<DataT, kChannelsT>& image) { m_data.copyFrom(image.data()); }
    void copyTo(ImageCPU<DataT, kChannelsT>& image) const { m_data.copyTo(image.data()); }
    [[nodiscard]] DataT* data() { return static_cast<DataT*>(m_data.data()); }
    [[nodiscard]] const DataT* data() const { return static_cast<const DataT*>(m_data.data()); }

   private:
    std::size_t m_width;
    std::size_t m_height;
    GpuBlob m_data;
};

#endif  // TYPES_HPP
