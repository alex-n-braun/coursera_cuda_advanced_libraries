#ifndef TYPES_HPP
#define TYPES_HPP

#include <cassert>
#include <gpuBlob.hpp>
#include <vector>

template <typename T, std::size_t Filters, std::size_t Width, std::size_t Height,
          std::size_t Channels>
class Kernel {
   public:
    Kernel(const std::vector<T>& numbers) : m_kernel(numbers.size() * sizeof(T)) {
        if (!(numbers.size() == Filters * Width * Height * Channels)) {
            throw std::runtime_error("Kernel size does not match the expected size");
        }
        m_kernel.copy_from(numbers.data());
    }
    const T* data() const { return static_cast<const T*>(m_kernel.data()); }

    static constexpr std::size_t filters() { return Filters; }
    static constexpr std::size_t width() { return Width; }
    static constexpr std::size_t height() { return Height; }
    static constexpr std::size_t channels() { return Channels; }

   private:
    GpuBlob m_kernel;
};

template <typename T, std::size_t Channels>
class ImageCPU {
   public:
    using type_t = T;
    ImageCPU(std::size_t width, std::size_t height)
        : m_width(width), m_height(height), m_data(width * height * Channels) {}
    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t pitch() const { return m_width * Channels * sizeof(type_t); }
    std::size_t numPixels() const { return width() * height(); }
    std::size_t size() const { return m_data.size(); }
    static constexpr std::size_t channels() { return Channels; }
    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }

   private:
    std::size_t m_width;
    std::size_t m_height;
    std::vector<T> m_data;
};

template <typename T, std::size_t Channels>
class ImageGPU {
   public:
    using type_t = T;
    ImageGPU(std::size_t width, std::size_t height)
        : m_width(width), m_height(height), m_data(width * height * Channels * sizeof(T)) {}
    ImageGPU(const ImageCPU<T, Channels>& image) : ImageGPU(image.width(), image.height()) {
        m_data.copy_from(image.data());
    }
    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t pitch() const { return m_width * Channels * sizeof(type_t); }
    std::size_t numPixels() const { return width() * height(); }
    std::size_t size() const { return width() * height() * Channels; }
    static constexpr std::size_t channels() { return Channels; }
    void copy_from(const ImageCPU<T, Channels>& image) { m_data.copy_from(image.data()); }
    void copy_to(ImageCPU<T, Channels>& image) const { m_data.copy_to(image.data()); }
    T* data() { return static_cast<T*>(m_data.data()); }
    const T* data() const { return static_cast<const T*>(m_data.data()); }

   private:
    std::size_t m_width;
    std::size_t m_height;
    GpuBlob m_data;
};

#endif  // TYPES_HPP
