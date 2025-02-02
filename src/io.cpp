
#include "io.hpp"

#include <FreeImage.h>
#include <opencv2/core/hal/interface.h>

#include <cstdint>
#include <cstring>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

#include "types.hpp"

namespace {

constexpr void ioAssert(bool condition, const char *condition_str, const char *file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(condition_str) + " assertion failed! " + file + ":" +
                                 std::to_string(line));
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define IO_ASSERT(C) \
    { ioAssert((C), #C, __FILE__, __LINE__); }

inline void freeImageErrorHandler(FREE_IMAGE_FORMAT /*oFif*/, const char *message) {
    throw std::runtime_error(message);
}

}  // namespace

// Load an RGB image from disk.
ImageCPU<std::uint8_t, 4> loadImage(const std::string &file_name) {
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(freeImageErrorHandler);

    FREE_IMAGE_FORMAT e_format = FreeImage_GetFileType(file_name.c_str());

    // no signature? try to guess the file format from the file extension
    if (e_format == FIF_UNKNOWN) {
        e_format = FreeImage_GetFIFFromFilename(file_name.c_str());
    }

    IO_ASSERT(e_format != FIF_UNKNOWN);
    // check that the plugin has reading capabilities ...
    FIBITMAP *p_bitmap = nullptr;

    if (FreeImage_FIFSupportsReading(e_format) != 0) {
        p_bitmap = FreeImage_Load(e_format, file_name.c_str());
    }

    IO_ASSERT(p_bitmap != nullptr);
    // make sure this is an 8-bit single channel image
    IO_ASSERT(FreeImage_GetColorType(p_bitmap) == FIC_RGB);
    IO_ASSERT(!FreeImage_IsTransparent(p_bitmap));
    IO_ASSERT(FreeImage_GetBPP(p_bitmap) == 32);

    // create an ImageCPU to receive the loaded image data
    ImageCPU<std::uint8_t, 4> image(FreeImage_GetWidth(p_bitmap), FreeImage_GetHeight(p_bitmap));

    std::memcpy(image.data(), FreeImage_GetBits(p_bitmap), image.size() * sizeof(std::uint8_t));

    return image;
}

// Save an RGB image to disk.
void saveImage(const std::string &file_name, const ImageCPU<std::uint8_t, 4> &image) {
    // create the result image storage using FreeImage so we can easily
    // save
    constexpr int kBitsperpixel = 32;
    FIBITMAP *p_result_bitmap = FreeImage_Allocate(static_cast<int>(image.width()),
                                                   static_cast<int>(image.height()), kBitsperpixel);
    IO_ASSERT(p_result_bitmap != nullptr);

    // Copy the image data directly without mirroring
    memcpy(FreeImage_GetBits(p_result_bitmap), image.data(), image.size() * sizeof(std::uint8_t));

    const unsigned int n_dst_pitch = FreeImage_GetPitch(p_result_bitmap);
    IO_ASSERT(n_dst_pitch == image.pitch());

    // now save the result image
    IO_ASSERT(FreeImage_Save(FIF_PNG, p_result_bitmap, file_name.c_str(), 0) == TRUE);
}

void loadFromFrame(const cv::Mat &frame, ImageCPU<std::uint8_t, 4> &image) {
    // Ensure the input frame has 4 channels (RGBA)
    cv::Mat rgba_frame;
    constexpr int kRgbChannels = 3;
    constexpr int kGrayscaleChannels = 1;
    if (frame.channels() == kRgbChannels) {
        // Convert from BGR to RGBA
        cv::cvtColor(frame, rgba_frame, cv::COLOR_BGR2RGBA);
    } else if (frame.channels() == kGrayscaleChannels) {
        // Convert from grayscale to RGBA
        cv::cvtColor(frame, rgba_frame, cv::COLOR_GRAY2RGBA);
    } else {
        rgba_frame = frame;
    }

    // Copy pixel data row by row, considering pitch
    const std::size_t row_size =
        rgba_frame.cols * rgba_frame.elemSize();  // Effective row size in bytes
    for (int row = 0; row < rgba_frame.rows; ++row) {
        std::memcpy(
            image.data() + row * image.pitch(),       // Destination (row by row, respecting pitch)
            rgba_frame.data + row * rgba_frame.step,  // Source
            row_size                                  // Number of bytes in the row
        );
    }
}

void saveToFrame(const ImageCPU<std::uint8_t, 4> &image, cv::Mat &mat) {
    // Copy row by row, respecting pitch
    // NOLINTNEXTLINE(*-signed-bitwise)
    const cv::Mat rgba_frame(static_cast<int>(image.height()), static_cast<int>(image.width()),
                             CV_8UC4);           // NOLINT(hicpp-signed-bitwise)
    const std::size_t row_size = image.pitch();  // 4 bytes per pixel (RGBA)
    for (int row = 0; row < image.height(); ++row) {
        std::memcpy(rgba_frame.data + row * rgba_frame.step,  // Destination row
                    image.data() + row * row_size,            // Source row
                    row_size                                  // Number of bytes in the row
        );
    }
    cv::cvtColor(rgba_frame, mat, cv::COLOR_RGBA2BGR);
}
