/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef IO_HPP
#define IO_HPP

#include <FreeImage.h>

#include <opencv2/opencv.hpp>
#include <string>

#include "types.hpp"

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage) {
    throw std::runtime_error(zMessage);
}

#define IO_ASSERT(C)                                                                          \
    {                                                                                         \
        if (!(C))                                                                             \
            throw std::runtime_error(std::string(#C " assertion failed! ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                               \
    }

// Load an RGB image from disk.
ImageCPU<std::uint8_t, 4> loadImage(const std::string &fileName) {
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(fileName.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN) {
        eFormat = FreeImage_GetFIFFromFilename(fileName.c_str());
    }

    IO_ASSERT(eFormat != FIF_UNKNOWN);
    // check that the plugin has reading capabilities ...
    FIBITMAP *pBitmap;

    if (FreeImage_FIFSupportsReading(eFormat)) {
        pBitmap = FreeImage_Load(eFormat, fileName.c_str());
    }

    IO_ASSERT(pBitmap != 0);
    // make sure this is an 8-bit single channel image
    IO_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_RGB);
    IO_ASSERT(!FreeImage_IsTransparent(pBitmap));
    IO_ASSERT(FreeImage_GetBPP(pBitmap) == 32);

    // create an ImageCPU to receive the loaded image data
    ImageCPU<std::uint8_t, 4> image(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

    memcpy(image.data(), FreeImage_GetBits(pBitmap), image.size() * sizeof(std::uint8_t));

    return image;
}

// Save an RGB image to disk.
void saveImage(const std::string &fileName, const ImageCPU<std::uint8_t, 4> &image) {
    // create the result image storage using FreeImage so we can easily
    // save
    FIBITMAP *pResultBitmap =
        FreeImage_Allocate(image.width(), image.height(), 32 /* bits per pixel */);
    IO_ASSERT(pResultBitmap != nullptr);

    // Copy the image data directly without mirroring
    memcpy(FreeImage_GetBits(pResultBitmap), image.data(), image.size() * sizeof(std::uint8_t));

    unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
    IO_ASSERT(nDstPitch == image.pitch());

    // now save the result image
    IO_ASSERT(FreeImage_Save(FIF_PNG, pResultBitmap, fileName.c_str(), 0) == TRUE);
}

void loadFromFrame(const cv::Mat &frame, ImageCPU<std::uint8_t, 4> &image) {
    // Ensure the input frame has 4 channels (RGBA)
    cv::Mat rgbaFrame;
    if (frame.channels() == 3) {
        // Convert from BGR to RGBA
        cv::cvtColor(frame, rgbaFrame, cv::COLOR_BGR2RGBA);
    } else if (frame.channels() == 1) {
        // Convert from grayscale to RGBA
        cv::cvtColor(frame, rgbaFrame, cv::COLOR_GRAY2RGBA);
    } else {
        rgbaFrame = frame;
    }

    // Copy pixel data row by row, considering pitch
    const int rowSize = rgbaFrame.cols * rgbaFrame.elemSize();  // Effective row size in bytes
    for (int row = 0; row < rgbaFrame.rows; ++row) {
        std::memcpy(
            image.data() + row * image.pitch(),     // Destination (row by row, respecting pitch)
            rgbaFrame.data + row * rgbaFrame.step,  // Source
            rowSize                                 // Number of bytes in the row
        );
    }
}

void saveToFrame(const ImageCPU<std::uint8_t, 4> &image, cv::Mat &mat) {
    // Copy row by row, respecting pitch
    cv::Mat rgbaFrame(image.height(), image.width(), CV_8UC4);
    const int rowSize = image.pitch();  // 4 bytes per pixel (RGBA)
    for (int row = 0; row < image.height(); ++row) {
        std::memcpy(rgbaFrame.data + row * rgbaFrame.step,  // Destination row
                    image.data() + row * rowSize,           // Source row
                    rowSize                                 // Number of bytes in the row
        );
    }
    cv::cvtColor(rgbaFrame, mat, cv::COLOR_RGBA2BGR);
}

#endif  // IO_HPP
