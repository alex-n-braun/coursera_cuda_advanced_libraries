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

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "cli.hpp"
#include "cuda_graph.hpp"
#include "filter.hpp"
#include "gpu_session.hpp"
#include "helper_cuda.h"
#include "io.hpp"
#include "timer.hpp"
#include "types.hpp"

bool printCUDAinfo() {
    int driver_version;  // NOLINT(cppcoreguidelines-init-variables)
    cudaDriverGetVersion(&driver_version);
    int runtime_version;  // NOLINT(cppcoreguidelines-init-variables)
    cudaRuntimeGetVersion(&runtime_version);

    constexpr int kMajorDivisor = 1000;
    constexpr int kMinorDivisor = 10;
    std::cout << "  CUDA Driver  Version: " << driver_version / kMajorDivisor << "."
              << (driver_version % kMajorDivisor) / kMinorDivisor << "\n";
    std::cout << "  CUDA Runtime Version: " << runtime_version / kMajorDivisor << "."
              << (runtime_version % kMajorDivisor) / kMinorDivisor << "\n";

    // Min spec is SM 1.0 devices
    return checkCudaCapabilities(1, 0);
}

int processVideo(const std::string &infilename, const std::string &outfilename) {
    cv::VideoCapture capture(infilename);
    const int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
    const int fourcc = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

    cv::VideoWriter writer(outfilename, fourcc, fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video file: " << outfilename << '\n';
        return -1;
    }

    cv::Mat frame;

    ImageCPU<std::uint8_t, 4> image_src(frame_width, frame_height);
    ImageCPU<std::uint8_t, 4> image_dest(image_src.width(), image_src.height());
    GpuSession gpu_session;
    Filter filter(gpu_session, frame_width, frame_height);

    CudaGraph graph{gpu_session,
                    [&filter](const cudaStream_t &stream) { filter.prepareGraph(stream); }};

    // measure runtime: start
    Timer global_timer;
    Timer processing_timer;
    Timer gpu_timer;

    global_timer.start();
    int count = 0;
    while (true) {
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        loadFromFrame(frame, image_src);
        processing_timer.start();
        filter.setInput(image_src);
        gpu_timer.start();
        graph.run();
        gpu_timer.stop();
        filter.retrieveOutput(image_dest);
        processing_timer.stop();
        saveToFrame(image_dest, frame);
        writer.write(frame);
        ++count;
    }
    global_timer.stop();

    std::cout << "Elapsed time in nanoseconds:" << '\n';
    std::cout << "\t\t\t  Total\t\t  per frame" << '\n';
    std::cout << "incl. io\t\t" << global_timer.duration() << "\t"
              << global_timer.duration() / count << '\n';
    std::cout << "excl. io\t\t" << processing_timer.duration() << "\t"
              << processing_timer.duration() / count << '\n';
    std::cout << "gpu\t\t\t" << gpu_timer.duration() << "\t" << gpu_timer.duration() / count
              << '\n';

    capture.release();
    writer.release();

    return 0;
}

int processPng(const std::string &infilename, const std::string &outfilename) {
    // load rgb image from disk
    const ImageCPU<std::uint8_t, 4> image_src = loadImage(infilename);

    GpuSession gpu_session;
    const Filter filter{gpu_session, image_src.width(), image_src.height()};
    // declare a host image for the result
    ImageCPU<std::uint8_t, 4> image_dest(image_src.width(), image_src.height());
    // measure runtime: start
    auto start = std::chrono::high_resolution_clock::now();
    filter.filter(image_src, image_dest);

    // measure runtime: end
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " nanoseconds" << '\n';

    saveImage(outfilename, image_dest);
    std::cout << "Saved image: " << outfilename << '\n';

    return 0;
}

int main(int argc, char *argv[]) {  // NOLINT(bugprone-exception-escape)
    findCudaDevice(
        argc, const_cast<const char **>(argv));  // NOLINT(cppcoreguidelines-pro-type-const-cast)

    if (!printCUDAinfo()) {
        return 0;
    }

    const Cli cli{argc, argv};
    const std::string filename = cli.file_name;
    const std::string result_filename = cli.result_file_name;

    if (cli.file_extension == ".mp4") {
        return processVideo(filename, result_filename);
    }
    if (cli.file_extension == ".png") {
        return processPng(filename, result_filename);
    }

    return 0;
}
