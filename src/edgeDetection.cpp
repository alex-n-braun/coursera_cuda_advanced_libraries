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

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo()
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

struct Cli {
    Cli(int argc, char* argv[]) {
        char *filePath;
        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }
        if (filePath)
        {
            filename = filePath;
        }
        else
        {
            filename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // filename[0].
        int file_errors = 0;
        std::ifstream infile(filename, std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << filename
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << filename << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        resultFilename = filename;
        std::string::size_type dot = resultFilename.rfind('.');
        if (dot != std::string::npos)
        {
            resultFilename = resultFilename.substr(0, dot);
        }
        resultFilename += "_rotate.pgm";
        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            resultFilename = outputFilePath;
        }

        angle = -24.0; // Rotation angle in degrees
        if (checkCmdLineFlag(argc, (const char **)argv, "angle"))
        {
            char *outputFilePath;
            angle = getCmdLineArgumentFloat(argc, (const char **)argv, "angle");
        }

    }

    std::string filename;
    std::string resultFilename;
    double angle;
};

// Load an RGB image from disk.
void
loadImage(const std::string &rFileName, npp::ImageCPU_8u_C4 &rImage)
{
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
    }

    NPP_ASSERT(eFormat != FIF_UNKNOWN);
    // check that the plugin has reading capabilities ...
    FIBITMAP *pBitmap;

    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, rFileName.c_str());
    }

    NPP_ASSERT(pBitmap != 0);
    // make sure this is an 8-bit single channel image
    NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_RGB);
    NPP_ASSERT(!FreeImage_IsTransparent(pBitmap));
    NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 32);

    // create an ImageCPU to receive the loaded image data
    npp::ImageCPU_8u_C4 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

    // Copy the FreeImage data into the new ImageCPU
    unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
    const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) -1);
    Npp8u *pDstLine = oImage.data();
    unsigned int nDstPitch = oImage.pitch();

    for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
    {
        memcpy(pDstLine, pSrcLine, oImage.width() * sizeof(Npp8u) * 4);
        pSrcLine -= nSrcPitch;
        pDstLine += nDstPitch;
    }

    // swap the user given image with our result image, effecively
    // moving our newly loaded image data into the user provided shell
    oImage.swap(rImage);
}

// Save an gray-scale image to disk.
void
saveImage(const std::string &rFileName, const npp::ImageCPU_8u_C4 &rImage)
{
    // create the result image storage using FreeImage so we can easily
    // save
    FIBITMAP *pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 32 /* bits per pixel */);
    NPP_ASSERT_NOT_NULL(pResultBitmap);
    unsigned int nDstPitch   = FreeImage_GetPitch(pResultBitmap);
    Npp8u *pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height()-1);
    const Npp8u *pSrcLine = rImage.data();
    unsigned int nSrcPitch = rImage.pitch();

    for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
    {
        memcpy(pDstLine, pSrcLine, rImage.width() * sizeof(Npp8u) * 4);
        pSrcLine += nSrcPitch;
        pDstLine -= nDstPitch;
    }

    // now save the result image
    bool bSuccess;
    bSuccess = FreeImage_Save(FIF_PNG, pResultBitmap, rFileName.c_str(), 0) == TRUE;
    NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
}


class Kernel {
public:
    Kernel(const std::vector<Npp32f>& numbers) {
        cudaMalloc(&d_kernel, 3 * 3 * sizeof(Npp32f));
        cudaMemcpy(d_kernel, &numbers[0], 3 * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
    }
    ~Kernel() {
        cudaFree(d_kernel);
    }
    const Npp32f* data() const { return d_kernel; }
private:
    Npp32f *d_kernel;
};

class EdgeFilter_8u_C4 {
public:
    EdgeFilter_8u_C4(unsigned int width, unsigned int height) 
    : oDeviceSrc(width, height)
    , oDeviceTmp(width, height)
    , edgesImage(width, height)
    , oDeviceDstBroadcast(width, height)
    , kernel_horz({
            -0.25,  0,  0.25,
            -0.5,   0,  0.5,
            -0.25,  0,  0.25
        })
    , kernel_vert({
            -0.25, -0.5, -0.25,
             0,     0,    0,
             0.25,  0.5,  0.25
        })
    {}

    void filter(const npp::ImageCPU_8u_C4& input, npp::ImageCPU_8u_C4& output) const {

        const int imageWidth = static_cast<int>(input.width());
        const int imageHeight = static_cast<int>(input.height());
        const NppiSize roiSize{imageWidth, imageHeight};

        // copy from the host image,
        // i.e. upload host to device
        oDeviceSrc.copyFrom(const_cast<Npp8u*>(input.data()), input.pitch());

        // convert to gray-scale img (tmp)
        NPP_CHECK_NPP(nppiRGBToGray_8u_AC4C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceTmp.data(), oDeviceTmp.pitch(),
            roiSize
        ));

        // apply sobel operator
        edgeFilter(oDeviceTmp, edgesImage, kernel_horz);
        edgeFilter(oDeviceTmp, oDeviceTmp, kernel_vert);
        NPP_CHECK_NPP(nppiOr_8u_C1R(
            edgesImage.data(), edgesImage.pitch(),
            oDeviceTmp.data(), oDeviceTmp.pitch(),
            oDeviceTmp.data(), oDeviceTmp.pitch(),
            roiSize
        ));

        // boaadcast gray-scale edges to RGBA image
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
            oDeviceTmp.data(), oDeviceTmp.pitch(),                   
            oDeviceDstBroadcast.data(), oDeviceDstBroadcast.pitch(), 
            roiSize
        ));
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
            oDeviceTmp.data(), oDeviceTmp.pitch(),                   
            oDeviceDstBroadcast.data() + 1, oDeviceDstBroadcast.pitch(),
            roiSize
        ));
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
            oDeviceTmp.data(), oDeviceTmp.pitch(),                   
            oDeviceDstBroadcast.data() + 2, oDeviceDstBroadcast.pitch(),
            roiSize
        ));
        NPP_CHECK_NPP(nppiSet_8u_C4CR(
            255,                                                        
            oDeviceDstBroadcast.data() + 3, oDeviceDstBroadcast.pitch(),
            roiSize                                                     
        ));

        // combine edges with rgba input image
        NPP_CHECK_NPP(nppiMul_8u_C4RSfs(
            oDeviceDstBroadcast.data(), oDeviceDstBroadcast.pitch(),
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            roiSize,
            8
        ));

        // and copy the device result data into it
        oDeviceSrc.copyTo(output.data(), output.pitch());
    }
private:

    void edgeFilter(const npp::ImageNPP_8u_C1& deviceSrc, npp::ImageNPP_8u_C1& deviceDest, const Kernel& kernel) const {
        const int imageWidth = static_cast<int>(deviceSrc.width());
        const int imageHeight = static_cast<int>(deviceSrc.height());

        npp::ImageNPP_16s_C1 oDeviceTmp(imageWidth, imageHeight);

        // Define filter parameters
        NppiSize kernelSize = {3, 3};                   // Kernel size
        NppiPoint anchor = {1, 1};                      // Anchor point (center of the kernel)
        NppiSize roiSize = {imageWidth, imageHeight};   // ROI size (full image)

        // Apply the kernel using nppiFilter
        NPP_CHECK_NPP(nppiFilter32f_8u16s_C1R(
            deviceSrc.data(), deviceSrc.pitch(),        // Input image and stride
            oDeviceTmp.data(), oDeviceTmp.pitch(),      // Output image and stride
            roiSize,                                    // Region of interest (ROI)
            kernel.data(), kernelSize, anchor     // Kernel and anchor point
        ));

        NPP_CHECK_NPP(nppiAbs_16s_C1R(oDeviceTmp.data(), oDeviceTmp.pitch(), oDeviceTmp.data(), oDeviceTmp.pitch(), roiSize));
        NPP_CHECK_NPP(nppiConvert_16s8u_C1R(oDeviceTmp.data(), oDeviceTmp.pitch(), deviceDest.data(), deviceDest.pitch(), roiSize));
    }
    
    mutable npp::ImageNPP_8u_C4 oDeviceSrc;
    mutable npp::ImageNPP_8u_C1 oDeviceTmp;
    mutable npp::ImageNPP_8u_C1 edgesImage;
    mutable npp::ImageNPP_8u_C4 oDeviceDstBroadcast;
    const Kernel kernel_horz;
    const Kernel kernel_vert;
};


int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo() == false)
        {
            exit(EXIT_SUCCESS);
        }

        Cli cli{argc, argv};
        std::string filename = cli.filename;
        double angle = cli.angle;
        std::string resultFilename = cli.resultFilename;

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C4 oHostSrc;
        // load gray-scale image from disk
        loadImage(filename, oHostSrc);

        EdgeFilter_8u_C4 filter{oHostSrc.width(), oHostSrc.height()};
        // declare a host image for the result
        npp::ImageCPU_8u_C4 oHostDst(oHostSrc.width(), oHostSrc.height());
        // measure runtime: start
        auto start = std::chrono::high_resolution_clock::now();
        filter.filter(oHostSrc, oHostDst);

        // measure runtime: end
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "Elapsed time: " << duration << " nanoseconds" << std::endl;

        saveImage(resultFilename, oHostDst);
        std::cout << "Saved image: " << resultFilename << std::endl;

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }

    return 0;
}
