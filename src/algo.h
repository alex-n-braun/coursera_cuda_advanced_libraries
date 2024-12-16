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

#include <vector>

#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <helper_cuda.h>

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
