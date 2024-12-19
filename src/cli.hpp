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

#include <helper_string.h>

#include <filesystem>
#include <iostream>

struct Cli {
    Cli(int argc, char *argv[]) {
        char *filePath;
        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        } else {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }
        if (filePath) {
            fileName = filePath;
        } else {
            fileName = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // filename[0].
        int file_errors = 0;
        std::ifstream infile(fileName, std::ifstream::in);

        if (infile.good()) {
            std::cout << "edgeDetection opened: <" << fileName << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        } else {
            std::cout << "edgeDetection unable to open: <" << fileName << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0) {
            exit(EXIT_FAILURE);
        }

        fileExtension = getFileExtension(fileName);

        std::filesystem::path path(fileName);
        resultFilename = (path.parent_path() / path.stem()).string() + "_edge" + fileExtension;

        if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            resultFilename = outputFilePath;
        }
        if (fileExtension != getFileExtension(resultFilename)) {
            throw std::runtime_error(
                "input and output filename need to have the same file extension");
        }

        std::cout << "output File: " << resultFilename << std::endl;
        std::cout << "extension: " << fileExtension << std::endl;
    }

    std::string fileName;
    std::string resultFilename;
    std::string fileExtension;

   private:
    static std::string getFileExtension(const std::string &filename) {
        return std::filesystem::path(filename).extension().string();
    }
};
