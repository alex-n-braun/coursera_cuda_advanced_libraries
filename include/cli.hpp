#ifndef CLI_HPP
#define CLI_HPP

#include <filesystem>
#include <iostream>

#include "helper_string.h"

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

#endif  // CLI_HPP
