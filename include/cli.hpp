#ifndef CLI_HPP
#define CLI_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "helper_string.h"

struct Cli {
    Cli(int argc, char *argv[]) {  // NOLINT(*-avoid-c-arrays)
        char *file_path = nullptr;
        if (checkCmdLineFlag(
                argc,
                const_cast<const char **>(argv),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                "input")) {
            getCmdLineArgumentString(
                argc,
                const_cast<const char **>(argv),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                "input", &file_path);
        }
        if (file_path != nullptr) {
            file_name = file_path;
        } else {
            file_name = "data/Lena.png";
        }

        // if we specify the filename at the command line, then we only test
        // filename[0].
        int file_errors = 0;
        std::ifstream infile(file_name, std::ifstream::in);

        if (infile.good()) {
            std::cout << "edgeDetection opened: <" << file_name << "> successfully!" << '\n';
            file_errors = 0;
            infile.close();
        } else {
            std::cout << "edgeDetection unable to open: <" << file_name << ">" << '\n';
            file_errors++;
            infile.close();
        }

        if (file_errors > 0) {
            throw std::runtime_error("File errors encountered, exiting.");
        }

        file_extension = getFileExtension(file_name);

        const std::filesystem::path path(file_name);
        result_file_name = (path.parent_path() / path.stem()).string() + "_edge" + file_extension;

        if (checkCmdLineFlag(
                argc,
                const_cast<const char **>(argv),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                "output")) {
            char *output_file_path;  // NOLINT(cppcoreguidelines-init-variables)
            getCmdLineArgumentString(
                argc,
                const_cast<const char **>(argv),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                "output", &output_file_path);
            result_file_name = output_file_path;
        }
        if (file_extension != getFileExtension(result_file_name)) {
            throw std::runtime_error(
                "input and output filename need to have the same file extension");
        }

        std::cout << "output File: " << result_file_name << '\n';
        std::cout << "extension: " << file_extension << '\n';
    }

    std::string file_name;
    std::string result_file_name;
    std::string file_extension;

   private:
    static std::string getFileExtension(const std::string &filename) {
        return std::filesystem::path(filename).extension().string();
    }
};

#endif  // CLI_HPP
