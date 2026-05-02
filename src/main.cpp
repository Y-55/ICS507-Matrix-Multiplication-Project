#include "algorithms.hpp"
#include "io.hpp"
#include "timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct MethodRun {
    std::string name;
    std::string parameterTag;
    std::vector<std::string> metadataLines;
    std::function<Matrix()> multiply;
};

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <input-file> [threads] [strassen-base-case] [mode]\n"
              << "  mode: all | Sequential | ParMtrixMult | Strassen | ParStrassen\n";
}

int availableThreads(int requestedThreads) {
    if (requestedThreads > 0) {
        return requestedThreads;
    }

#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool shouldRunMethod(const std::string& requestedMode, const std::string& methodName) {
    return requestedMode == "all" || requestedMode == methodName;
}

void runTimedMethod(
    const MethodRun& method,
    const MatrixInput& input,
    const std::filesystem::path& outputDirectory,
    int coresUsed) {
    ScopedTimer timer;
    Matrix result = method.multiply();
    const auto elapsed = timer.elapsed();
    const std::string elapsedText = formatDuration(elapsed);

    writeMatrixFile(outputDirectory, input.stem, method.name, method.parameterTag, result);
    writeInfoFile(
        outputDirectory,
        input.stem,
        method.name,
        elapsedText,
        coresUsed,
        method.parameterTag,
        method.metadataLines);

    std::cout << method.name << " completed in " << elapsedText
              << " using " << coresUsed << " core(s)\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 5) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path inputPath = argv[1];
        const int threads = argc >= 3 ? std::stoi(argv[2]) : availableThreads(0);
        const int strassenBaseCase = argc >= 4 ? std::stoi(argv[3]) : 64;
        const std::string mode = argc >= 5 ? argv[4] : "all";

        if (threads <= 0) {
            throw std::invalid_argument("threads must be a positive integer");
        }
        if (strassenBaseCase <= 0) {
            throw std::invalid_argument("strassen-base-case must be a positive integer");
        }

#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif

        const MatrixInput input = readInputFile(inputPath);
        const std::filesystem::path outputDirectory = std::filesystem::current_path();

        std::vector<MethodRun> methods = {
            // {"Sequential", "", {}, [&]() { return multiplySequential(input.a, input.b); }},
            {
                "ParMtrixMult",
                "threads-" + std::to_string(threads),
                {"threads: " + std::to_string(threads)},
                [&]() { return multiplyParMatrixMult(input.a, input.b, threads); },
            },
            // {
            //     "Strassen",
            //     "threads-" + std::to_string(threads) + "-basecase-" + std::to_string(strassenBaseCase),
            //     {
            //         "threads: " + std::to_string(threads),
            //         "basecase: " + std::to_string(strassenBaseCase),
            //     },
            //     [&]() { return multiplyStrassen(input.a, input.b, strassenBaseCase); },
            // },
            {
                "ParStrassen",
                "threads-" + std::to_string(threads) + "-basecase-" + std::to_string(strassenBaseCase),
                {
                    "threads: " + std::to_string(threads),
                    "basecase: " + std::to_string(strassenBaseCase),
                },
                [&]() { return multiplyParStrassen(input.a, input.b, strassenBaseCase, threads); },
            },
        };

        bool ranAnyMethod = false;

        for (const MethodRun& method : methods) {
            if (!shouldRunMethod(mode, method.name)) {
                continue;
            }

            ranAnyMethod = true;
            runTimedMethod(method, input, outputDirectory, threads);
        }

        if (!ranAnyMethod) {
            throw std::invalid_argument("unknown mode: " + mode);
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
