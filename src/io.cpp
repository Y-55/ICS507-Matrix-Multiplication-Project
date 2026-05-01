#include "io.hpp"

#include <fstream>
#include <stdexcept>

namespace {

void readMatrixValues(std::ifstream& input, Matrix& matrix) {
    const std::size_t n = matrix.size();
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            if (!(input >> matrix(row, column))) {
                throw std::runtime_error("input file does not contain enough matrix values");
            }
        }
    }
}

} // namespace

MatrixInput readInputFile(const std::filesystem::path& inputPath) {
    std::ifstream input(inputPath);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + inputPath.string());
    }

    std::size_t dimension = 0;
    if (!(input >> dimension)) {
        throw std::runtime_error("input file must start with the matrix dimension");
    }
    if (!isPowerOfTwo(dimension)) {
        throw std::runtime_error("matrix dimension must be a power of 2");
    }

    MatrixInput result;
    result.path = inputPath;
    result.stem = inputPath.stem().string();
    result.a = Matrix(dimension);
    result.b = Matrix(dimension);

    readMatrixValues(input, result.a);
    readMatrixValues(input, result.b);

    return result;
}

std::filesystem::path makeOutputPath(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    std::size_t dimension,
    const std::string& kind,
    const std::string& method) {
    const std::string filename = inputStem + "-" + std::to_string(dimension) + "-" + kind + "-" + method + ".txt";
    return outputDirectory / filename;
}

void writeMatrixFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& method,
    const Matrix& matrix) {
    const auto outputPath = makeOutputPath(outputDirectory, inputStem, matrix.size(), "output", method);
    std::ofstream output(outputPath);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + outputPath.string());
    }

    const std::size_t n = matrix.size();
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            if (column > 0) {
                output << ' ';
            }
            output << matrix(row, column);
        }
        output << '\n';
    }
}

void writeInfoFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    std::size_t dimension,
    const std::string& method,
    const std::string& formattedTime,
    int coresUsed) {
    const auto outputPath = makeOutputPath(outputDirectory, inputStem, dimension, "info", method);
    std::ofstream output(outputPath, std::ios::app);
    if (!output) {
        throw std::runtime_error("failed to open info file: " + outputPath.string());
    }

    output << formattedTime << '\n';
    output << coresUsed << '\n';
    output << "---\n";
}
