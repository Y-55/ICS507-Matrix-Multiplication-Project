#include "io.hpp"

#include <array>
#include <cstdio>
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
    const std::string& kind,
    const std::string& method,
    const std::string& parameterTag) {
    std::string filename = inputStem + "-" + kind + "-" + method;
    if (!parameterTag.empty()) {
        filename += "-" + parameterTag;
    }
    filename += ".txt";
    return outputDirectory / filename;
}

void writeMatrixFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& method,
    const std::string& parameterTag,
    const Matrix& matrix) {
    const auto outputPath = makeOutputPath(outputDirectory, inputStem, "output", method, parameterTag);
    if (std::filesystem::exists(outputPath)) {
        return;
    }
    std::filebuf fileBuffer;
    std::array<char, 1 << 22> streamBuffer{};
    fileBuffer.pubsetbuf(streamBuffer.data(), static_cast<std::streamsize>(streamBuffer.size()));
    if (!fileBuffer.open(outputPath, std::ios::out | std::ios::trunc)) {
        throw std::runtime_error("failed to open output file: " + outputPath.string());
    }
    std::ostream output(&fileBuffer);

    const std::size_t n = matrix.size();
    const auto& values = matrix.values();

    std::string chunk;
    chunk.reserve(1 << 22);

    for (std::size_t row = 0; row < n; ++row) {
        const std::size_t rowOffset = row * n;
        for (std::size_t column = 0; column < n; ++column) {
            if (column > 0) {
                chunk.push_back(' ');
            }

            char numberBuffer[32];
            const long value = values[rowOffset + column];
            const int printed = std::snprintf(numberBuffer, sizeof(numberBuffer), "%ld", value);
            if (printed < 0 || printed >= static_cast<int>(sizeof(numberBuffer))) {
                throw std::runtime_error("failed to format matrix value for output");
            }
            chunk.append(numberBuffer, static_cast<std::size_t>(printed));
        }
        chunk.push_back('\n');

        if (chunk.size() >= (1 << 22)) {
            output.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            chunk.clear();
        }
    }

    if (!chunk.empty()) {
        output.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }
}

void writeInfoFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& method,
    const std::string& formattedTime,
    int coresUsed,
    const std::string& parameterTag,
    const std::vector<std::string>& metadataLines) {
    (void)parameterTag;
    const auto outputPath = makeOutputPath(outputDirectory, inputStem, "info", method, "");
    std::ofstream output(outputPath, std::ios::app);
    if (!output) {
        throw std::runtime_error("failed to open info file: " + outputPath.string());
    }

    output << formattedTime << '\n';
    output << coresUsed << '\n';
    for (const std::string& metadataLine : metadataLines) {
        output << metadataLine << '\n';
    }
    output << "---\n";
}
