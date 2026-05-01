#pragma once

#include "matrix.hpp"

#include <filesystem>
#include <string>
#include <vector>

struct MatrixInput {
    std::filesystem::path path;
    std::string stem;
    Matrix a;
    Matrix b;
};

MatrixInput readInputFile(const std::filesystem::path& inputPath);

std::filesystem::path makeOutputPath(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& kind,
    const std::string& method,
    const std::string& parameterTag = "");

void writeMatrixFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& method,
    const std::string& parameterTag,
    const Matrix& matrix);

void writeInfoFile(
    const std::filesystem::path& outputDirectory,
    const std::string& inputStem,
    const std::string& method,
    const std::string& formattedTime,
    int coresUsed,
    const std::string& parameterTag,
    const std::vector<std::string>& metadataLines);
