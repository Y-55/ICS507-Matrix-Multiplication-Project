#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUTPUT="${BUILD_DIR}/matrix_mult"

COMPILER="${CXX:-g++}"
CXX_STANDARD="-std=c++17"
COMMON_FLAGS="-Wall -Wextra -pedantic -Isrc"
OPENMP_COMPILE_FLAGS=""
OPENMP_LINK_FLAGS=""
COMPILER_VERSION="$("${COMPILER}" --version 2>/dev/null || true)"
COMPILER_VERSION_LOWER="$(printf '%s' "${COMPILER_VERSION}" | tr '[:upper:]' '[:lower:]')"

if [[ -z "${COMPILER_VERSION}" ]]; then
  echo "Error: unable to detect compiler '${COMPILER}'." >&2
  exit 1
fi

if [[ "${COMPILER_VERSION_LOWER}" == *"clang"* ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
    else
      LIBOMP_PREFIX=""
    fi

    if [[ -n "${LIBOMP_PREFIX}" ]]; then
      OPENMP_COMPILE_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
      OPENMP_LINK_FLAGS="-L${LIBOMP_PREFIX}/lib -lomp"
    else
      echo "Error: OpenMP runtime not found for Apple Clang." >&2
      echo "Install it with: brew install libomp" >&2
      exit 1
    fi
  else
    OPENMP_COMPILE_FLAGS="-fopenmp"
    OPENMP_LINK_FLAGS="-fopenmp"
  fi
else
  OPENMP_COMPILE_FLAGS="-fopenmp"
  OPENMP_LINK_FLAGS="-fopenmp"
fi

mkdir -p "${BUILD_DIR}"

echo "Building with compiler: ${COMPILER}"
"${COMPILER}" ${CXX_STANDARD} ${COMMON_FLAGS} ${OPENMP_COMPILE_FLAGS} \
  src/main.cpp src/matrix.cpp src/io.cpp src/sequential.cpp \
  src/par_matrix_mult.cpp src/strassen.cpp src/par_strassen.cpp \
  ${OPENMP_LINK_FLAGS} -o "${OUTPUT}"

echo "Build complete: ${OUTPUT}"
