#include "algorithms.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

Matrix add(const Matrix& a, const Matrix& b) {
    const std::size_t n = a.size();
    Matrix result(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            result(i, j) = a(i, j) + b(i, j);
    return result;
}

Matrix sub(const Matrix& a, const Matrix& b) {
    const std::size_t n = a.size();
    Matrix result(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            result(i, j) = a(i, j) - b(i, j);
    return result;
}

Matrix extractQuadrant(const Matrix& src, std::size_t rowOff, std::size_t colOff) {
    const std::size_t half = src.size() / 2;
    Matrix result(half);
    for (std::size_t i = 0; i < half; ++i)
        for (std::size_t j = 0; j < half; ++j)
            result(i, j) = src(rowOff + i, colOff + j);
    return result;
}

Matrix parStrassenRecursive(const Matrix& a, const Matrix& b, int baseCase, int depth) {
    const std::size_t n = a.size();

    if (n == 1) {
        Matrix result(1);
        result(0, 0) = a(0, 0) * b(0, 0);
        return result;
    }

    if (static_cast<int>(n) <= baseCase) {
        return multiplySequential(a, b);
    }

    const std::size_t half = n / 2;

    const Matrix a11 = extractQuadrant(a, 0,    0);
    const Matrix a12 = extractQuadrant(a, 0,    half);
    const Matrix a21 = extractQuadrant(a, half, 0);
    const Matrix a22 = extractQuadrant(a, half, half);

    const Matrix b11 = extractQuadrant(b, 0,    0);
    const Matrix b12 = extractQuadrant(b, 0,    half);
    const Matrix b21 = extractQuadrant(b, half, 0);
    const Matrix b22 = extractQuadrant(b, half, half);

    const Matrix s1 = add(a11, a22);  const Matrix t1 = add(b11, b22);
    const Matrix s2 = add(a21, a22);
    const Matrix t3 = sub(b12, b22);
    const Matrix t4 = sub(b21, b11);
    const Matrix s5 = add(a11, a12);
    const Matrix s6 = sub(a21, a11);  const Matrix t6 = add(b11, b12);
    const Matrix s7 = sub(a12, a22);  const Matrix t7 = add(b21, b22);

    Matrix m1, m2, m3, m4, m5, m6, m7;

#ifdef _OPENMP
    if (depth > 0) {
        const int nextDepth = depth - 1;

        #pragma omp task shared(m1)
        m1 = parStrassenRecursive(s1,  t1,  baseCase, nextDepth);
        #pragma omp task shared(m2)
        m2 = parStrassenRecursive(s2,  b11, baseCase, nextDepth);
        #pragma omp task shared(m3)
        m3 = parStrassenRecursive(a11, t3,  baseCase, nextDepth);
        #pragma omp task shared(m4)
        m4 = parStrassenRecursive(a22, t4,  baseCase, nextDepth);
        #pragma omp task shared(m5)
        m5 = parStrassenRecursive(s5,  b22, baseCase, nextDepth);
        #pragma omp task shared(m6)
        m6 = parStrassenRecursive(s6,  t6,  baseCase, nextDepth);
        #pragma omp task shared(m7)
        m7 = parStrassenRecursive(s7,  t7,  baseCase, nextDepth);

        #pragma omp taskwait
    } else {
#endif
        m1 = parStrassenRecursive(s1,  t1,  baseCase, 0);
        m2 = parStrassenRecursive(s2,  b11, baseCase, 0);
        m3 = parStrassenRecursive(a11, t3,  baseCase, 0);
        m4 = parStrassenRecursive(a22, t4,  baseCase, 0);
        m5 = parStrassenRecursive(s5,  b22, baseCase, 0);
        m6 = parStrassenRecursive(s6,  t6,  baseCase, 0);
        m7 = parStrassenRecursive(s7,  t7,  baseCase, 0);
#ifdef _OPENMP
    }
#endif

    Matrix c(n);
    for (std::size_t i = 0; i < half; ++i) {
        for (std::size_t j = 0; j < half; ++j) {
            c(i,        j)        = m1(i,j) + m4(i,j) - m5(i,j) + m7(i,j);
            c(i,        j + half) = m3(i,j) + m5(i,j);
            c(i + half, j)        = m2(i,j) + m4(i,j);
            c(i + half, j + half) = m1(i,j) - m2(i,j) + m3(i,j) + m6(i,j);
        }
    }
    return c;
}

} // namespace

Matrix multiplyParStrassen(const Matrix& a, const Matrix& b, int baseCase, int threads) {
    if (!sameDimensions(a, b)) {
        throw std::invalid_argument("matrices must have the same dimensions");
    }
    if (baseCase <= 0) {
        throw std::invalid_argument("base case must be positive");
    }
    if (threads <= 0) {
        throw std::invalid_argument("threads must be positive");
    }

    // depth d spawns 7^d tasks; stop when task count >= threads.
    int depth = 0;
    long long taskCount = 1;
    while (taskCount < static_cast<long long>(threads)) {
        taskCount *= 7;
        ++depth;
    }

    Matrix result;
#ifdef _OPENMP
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        result = parStrassenRecursive(a, b, baseCase, depth);
    }
#else
    result = parStrassenRecursive(a, b, baseCase, 0);
#endif
    return result;
}
