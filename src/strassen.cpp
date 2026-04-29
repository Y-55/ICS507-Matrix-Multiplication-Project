#include "algorithms.hpp"

#include <stdexcept>

namespace {

// A view into a row-major matrix without copying: (row, col) maps into
// the full matrix using a stride equal to the original dimension.
struct MatrixView {
    const Matrix& mat;
    std::size_t rowOff;
    std::size_t colOff;
    std::size_t stride; // full matrix dimension (for index arithmetic)
    std::size_t size;   // this view's dimension

    long operator()(std::size_t r, std::size_t c) const {
        return mat.values()[(rowOff + r) * stride + (colOff + c)];
    }
};

void addViews(const MatrixView& a, const MatrixView& b, Matrix& out) {
    const std::size_t n = a.size;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out(i, j) = a(i, j) + b(i, j);
}

void subViews(const MatrixView& a, const MatrixView& b, Matrix& out) {
    const std::size_t n = a.size;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            out(i, j) = a(i, j) - b(i, j);
}

// Write result matrix r into a quadrant of dst.
void insertQuadrant(Matrix& dst, const Matrix& r, std::size_t rowOff, std::size_t colOff) {
    const std::size_t n = dst.size();
    const std::size_t half = r.size();
    for (std::size_t i = 0; i < half; ++i)
        for (std::size_t j = 0; j < half; ++j)
            dst.values()[(rowOff + i) * n + (colOff + j)] = r(i, j);
}

Matrix strassenRecursive(const MatrixView& a, const MatrixView& b, int baseCase);

// Compute a temp matrix from two views, then recurse.
Matrix strassenAdd(const MatrixView& va, const MatrixView& vb,
                   const MatrixView& wa, const MatrixView& wb, int baseCase) {
    const std::size_t half = va.size;
    Matrix ta(half), tb(half);
    addViews(va, vb, ta);
    addViews(wa, wb, tb);
    MatrixView vta{ta, 0, 0, half, half};
    MatrixView vtb{tb, 0, 0, half, half};
    return strassenRecursive(vta, vtb, baseCase);
}

Matrix strassenOp(const MatrixView& va, bool addA, const MatrixView& vb1, const MatrixView& vb2, bool addB, int baseCase) {
    const std::size_t half = va.size;
    Matrix tb(half);
    if (addB) addViews(vb1, vb2, tb); else subViews(vb1, vb2, tb);
    MatrixView vtb{tb, 0, 0, half, half};
    (void)addA;
    return strassenRecursive(va, vtb, baseCase);
}

Matrix strassenRecursive(const MatrixView& a, const MatrixView& b, int baseCase) {
    const std::size_t n = a.size;

    if (static_cast<int>(n) <= baseCase || n == 1) {
        Matrix result(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                long sum = 0;
                for (std::size_t k = 0; k < n; ++k)
                    sum += a(i, k) * b(k, j);
                result(i, j) = sum;
            }
        return result;
    }

    const std::size_t half = n / 2;

    // Quadrant views — no copying, just offset arithmetic.
    const MatrixView a11{a.mat, a.rowOff,        a.colOff,        a.stride, half};
    const MatrixView a12{a.mat, a.rowOff,        a.colOff + half, a.stride, half};
    const MatrixView a21{a.mat, a.rowOff + half, a.colOff,        a.stride, half};
    const MatrixView a22{a.mat, a.rowOff + half, a.colOff + half, a.stride, half};

    const MatrixView b11{b.mat, b.rowOff,        b.colOff,        b.stride, half};
    const MatrixView b12{b.mat, b.rowOff,        b.colOff + half, b.stride, half};
    const MatrixView b21{b.mat, b.rowOff + half, b.colOff,        b.stride, half};
    const MatrixView b22{b.mat, b.rowOff + half, b.colOff + half, b.stride, half};

    // Build temp matrices for operands that need add/sub, then recurse.
    Matrix ta(half), tb(half);

    addViews(a11, a22, ta); addViews(b11, b22, tb);
    const Matrix m1 = strassenRecursive({ta, 0, 0, half, half}, {tb, 0, 0, half, half}, baseCase);

    addViews(a21, a22, ta);
    const Matrix m2 = strassenRecursive({ta, 0, 0, half, half}, b11, baseCase);

    subViews(b12, b22, tb);
    const Matrix m3 = strassenRecursive(a11, {tb, 0, 0, half, half}, baseCase);

    subViews(b21, b11, tb);
    const Matrix m4 = strassenRecursive(a22, {tb, 0, 0, half, half}, baseCase);

    addViews(a11, a12, ta);
    const Matrix m5 = strassenRecursive({ta, 0, 0, half, half}, b22, baseCase);

    subViews(a21, a11, ta); addViews(b11, b12, tb);
    const Matrix m6 = strassenRecursive({ta, 0, 0, half, half}, {tb, 0, 0, half, half}, baseCase);

    subViews(a12, a22, ta); addViews(b21, b22, tb);
    const Matrix m7 = strassenRecursive({ta, 0, 0, half, half}, {tb, 0, 0, half, half}, baseCase);

    Matrix c(n);
    for (std::size_t i = 0; i < half; ++i) {
        for (std::size_t j = 0; j < half; ++j) {
            c(i,        j)        = m1(i,j) + m4(i,j) - m5(i,j) + m7(i,j); // C11
            c(i,        j + half) = m3(i,j) + m5(i,j);                       // C12
            c(i + half, j)        = m2(i,j) + m4(i,j);                       // C21
            c(i + half, j + half) = m1(i,j) - m2(i,j) + m3(i,j) + m6(i,j); // C22
        }
    }
    return c;
}

} // namespace

Matrix multiplyStrassen(const Matrix& a, const Matrix& b, int baseCase) {
    if (!sameDimensions(a, b)) {
        throw std::invalid_argument("matrices must have the same dimensions");
    }
    if (baseCase <= 0) {
        throw std::invalid_argument("base case must be positive");
    }

    const std::size_t n = a.size();
    MatrixView va{a, 0, 0, n, n};
    MatrixView vb{b, 0, 0, n, n};
    return strassenRecursive(va, vb, baseCase);
}
