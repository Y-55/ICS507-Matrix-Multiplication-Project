#include "algorithms.hpp"

#include <stdexcept>

#if !defined(_OPENMP) && !defined(__clangd__) && !defined(__INTELLISENSE__)
#error "OpenMP is required to build this project. Rebuild with OpenMP enabled (e.g., -fopenmp)."
#endif

long dotproduct(const long i[], const long j[], std::size_t size) {
    if (size == 0) {
        return 0;
    }

    long sum = 0;
    // We intentionally use OpenMP reduction instead of the textbook binary-tree array method.
    // Why:
    // 1) The book's algorithm assumes n processors and n = 2^k, then minimizes span with an explicit
    //    tree over temporary arrays W and V. On real CPUs we usually have far fewer threads than n.
    // 2) With limited threads, creating/copying W and V and launching multiple parallel regions adds
    //    extra memory traffic and synchronization overhead that often outweighs any benefit.
    // 3) reduction(+:sum) keeps each thread's partial sum private (no data races), then combines
    //    partials efficiently at the end. Most OpenMP runtimes already use tree-like combining
    //    internally, so we still get a good reduction strategy without manual index math.
    // 4) This version is correct for any size (not only powers of two), simpler to read, and easier
    //    for compilers to optimize.
    //
    // Pragma options used here:
    // - simd: vectorize the loop to use SIMD lanes.
    // - reduction(+:sum): each SIMD lane accumulates privately, then lanes are combined safely.
    // We keep this helper SIMD-only and parallelize at the (i, j) matrix level to avoid creating
    // a new thread team for every single output cell.
    #pragma omp simd reduction(+:sum)
    for (std::size_t l = 0; l < size; ++l) {
        sum += i[l] * j[l];
    }

    return sum;
}

Matrix multiplyParMatrixMult(const Matrix& a, const Matrix& b, int threads) {
    if (!sameDimensions(a, b)) {
        throw std::invalid_argument("matrices must have the same dimensions");
    }
    if (threads <= 0) {
        throw std::invalid_argument("threads must be positive");
    }

    const std::size_t n = a.size();
    Matrix c(n);
    Matrix bt(n);

    // Build B^T once so each original column of B becomes a contiguous row in bt.
    // This lets dotproduct read contiguous memory from both inputs.
    // OpenMP options used here:
    // - parallel for: run loop iterations on multiple threads.
    // - collapse(2): flatten (row, col) into one large iteration space for better load balance.
    // - schedule(static): pre-assign equal chunks because each transpose write has uniform cost.
    // - num_threads(threads): use the caller-requested thread count for this kernel.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            bt(col, row) = b(row, col);
        }
    }

    // Same pragma choices for computing C: each (i, j) cell has similar work, so static scheduling
    // with a collapsed 2D loop gives low overhead and good distribution across threads.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            // Row i of A starts at i * n. Row j of bt equals column j of original B.
            c(i, j) = dotproduct(a.values().data() + i * n, bt.values().data() + j * n, n);
        }
    }

    return c;
}
