#include "algorithms.hpp"

#include <stdexcept>

Matrix multiplySequential(const Matrix& a, const Matrix& b) { // ReturnType FunctionName(const Type& param1, const Type& param2)
    if (!sameDimensions(a, b)) {
        throw std::invalid_argument("matrices must have the same dimensions");
    }

    const std::size_t n = a.size(); // const + std::size_t (type from the std namespace) + variable name + initialization
    Matrix c(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            long sum = 0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }

    return c;
}
