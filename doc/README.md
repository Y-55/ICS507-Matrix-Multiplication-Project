# ICS 507 Matrix Multiplication Project

This directory documents how to build, run, and divide the Phase II work for the C++ OpenMP implementation.

## Build

From the repository root:

```sh
./scripts/build.sh
```

The executable is created at `build/matrix_mult`.

## Run

```sh
./build/matrix_mult <input-file> [threads] [strassen-base-case] [mode]
```

Arguments:

- `input-file`: text file containing `n`, then matrix `A`, then matrix `B`.
- `threads`: OpenMP thread count. Defaults to the maximum available thread count.
- `strassen-base-case`: dimension threshold for switching Strassen to the baseline multiplication. Defaults to `64`.
- `mode`: one of `all`, `Sequential`, `ParMtrixMult`, `Strassen`, or `ParStrassen`. Defaults to `all`.

Example:

```sh
./build/matrix_mult tests/sample_2.txt 4 2 all
```

## Input Format

The first line is the matrix dimension `n`. The project assumes `n` is a power of 2. The next `2n^2` values are long integers in row-major order:

```text
n
A values
B values
```

## Output Files

For an input file named `input1.txt` with `n = 128`, the program writes:

- `input1-128-output-Sequential.txt`
- `input1-128-info-Sequential.txt`
- `input1-128-output-ParMtrixMult.txt`
- `input1-128-info-ParMtrixMult.txt`
- `input1-128-output-Strassen.txt`
- `input1-128-info-Strassen.txt`
- `input1-128-output-ParStrassen.txt`
- `input1-128-info-ParStrassen.txt`

Each output file contains the resulting matrix. Each info file contains the multiplication-only elapsed time in `hh:mm:ss` format followed by the number of cores used.

## Contributor Ownership

| Contributor | Responsibility | Main Files |
| --- | --- | --- |
| 1 | Sequential and ParMatrixMult implementation | `src/sequential.cpp`, `src/par_matrix_mult.cpp` |
| 2 | Sequential and parallel Strassen implementation | `src/strassen.cpp`, `src/par_strassen.cpp` |
| 3 | Documentation, experiments, report, and presentation notes | `doc/README.md`, report tables, experiment logs |

Shared files should only change after coordination:

- `src/matrix.hpp`
- `src/matrix.cpp`
- `src/io.hpp`
- `src/io.cpp`
- `src/algorithms.hpp`
- `src/main.cpp`

## Algorithm Contract

Both implementation contributors should preserve these signatures:

```cpp
Matrix multiplySequential(const Matrix& a, const Matrix& b);
Matrix multiplyParMatrixMult(const Matrix& a, const Matrix& b, int threads);
Matrix multiplyStrassen(const Matrix& a, const Matrix& b, int baseCase);
Matrix multiplyParStrassen(const Matrix& a, const Matrix& b, int baseCase, int threads);
```

The runner compares non-sequential results against the sequential result during development and prints a warning if an implementation differs.

## Experiment Checklist

Record the following for Phase II reporting:

- Machine specifications: CPU, core count, RAM, cache information, and OS.
- Matrix sizes tested, such as `128`, `256`, `512`, `1024`, and larger if possible.
- Thread counts tested, such as `1`, `2`, `4`, `8`, and the machine maximum.
- Strassen base cases tested, such as `16`, `32`, `64`, and `128`.
- Runtime table for all four methods.
- Correctness status compared with the sequential baseline.
- Observed differences between theoretical and actual performance.
