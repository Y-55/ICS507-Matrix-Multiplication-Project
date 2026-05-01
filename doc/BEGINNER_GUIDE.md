# Beginner Guide and Contribution Plan

This guide explains the project setup for a developer who knows Python and is new to C++.

The goal of Phase II is to implement and test four matrix multiplication methods:

1. `Sequential`
2. `ParMtrixMult`
3. `Strassen`
4. `ParStrassen`

The project is organized so three people can work without stepping on each other's files.

## Big Picture

The executable reads one input text file containing two matrices. It runs one or more multiplication methods, writes the resulting matrix to output files, and writes timing information to info files.

Example command:

```sh
./build/matrix_mult tests/sample_2.txt 4 2 all
```

This means:

- `tests/sample_2.txt`: input file.
- `4`: use 4 OpenMP threads for parallel methods.
- `2`: use base case size 2 for Strassen methods.
- `all`: run all four methods.

## Setup

Install these tools:

- A C++ compiler, such as `clang++` or `g++`.
- OpenMP.
- Python 3 for generating test input files.

Build with the project script:

```sh
./scripts/build.sh
```

Run:

```sh
./build/matrix_mult tests/sample_2.txt 4 2 all
```

The build script handles OpenMP flags and outputs `build/matrix_mult`.

## Python vs C++ Basics

In Python, one `.py` file can define functions and run code directly. In C++, projects are usually split into two kinds of files:

- `.hpp` files are header files. They describe what classes and functions exist. Think of them like a public API or function signatures.
- `.cpp` files are source files. They contain the real implementation of those functions.

Example:

```cpp
// In a .hpp file: declaration
Matrix multiplySequential(const Matrix& a, const Matrix& b);
```

```cpp
// In a .cpp file: implementation
Matrix multiplySequential(const Matrix& a, const Matrix& b) {
    // actual code here
}
```

The reason for this split is that multiple `.cpp` files can include the same `.hpp` file and agree on the same function contract.

## Files Created

### Build Script

- `scripts/build.sh`
  - Compiles the project using C++17.
  - Adds OpenMP flags.
  - Builds the executable named `matrix_mult` under `build/`.

### Shared C++ Files

- `src/matrix.hpp`
  - Declares the `Matrix` class.
  - Declares helper functions like `matricesEqual` and `isPowerOfTwo`.

- `src/matrix.cpp`
  - Implements the `Matrix` class.
  - Stores matrix values in a one-dimensional `std::vector<long>`.
  - Provides safe access using `matrix(row, column)`.

- `src/io.hpp`
  - Declares input and output helper functions.
  - Defines `MatrixInput`, which holds the input file path, file name, matrix `A`, and matrix `B`.

- `src/io.cpp`
  - Reads the assignment input format.
  - Writes result matrix files.
  - Writes timing info files.
  - Creates file names like `input1-128-output-Sequential.txt`.

- `src/timer.hpp`
  - Defines a small timer helper using `std::chrono`.
  - Formats elapsed time as `hh:mm:ss`.

- `src/algorithms.hpp`
  - Declares the four algorithm functions.
  - This is the most important coordination file between Contributor 1 and Contributor 2.

- `src/main.cpp`
  - Parses command-line arguments.
  - Reads the input file.
  - Runs the requested algorithm or all algorithms.
  - Measures only multiplication time.
  - Writes all required output and info files.
  - Compares non-sequential results with the sequential result and prints a warning if they differ.

### Algorithm Files

- `src/sequential.cpp`
  - Implements the basic triple-loop sequential matrix multiplication.
  - This is the correctness baseline.

- `src/par_matrix_mult.cpp`
  - Implements the parallel version of the basic triple-loop method using OpenMP.
  - This is where Contributor 1 should continue improving or tuning `ParMtrixMult`.

- `src/strassen.cpp`
  - Currently delegates to sequential multiplication.
  - Contributor 2 should replace this with sequential Strassen recursion.

- `src/par_strassen.cpp`
  - Currently delegates to sequential multiplication.
  - Contributor 2 should replace this with parallel Strassen using OpenMP tasks or another OpenMP strategy.

### Documentation and Tests

- `doc/README.md`
  - Short build/run reference.
  - Output file naming rules.
  - Experiment checklist.

- `doc/BEGINNER_GUIDE.md`
  - This file.
  - Explains the project for beginner contributors.

- `tests/sample_2.txt`
  - Small 2 by 2 input file for quick testing.

- `scripts/generate_input.py`
  - Python script for generating random matrix input files in the required assignment format.

## How to Contribute

Before changing code:

1. Pull the latest code from the repository.
2. Build the project.
3. Run the sample input and confirm it works.
4. Edit only the files assigned to you unless the team agrees otherwise.
5. Run the sample again before pushing your work.

Recommended test command:

```sh
./build/matrix_mult tests/sample_2.txt 4 2 all
```

For larger tests, generate input files with:

```sh
python3 scripts/generate_input.py 128 tests/input_128.txt --seed 507
```

Then run:

```sh
./build/matrix_mult tests/input_128.txt 4 64 all
```

## Contributor Responsibilities

### Contributor 1: Sequential and ParMtrixMult

Main files:

- `src/sequential.cpp`
- `src/par_matrix_mult.cpp`

Tasks:

- Keep sequential multiplication correct and simple.
- Improve `ParMtrixMult` using the textbook parallel matrix multiplication algorithm.
- Test several thread counts.
- Report performance results to Contributor 3.

### Contributor 2: Strassen and ParStrassen

Main files:

- `src/strassen.cpp`
- `src/par_strassen.cpp`

Tasks:

- Implement sequential Strassen in `src/strassen.cpp`.
- Implement parallel Strassen in `src/par_strassen.cpp`.
- Use the `baseCase` argument to decide when recursion stops.
- Compare results against `Sequential`.
- Report performance results to Contributor 3.

### Contributor 3: Documentation and Report

Main files:

- `doc/README.md`
- `doc/BEGINNER_GUIDE.md`
- Report and presentation files created later.

Tasks:

- Keep build and run instructions updated.
- Record who did what.
- Record machine specifications.
- Prepare experiment tables.
- Write the Phase II report sections.
- Prepare Phase III presentation material.

## Shared Files: Be Careful

These files affect everyone:

- `src/matrix.hpp`
- `src/matrix.cpp`
- `src/io.hpp`
- `src/io.cpp`
- `src/algorithms.hpp`
- `src/main.cpp`

Only edit these after discussing with the group. If the function signatures in `src/algorithms.hpp` change, both algorithm contributors may need to update their code.

## What Is Left

The skeleton compiles and runs, but the full project is not finished yet.

Still left to do:

- Implement real sequential Strassen in `src/strassen.cpp`.
- Implement real parallel Strassen in `src/par_strassen.cpp`.
- Confirm `ParMtrixMult` matches the exact textbook algorithm expected by the instructor.
- Run experiments for many matrix sizes.
- Run experiments for many thread counts.
- Run experiments for several Strassen base-case values.
- Record machine specs.
- Fill in the report tables.
- Analyze theoretical running time for both parallel algorithms.
- Explain differences between theory and experiment.
- Prepare the final presentation.

## Suggested Workflow

Use this loop while developing:

1. Make a small code change.
2. Build.
3. Run `tests/sample_2.txt`.
4. Generate a bigger input file.
5. Run all four algorithms.
6. Check for result mismatch warnings.
7. Save timing results for the report.

## Important Notes

- The assignment requires matrix values to use long integers. In C++, this project uses `long`.
- The assignment says input dimensions are powers of 2.
- `ParMtrixMult` is intentionally spelled like the assignment output file name.
- Generated output and info files are ignored by Git using `.gitignore`.
- The current Strassen files are placeholders. They are correct for output because they call sequential multiplication, but they do not yet implement the required Strassen algorithms.
