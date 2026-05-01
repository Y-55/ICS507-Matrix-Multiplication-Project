# ICS 507 Matrix Multiplication Project

This directory documents how to build, run, and divide the Phase II work for the C++ OpenMP implementation.

## Commands

### Build

```sh
./scripts/build.sh
```

### Generate Input Files

Generate a default input file at `tests/input_<n>.txt`:

```sh
python3 scripts/generate_input.py 1024
```

Generate with custom output path and fixed seed:

```sh
python3 scripts/generate_input.py 1024 --output tests/input_custom_1024.txt --seed 507
```

Generate with custom value range:

```sh
python3 scripts/generate_input.py 1024 --min -5 --max 5
```

### Run Program

```sh
./build/matrix_mult <input-file> [threads] [strassen-base-case] [mode]
```

Argument order is required:

1. `input-file`
2. `threads` (default: max available threads)
3. `strassen-base-case` (default: `64`)
4. `mode` (default: `all`)

Valid `mode` values:

- `all`
- `Sequential`
- `ParMtrixMult`
- `Strassen`
- `ParStrassen`

Run all methods with explicit values:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 all
```

Run only Sequential:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 Sequential
```

Run only parallel direct multiplication:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 ParMtrixMult
```

Run only Strassen:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 Strassen
```

Run only parallel Strassen:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 ParStrassen
```

Use defaults for optional arguments:

```sh
./build/matrix_mult tests/input_1024.txt
```

Run one specific mode with an explicit base case:

```sh
./build/matrix_mult tests/input_1024.txt 8 64 ParMtrixMult
```

If you want to pass `mode`, you must also provide `threads` and `strassen-base-case` first because the CLI is positional.
 |
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
