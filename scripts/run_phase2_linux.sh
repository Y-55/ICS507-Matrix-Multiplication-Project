#!/usr/bin/env bash
# Phase 2 (Linux): Vary base case, fix dimension and thread count.
# Only runs Strassen and ParStrassen since base case does not affect the other two.

set -euo pipefail

BINARY="./matrix_mult"
TESTS_DIR="tests"
SCRIPT_DIR="scripts"
SEED=42
BASE_CASES=(2 4 8 16 32 64 128 256)
DIMENSIONS=(2048 4096)
THREADS=16

if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY not found. Build first with:"
    echo "  g++ -std=c++17 -O2 -fopenmp src/main.cpp src/matrix.cpp src/io.cpp src/sequential.cpp src/par_matrix_mult.cpp src/strassen.cpp src/par_strassen.cpp -Isrc -o matrix_mult"
    exit 1
fi

for dim in "${DIMENSIONS[@]}"; do
    input="$TESTS_DIR/sample_${dim}.txt"
    if [[ ! -f "$input" ]]; then
        echo "Generating $input ..."
        python3 "$SCRIPT_DIR/generate_input.py" "$dim" "$input" --seed "$SEED"
    fi
done

echo ""
echo "============================================================"
echo " Phase 2 (Linux) — Base Case Tuning  threads=$THREADS"
echo "============================================================"

for dim in "${DIMENSIONS[@]}"; do
    input="$TESTS_DIR/sample_${dim}.txt"

    echo ""
    echo "------------------------------------------------------------"
    echo " Dimension: ${dim}x${dim}"
    echo "------------------------------------------------------------"

    for base in "${BASE_CASES[@]}"; do
        echo "[Strassen    base=$base  threads=1]"
        "$BINARY" "$input" 1 "$base" Strassen

        echo "[ParStrassen base=$base  threads=$THREADS]"
        "$BINARY" "$input" "$THREADS" "$base" ParStrassen
    done

done

echo ""
echo "============================================================"
echo " Phase 2 (Linux) complete."
echo "============================================================"
