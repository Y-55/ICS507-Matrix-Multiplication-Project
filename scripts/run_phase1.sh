#!/usr/bin/env bash
# Phase 1: Fixed base case (16), vary dimension and thread count.
# Runs all 4 algorithms in order: Sequential, ParMtrixMult, Strassen, ParStrassen.

set -euo pipefail

BINARY="./matrix_mult"
TESTS_DIR="tests"
SCRIPT_DIR="scripts"
BASE_CASE=16
SEED=42
THREAD_COUNTS=(1 2 4)
DIMENSIONS=(32 64 128 256 512 1024 2048 4096)

if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY not found. Build first with:"
    echo "  g++-15 -std=c++17 -O2 -fopenmp src/main.cpp src/matrix.cpp src/io.cpp src/sequential.cpp src/par_matrix_mult.cpp src/strassen.cpp src/par_strassen.cpp -Isrc -o matrix_mult"
    exit 1
fi

# Generate any missing input files.
for dim in "${DIMENSIONS[@]}"; do
    input="$TESTS_DIR/sample_${dim}.txt"
    if [[ ! -f "$input" ]]; then
        echo "Generating $input ..."
        python3 "$SCRIPT_DIR/generate_input.py" "$dim" "$input" --seed "$SEED"
    fi
done

echo ""
echo "============================================================"
echo " Phase 1 — Algorithm Comparison (base case = $BASE_CASE)"
echo "============================================================"

for dim in "${DIMENSIONS[@]}"; do
    input="$TESTS_DIR/sample_${dim}.txt"

    echo ""
    echo "------------------------------------------------------------"
    echo " Dimension: ${dim}x${dim}"
    echo "------------------------------------------------------------"

    # Sequential and Strassen use 1 thread only — no benefit from more.
    echo "[Sequential]"
    "$BINARY" "$input" 1 "$BASE_CASE" Sequential

    echo "[Strassen]"
    "$BINARY" "$input" 1 "$BASE_CASE" Strassen

    # Parallel algorithms: run across all thread counts.
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "[ParMtrixMult  threads=$threads]"
        "$BINARY" "$input" "$threads" "$BASE_CASE" ParMtrixMult

        echo "[ParStrassen   threads=$threads]"
        "$BINARY" "$input" "$threads" "$BASE_CASE" ParStrassen
    done

done

echo ""
echo "============================================================"
echo " Phase 1 complete. Info files written to current directory."
echo "============================================================"
