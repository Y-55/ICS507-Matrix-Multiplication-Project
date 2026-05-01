#!/usr/bin/env bash
# Phase 1 (Linux): Fixed base case (4), vary dimension and thread count.
# Runs all 4 algorithms in order: Sequential, Strassen, ParMtrixMult, ParStrassen.
# Run with: bash scripts/run_phase1_linux.sh

set -euo pipefail

BINARY="./matrix_mult"
TESTS_DIR="tests"
SCRIPT_DIR="scripts"
BASE_CASE=4
SEED=42
THREAD_COUNTS=(1 2 4 6 8 10 12 14 16)
DIMENSIONS=(32 64 128 256 512 1024 2048 4096)

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
echo " Phase 1 (Linux) — base case=$BASE_CASE   threads up to 16"
echo "============================================================"

for dim in "${DIMENSIONS[@]}"; do
    input="$TESTS_DIR/sample_${dim}.txt"

    echo ""
    echo "------------------------------------------------------------"
    echo " Dimension: ${dim}x${dim}"
    echo "------------------------------------------------------------"

    echo "[Sequential  threads=1]"
    "$BINARY" "$input" 1 "$BASE_CASE" Sequential

    echo "[Strassen    threads=1]"
    "$BINARY" "$input" 1 "$BASE_CASE" Strassen

    for threads in "${THREAD_COUNTS[@]}"; do
        echo "[ParMtrixMult  threads=$threads]"
        "$BINARY" "$input" "$threads" "$BASE_CASE" ParMtrixMult

        echo "[ParStrassen   threads=$threads]"
        "$BINARY" "$input" "$threads" "$BASE_CASE" ParStrassen
    done

done

echo ""
echo "============================================================"
echo " Phase 1 (Linux) complete."
echo "============================================================"
