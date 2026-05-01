# Phase 1 (Windows): Fixed base case (4), vary dimension and thread count.
# Runs all 4 algorithms in order: Sequential, Strassen, ParMtrixMult, ParStrassen.
# Run with: .\scripts\run_phase1_windows.ps1

$ErrorActionPreference = "Stop"

$BINARY        = ".\matrix_mult.exe"
$TESTS_DIR     = "tests"
$SCRIPT_DIR    = "scripts"
$BASE_CASE     = 4
$SEED          = 42
$THREAD_COUNTS = @(1, 2, 4, 6, 8, 10, 12, 14, 16)
$DIMENSIONS    = @(32, 64, 128, 256, 512, 1024, 2048, 4096)

if (-not (Test-Path $BINARY)) {
    Write-Error "Error: $BINARY not found. Build first with:`n  g++ -std=c++17 -O2 -fopenmp src/main.cpp src/matrix.cpp src/io.cpp src/sequential.cpp src/par_matrix_mult.cpp src/strassen.cpp src/par_strassen.cpp -Isrc -o matrix_mult"
    exit 1
}

foreach ($dim in $DIMENSIONS) {
    $input = "$TESTS_DIR\sample_$dim.txt"
    if (-not (Test-Path $input)) {
        Write-Host "Generating $input ..."
        python3 "$SCRIPT_DIR\generate_input.py" $dim $input --seed $SEED
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 1 (Windows) — base case=$BASE_CASE   threads up to 16"
Write-Host "============================================================"

foreach ($dim in $DIMENSIONS) {
    $input = "$TESTS_DIR\sample_$dim.txt"

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host " Dimension: ${dim}x${dim}"
    Write-Host "------------------------------------------------------------"

    Write-Host "[Sequential  threads=1]"
    & $BINARY $input 1 $BASE_CASE Sequential

    Write-Host "[Strassen    threads=1]"
    & $BINARY $input 1 $BASE_CASE Strassen

    foreach ($threads in $THREAD_COUNTS) {
        Write-Host "[ParMtrixMult  threads=$threads]"
        & $BINARY $input $threads $BASE_CASE ParMtrixMult

        Write-Host "[ParStrassen   threads=$threads]"
        & $BINARY $input $threads $BASE_CASE ParStrassen
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 1 (Windows) complete."
Write-Host "============================================================"
