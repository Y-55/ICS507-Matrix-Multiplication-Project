# Phase 2 (Windows): Vary base case, fix dimension and thread count.
# Only runs Strassen and ParStrassen since base case does not affect the other two.
# Run with: .\scripts\run_phase2_windows.ps1

$ErrorActionPreference = "Stop"

$BINARY     = ".\matrix_mult.exe"
$TESTS_DIR  = "tests"
$SCRIPT_DIR = "scripts"
$SEED       = 42
$BASE_CASES = @(2, 4, 8, 16, 32, 64, 128, 256)
$DIMENSIONS = @(2048, 4096)
$THREADS    = 16

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
Write-Host " Phase 2 (Windows) — Base Case Tuning  threads=$THREADS"
Write-Host "============================================================"

foreach ($dim in $DIMENSIONS) {
    $input = "$TESTS_DIR\sample_$dim.txt"

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host " Dimension: ${dim}x${dim}"
    Write-Host "------------------------------------------------------------"

    foreach ($base in $BASE_CASES) {
        Write-Host "[Strassen    base=$base  threads=1]"
        & $BINARY $input 1 $base Strassen

        Write-Host "[ParStrassen base=$base  threads=$THREADS]"
        & $BINARY $input $THREADS $base ParStrassen
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 2 (Windows) complete."
Write-Host "============================================================"
