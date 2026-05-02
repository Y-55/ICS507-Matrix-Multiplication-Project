#!/usr/bin/env python3
"""
This is an automation script for running the experiments in this repo on a linux machine. It handles input generation, execution, data collection, and plotting.This script wraps the compiled C++ OpenMP executable. It generates format inputs, runs the required algorithms across multiple configurations,
collects timing data into CSV files, and produces report-ready PNG plots.
Generated with help of Chatgpt to help in experimentation and logging.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

try:
    from generate_input import is_power_of_two, write_matrix_rows
except ModuleNotFoundError:
    from .generate_input import is_power_of_two, write_matrix_rows


ALGORITHMS = ("Sequential", "ParMtrixMult", "Strassen", "ParStrassen")
DEFAULT_BASE_CASE = 64
ALGORITHM_CONFIGS = {
    "Sequential": {
        "enabled": True,
        "matrix_exponents": [7, 8, 9, 10],
        "threads": [1],
        "base_cases": [DEFAULT_BASE_CASE],
        "repetitions": 1,
    },
    "ParMtrixMult": {
        "enabled": True,
        "matrix_exponents": [7, 8, 9, 10, 11, 12, 13, 14],
        "threads": [2, 4, 8,16 , 18 , 20, 24, 28, 32],
        "base_cases": [DEFAULT_BASE_CASE],
        "repetitions": 1,
    },
    "Strassen": {
        "enabled": True,
        "matrix_exponents": [7, 8, 9, 10],
        "threads": [1],
        "base_cases": [16, 32, 64, 128],
        "repetitions": 1,
    },
    "ParStrassen": {
        "enabled": True,
        "matrix_exponents": [7, 8, 9, 10, 11, 12, 13, 14],
        "threads": [2, 4, 8,16 , 18 , 20, 24, 28, 32],
        "base_cases": [16, 32, 64, 128],
        "repetitions": 1,
    },
}
PNG_WIDTH = 1280
PNG_HEIGHT = 720
PLOT_MARGIN_LEFT = 88
PLOT_MARGIN_RIGHT = 36
PLOT_MARGIN_TOP = 56
PLOT_MARGIN_BOTTOM = 84
SERIES_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


@dataclass
class RunRecord:
    algorithm: str
    matrix_size: int
    threads: int
    base_case: int
    run_index: int
    seed: int
    status: str
    runtime_seconds: float | None
    formatted_time: str | None
    reported_cores: int | None
    input_file: str
    output_matrix_file: str
    info_file: str
    stdout_log: str
    stderr_log: str
    mismatch_warning: bool
    timed_out: bool


@dataclass
class ToolStatus:
    name: str
    found: bool
    detail: str


@dataclass
class OsInfo:
    system: str
    distro_id: str | None
    distro_like: str | None
    pretty_name: str


@dataclass
class AlgorithmRunConfig:
    name: str
    enabled: bool
    matrix_sizes: list[int]
    threads: list[int]
    base_cases: list[int]
    repetitions: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Question 2 experiments for the ICS 507 matrix multiplication project."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="directory that contains the compiled matrix_mult executable",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=None,
        help="explicit path to the compiled matrix_mult executable",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="skip automatic configure/build and use an existing executable",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/inputs"),
        help="directory for generated assignment-format input files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="directory for CSV files, logs, and summaries",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="directory for generated PNG plots",
    )
    parser.add_argument(
        "--min-exp",
        type=int,
        default=None,
        help="smallest matrix size exponent to test; actual size is 2^k",
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=None,
        help="largest matrix size exponent to test; actual size is 2^k",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default=None,
        help="comma-separated thread counts; default is powers of two up to the machine limit",
    )
    parser.add_argument(
        "--base-cases",
        type=str,
        default=None,
        help="comma-separated base cases; overrides per-algorithm base-case lists",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="number of runs per configuration; overrides per-algorithm repetitions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=507,
        help="base random seed used to generate deterministic experiment inputs",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=-9,
        help="minimum random matrix entry",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=9,
        help="maximum random matrix entry",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="per-run timeout in seconds",
    )
    parser.add_argument(
        "--comparison-base-case",
        type=int,
        default=DEFAULT_BASE_CASE,
        help="default base case used in overall comparison plots",
    )
    parser.add_argument(
        "--parallel-only-from-exp",
        type=int,
        default=None,
        help=(
            "for matrix sizes 2^k with k greater than or equal to this value, "
            "skip Sequential and Strassen and run only the parallel algorithms"
        ),
    )
    parser.add_argument(
        "--keep-going-after-size-failure",
        action="store_true",
        help="continue attempting larger sizes even after a failure at a smaller size",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return sorted(dict.fromkeys(values))


def default_thread_counts() -> list[int]:
    cpu_count = max(os.cpu_count() or 1, 1)
    threads = []
    value = 1
    while value < cpu_count:
        threads.append(value)
        value *= 2
    threads.append(cpu_count)
    return sorted(dict.fromkeys(threads))


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(dict.fromkeys(values))


def exponents_to_sizes(exponents: Iterable[int]) -> list[int]:
    return [2 ** exponent for exponent in exponents]


def option_was_provided(option_name: str) -> bool:
    return any(
        argument == option_name or argument.startswith(f"{option_name}=")
        for argument in sys.argv[1:]
    )


def effective_default_thread_counts() -> list[int]:
    return default_thread_counts()


def normalize_algorithm_configs(args: argparse.Namespace) -> dict[str, AlgorithmRunConfig]:
    thread_override = parse_int_list(args.threads) if option_was_provided("--threads") and args.threads else None
    base_case_override = (
        parse_int_list(args.base_cases)
        if option_was_provided("--base-cases") and args.base_cases
        else None
    )
    repetition_override = args.repetitions if option_was_provided("--repetitions") else None

    configs: dict[str, AlgorithmRunConfig] = {}
    for algorithm_name in ALGORITHMS:
        raw_config = ALGORITHM_CONFIGS[algorithm_name]
        enabled = bool(raw_config.get("enabled", True))

        raw_sizes = raw_config.get("matrix_sizes")
        raw_exponents = raw_config.get("matrix_exponents")
        if raw_sizes is not None:
            matrix_sizes = unique_sorted(int(size) for size in raw_sizes)
        elif raw_exponents is not None:
            matrix_sizes = unique_sorted(exponents_to_sizes(int(exp) for exp in raw_exponents))
        else:
            raise SystemExit(f"{algorithm_name} must define matrix_sizes or matrix_exponents")

        if args.min_exp is not None:
            matrix_sizes = [size for size in matrix_sizes if int(math.log2(size)) >= args.min_exp]
        if args.max_exp is not None:
            matrix_sizes = [size for size in matrix_sizes if int(math.log2(size)) <= args.max_exp]

        if args.parallel_only_from_exp is not None and algorithm_name in {"Sequential", "Strassen"}:
            matrix_sizes = [
                size for size in matrix_sizes
                if int(math.log2(size)) < args.parallel_only_from_exp
            ]

        raw_threads = raw_config.get("threads")
        if thread_override is not None:
            threads = thread_override
        elif raw_threads is None:
            threads = effective_default_thread_counts()
        else:
            threads = unique_sorted(int(thread) for thread in raw_threads)

        raw_base_cases = raw_config.get("base_cases", [DEFAULT_BASE_CASE])
        if base_case_override is not None:
            base_cases = base_case_override
        else:
            base_cases = unique_sorted(int(base_case) for base_case in raw_base_cases)

        repetitions = repetition_override if repetition_override is not None else int(raw_config.get("repetitions", 1))

        if not enabled:
            matrix_sizes = []

        configs[algorithm_name] = AlgorithmRunConfig(
            name=algorithm_name,
            enabled=enabled,
            matrix_sizes=matrix_sizes,
            threads=threads,
            base_cases=base_cases,
            repetitions=repetitions,
        )

    return configs


def validate_algorithm_configs(configs: dict[str, AlgorithmRunConfig]) -> None:
    for config in configs.values():
        if config.repetitions <= 0:
            raise SystemExit(f"{config.name} repetitions must be positive")
        if any(thread <= 0 for thread in config.threads):
            raise SystemExit(f"{config.name} thread counts must be positive integers")
        if any(base_case <= 0 for base_case in config.base_cases):
            raise SystemExit(f"{config.name} base cases must be positive integers")
        if any(not is_power_of_two(size) for size in config.matrix_sizes):
            raise SystemExit(f"{config.name} matrix sizes must all be powers of two")


def format_algorithm_config_summary(configs: dict[str, AlgorithmRunConfig]) -> list[str]:
    lines: list[str] = []
    for algorithm_name in ALGORITHMS:
        config = configs[algorithm_name]
        if not config.enabled:
            lines.append(f"- {algorithm_name}: disabled")
            continue

        sizes = ", ".join(str(size) for size in config.matrix_sizes) if config.matrix_sizes else "none"
        threads = ", ".join(str(thread) for thread in config.threads) if config.threads else "none"
        base_cases = ", ".join(str(base_case) for base_case in config.base_cases) if config.base_cases else "none"
        lines.append(
            f"- {algorithm_name}: sizes=[{sizes}] threads=[{threads}] "
            f"base_cases=[{base_cases}] repetitions={config.repetitions}"
        )
    return lines


def detect_executable(build_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.is_file():
            return explicit.resolve()
        raise FileNotFoundError(f"matrix_mult executable not found at {explicit}")

    candidates = (
        build_dir / "matrix_mult",
        build_dir / "matrix_mult.exe",
        build_dir / "Release" / "matrix_mult.exe",
        build_dir / "Debug" / "matrix_mult.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    for candidate in build_dir.rglob("matrix_mult*"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the compiled matrix_mult executable. Build the project first, "
        "or pass --executable /path/to/matrix_mult."
    )


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def detect_os_info() -> OsInfo:
    system = sys.platform
    pretty_name = system
    distro_id = None
    distro_like = None

    if Path("/etc/os-release").is_file():
        values: dict[str, str] = {}
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
        distro_id = values.get("ID")
        distro_like = values.get("ID_LIKE")
        pretty_name = values.get("PRETTY_NAME", pretty_name)
    elif os.name == "nt":
        pretty_name = "Windows"
    elif sys.platform == "darwin":
        pretty_name = "macOS"

    return OsInfo(
        system=system,
        distro_id=distro_id,
        distro_like=distro_like,
        pretty_name=pretty_name,
    )


def find_command(*candidates: str) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except OSError as error:
        return str(error)

    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    return output[0].strip() if output else "version output unavailable"


def check_required_tools() -> list[ToolStatus]:
    statuses = [
        ToolStatus(
            name="python3",
            found=bool(sys.executable),
            detail=command_version([sys.executable, "--version"]),
        )
    ]

    bash_path = find_command("bash")
    statuses.append(
        ToolStatus(
            name="bash",
            found=bash_path is not None,
            detail=command_version([bash_path, "--version"]) if bash_path else "not found in PATH",
        )
    )

    compiler_path = find_command("g++", "c++", "clang++")
    statuses.append(
        ToolStatus(
            name="C++ compiler",
            found=compiler_path is not None,
            detail=command_version([compiler_path, "--version"]) if compiler_path else "not found in PATH",
        )
    )

    statuses.append(
        ToolStatus(
            name="OpenMP support",
            found=compiler_path is not None,
            detail=(
                "compiler found; final OpenMP verification happens during the build script step"
                if compiler_path is not None
                else "cannot verify without a detected C++ compiler"
            ),
        )
    )
    return statuses


def report_tool_statuses(statuses: list[ToolStatus]) -> None:
    print("Environment checks:")
    for status in statuses:
        marker = "OK" if status.found else "MISSING"
        print(f"  [{marker}] {status.name}: {status.detail}")


def install_hint(os_info: OsInfo) -> str | None:
    distro_tokens = " ".join(
        token for token in [os_info.distro_id or "", os_info.distro_like or ""] if token
    ).lower()

    if "fedora" in distro_tokens or "rhel" in distro_tokens or "centos" in distro_tokens:
        return "sudo dnf install gcc-c++ make bash"
    if "ubuntu" in distro_tokens or "debian" in distro_tokens:
        return "sudo apt update && sudo apt install -y g++ make bash"
    if os_info.system == "darwin":
        return "brew install bash gcc"
    return None


def ensure_required_tools() -> None:
    os_info = detect_os_info()
    print(f"Detected OS: {os_info.pretty_name}")
    statuses = check_required_tools()
    report_tool_statuses(statuses)

    missing = [
        status.name
        for status in statuses
        if not status.found and status.name != "OpenMP support"
    ]
    if missing:
        hint = install_hint(os_info)
        if hint:
            raise FileNotFoundError(
                "Missing required tools: " + ", ".join(missing) +
                ". Install them first, then rerun the experiment driver.\n"
                f"Suggested setup command for {os_info.pretty_name}:\n{hint}"
            )
        raise FileNotFoundError(
            "Missing required tools: " + ", ".join(missing) +
            ". Install them first, then rerun the experiment driver."
        )


def run_checked_command(command: list[str], cwd: Path, description: str) -> None:
    print(description)
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")


def normalize_explicit_executable(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit is None:
        return None
    return explicit if explicit.is_absolute() else (repo_root / explicit)


def ensure_built_executable(repo_root: Path, args: argparse.Namespace) -> Path:
    ensure_required_tools()

    build_dir = (repo_root / args.build_dir).resolve()
    explicit = normalize_explicit_executable(repo_root, args.executable)

    if args.skip_build:
        return detect_executable(build_dir, explicit)

    if explicit is not None and explicit.is_file():
        print(f"Using explicit executable without rebuilding: {explicit.resolve()}")
        return explicit.resolve()

    build_dir.mkdir(parents=True, exist_ok=True)
    build_script = repo_root / "scripts" / "build.sh"
    if not build_script.is_file():
        raise FileNotFoundError(f"Build script not found at {build_script}")

    bash_path = find_command("bash")
    if bash_path is None:
        raise FileNotFoundError(
            "Could not find 'bash' in PATH, which is required to run scripts/build.sh. "
            "Install bash or pass --skip-build with --executable."
        )

    run_checked_command(
        [bash_path, str(build_script)],
        repo_root,
        "Building matrix_mult executable with scripts/build.sh...",
    )

    try:
        executable = detect_executable(build_dir, explicit)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Build finished, but the matrix_mult executable could not be found. "
            "Check the CMake output and target name."
        ) from error

    print(f"Using executable: {executable}")
    return executable


def derive_seed(base_seed: int, size: int, run_index: int) -> int:
    return base_seed + size * 1000 + run_index


def write_assignment_input(
    output_path: Path,
    dimension: int,
    seed: int,
    min_value: int,
    max_value: int,
) -> None:
    import random

    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_a = [
        [rng.randint(min_value, max_value) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    matrix_b = [
        [rng.randint(min_value, max_value) for _ in range(dimension)]
        for _ in range(dimension)
    ]

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"{dimension}\n")
        write_matrix_rows(output_file, matrix_a)
        write_matrix_rows(output_file, matrix_b)


def stem_for_run(
    algorithm: str,
    matrix_size: int,
    threads: int,
    base_case: int,
    run_index: int,
) -> str:
    normalized_name = algorithm.lower()
    return f"exp_n{matrix_size}_{normalized_name}_t{threads}_b{base_case}_r{run_index}"


def read_info_file(info_path: Path) -> tuple[str, int, float]:
    text = info_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Info file is empty: {info_path}")

    blocks = [block.strip() for block in text.split("---") if block.strip()]
    latest = blocks[-1].splitlines()
    if len(latest) < 2:
        raise ValueError(f"Unexpected info file format: {info_path}")

    formatted_time = latest[0].strip()
    cores_used = int(latest[1].strip())
    runtime_seconds = duration_to_seconds(formatted_time)
    return formatted_time, cores_used, runtime_seconds


def duration_to_seconds(raw: str) -> float:
    hours, minutes, seconds = raw.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def command_for_run(
    executable: Path,
    input_path: Path,
    threads: int,
    base_case: int,
    algorithm: str,
) -> list[str]:
    return [
        str(executable),
        str(input_path),
        str(threads),
        str(base_case),
        algorithm,
    ]


def parameter_tag_for_run(algorithm: str, threads: int, base_case: int) -> str:
    if algorithm == "Sequential":
        return ""
    if algorithm == "ParMtrixMult":
        return f"threads-{threads}"
    if algorithm in {"Strassen", "ParStrassen"}:
        return f"threads-{threads}-basecase-{base_case}"
    return ""


def run_algorithm(
    executable: Path,
    repo_root: Path,
    results_dir: Path,
    input_path: Path,
    algorithm: str,
    matrix_size: int,
    threads: int,
    base_case: int,
    run_index: int,
    seed: int,
    timeout_seconds: int,
) -> RunRecord:
    command = command_for_run(executable, input_path, threads, base_case, algorithm)
    parameter_tag = parameter_tag_for_run(algorithm, threads, base_case)
    output_matrix_name = f"{input_path.stem}-output-{algorithm}"
    if parameter_tag:
        output_matrix_name += f"-{parameter_tag}"
    output_matrix_file = repo_root / f"{output_matrix_name}.txt"
    info_file = repo_root / f"{input_path.stem}-info-{algorithm}.txt"
    stdout_log = results_dir / f"{input_path.stem}-{algorithm}-stdout.log"
    stderr_log = results_dir / f"{input_path.stem}-{algorithm}-stderr.log"

    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout_log.write_text(error.stdout or "", encoding="utf-8")
        stderr_log.write_text(error.stderr or "", encoding="utf-8")
        return RunRecord(
            algorithm=algorithm,
            matrix_size=matrix_size,
            threads=threads,
            base_case=base_case,
            run_index=run_index,
            seed=seed,
            status="timeout",
            runtime_seconds=None,
            formatted_time=None,
            reported_cores=None,
            input_file=relative_to(input_path, repo_root),
            output_matrix_file=relative_to(output_matrix_file, repo_root),
            info_file=relative_to(info_file, repo_root),
            stdout_log=relative_to(stdout_log, repo_root),
            stderr_log=relative_to(stderr_log, repo_root),
            mismatch_warning="Warning:" in (error.stderr or ""),
            timed_out=True,
        )

    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    mismatch_warning = "Warning:" in completed.stderr

    if completed.returncode != 0:
        return RunRecord(
            algorithm=algorithm,
            matrix_size=matrix_size,
            threads=threads,
            base_case=base_case,
            run_index=run_index,
            seed=seed,
            status=f"failed_returncode_{completed.returncode}",
            runtime_seconds=None,
            formatted_time=None,
            reported_cores=None,
            input_file=relative_to(input_path, repo_root),
            output_matrix_file=relative_to(output_matrix_file, repo_root),
            info_file=relative_to(info_file, repo_root),
            stdout_log=relative_to(stdout_log, repo_root),
            stderr_log=relative_to(stderr_log, repo_root),
            mismatch_warning=mismatch_warning,
            timed_out=False,
        )

    if not info_file.exists():
        return RunRecord(
            algorithm=algorithm,
            matrix_size=matrix_size,
            threads=threads,
            base_case=base_case,
            run_index=run_index,
            seed=seed,
            status="missing_info_file",
            runtime_seconds=None,
            formatted_time=None,
            reported_cores=None,
            input_file=relative_to(input_path, repo_root),
            output_matrix_file=relative_to(output_matrix_file, repo_root),
            info_file=relative_to(info_file, repo_root),
            stdout_log=relative_to(stdout_log, repo_root),
            stderr_log=relative_to(stderr_log, repo_root),
            mismatch_warning=mismatch_warning,
            timed_out=False,
        )

    formatted_time, reported_cores, runtime_seconds = read_info_file(info_file)
    return RunRecord(
        algorithm=algorithm,
        matrix_size=matrix_size,
        threads=threads,
        base_case=base_case,
        run_index=run_index,
        seed=seed,
        status="ok",
        runtime_seconds=runtime_seconds,
        formatted_time=formatted_time,
        reported_cores=reported_cores,
        input_file=relative_to(input_path, repo_root),
        output_matrix_file=relative_to(output_matrix_file, repo_root),
        info_file=relative_to(info_file, repo_root),
        stdout_log=relative_to(stdout_log, repo_root),
        stderr_log=relative_to(stderr_log, repo_root),
        mismatch_warning=mismatch_warning,
        timed_out=False,
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_records(records: list[RunRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, int], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.status == "ok" and record.runtime_seconds is not None:
            grouped[(record.algorithm, record.matrix_size, record.threads, record.base_case)].append(record)

    summary_rows = []
    for (algorithm, matrix_size, threads, base_case), runs in sorted(grouped.items()):
        values = [run.runtime_seconds for run in runs if run.runtime_seconds is not None]
        avg_seconds = statistics.mean(values)
        summary_rows.append(
            {
                "algorithm": algorithm,
                "matrix_size": matrix_size,
                "threads": threads,
                "base_case": base_case,
                "runs": len(values),
                "avg_runtime_seconds": round(avg_seconds, 6),
                "min_runtime_seconds": round(min(values), 6),
                "max_runtime_seconds": round(max(values), 6),
                "stdev_runtime_seconds": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "mismatch_warnings": sum(1 for run in runs if run.mismatch_warning),
            }
        )
    return summary_rows


def row_lookup(rows: Iterable[dict[str, object]]) -> dict[tuple[str, int, int, int], dict[str, object]]:
    lookup = {}
    for row in rows:
        lookup[(str(row["algorithm"]), int(row["matrix_size"]), int(row["threads"]), int(row["base_case"]))] = row
    return lookup


def best_parallel_thread(rows: list[dict[str, object]], algorithm: str, matrix_size: int) -> int | None:
    candidates = [
        row for row in rows
        if row["algorithm"] == algorithm and int(row["matrix_size"]) == matrix_size
    ]
    if not candidates:
        return None
    return max(int(row["threads"]) for row in candidates)


def algorithm_slug(name: str) -> str:
    return name.lower()


def build_plot_specs(
    aggregated_rows: list[dict[str, object]],
    comparison_base_case: int,
) -> list[dict[str, object]]:
    lookup = row_lookup(aggregated_rows)
    sizes = sorted({int(row["matrix_size"]) for row in aggregated_rows})
    threads = sorted({int(row["threads"]) for row in aggregated_rows})
    base_cases = sorted({int(row["base_case"]) for row in aggregated_rows if int(row["base_case"]) > 0})

    plot_specs: list[dict[str, object]] = []

    runtime_series = {}
    for algorithm in ALGORITHMS:
        points = []
        for size in sizes:
            if algorithm == "Sequential":
                key = (algorithm, size, 1, comparison_base_case)
            elif algorithm == "ParMtrixMult":
                best_threads = best_parallel_thread(aggregated_rows, algorithm, size)
                if best_threads is None:
                    continue
                key = (algorithm, size, best_threads, comparison_base_case)
            elif algorithm == "Strassen":
                key = (algorithm, size, 1, comparison_base_case)
            else:
                best_threads = best_parallel_thread(
                    [
                        row for row in aggregated_rows
                        if row["algorithm"] == "ParStrassen" and int(row["base_case"]) == comparison_base_case
                    ],
                    algorithm,
                    size,
                )
                if best_threads is None:
                    continue
                key = (algorithm, size, best_threads, comparison_base_case)
            row = lookup.get(key)
            if row is not None:
                points.append((size, float(row["avg_runtime_seconds"])))
        if points:
            runtime_series[algorithm] = points
    if runtime_series:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "runtime_vs_matrix_size.png",
                "title": "Runtime vs Matrix Size",
                "x_label": "Matrix size (n)",
                "y_label": "Average runtime (seconds)",
                "series": runtime_series,
            }
        )

    pm_speedup = {}
    for size in sizes:
        seq = lookup.get(("Sequential", size, 1, comparison_base_case))
        if seq is None:
            continue
        series_points = []
        for thread in threads:
            row = lookup.get(("ParMtrixMult", size, thread, comparison_base_case))
            if row is None:
                continue
            runtime = float(row["avg_runtime_seconds"])
            if runtime > 0:
                series_points.append((thread, float(seq["avg_runtime_seconds"]) / runtime))
        if series_points:
            pm_speedup[f"n={size}"] = series_points
    if pm_speedup:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "parmtrixmult_speedup_vs_threads.png",
                "title": "ParMtrixMult Speedup vs Threads",
                "x_label": "Threads",
                "y_label": "Speedup over Sequential",
                "series": pm_speedup,
            }
        )

    ps_speedup = {}
    for size in sizes:
        base_row = lookup.get(("Strassen", size, 1, comparison_base_case))
        if base_row is None:
            continue
        series_points = []
        for thread in threads:
            row = lookup.get(("ParStrassen", size, thread, comparison_base_case))
            if row is None:
                continue
            runtime = float(row["avg_runtime_seconds"])
            if runtime > 0:
                series_points.append((thread, float(base_row["avg_runtime_seconds"]) / runtime))
        if series_points:
            ps_speedup[f"n={size}"] = series_points
    if ps_speedup:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "parstrassen_speedup_vs_threads.png",
                "title": "ParStrassen Speedup vs Threads",
                "x_label": "Threads",
                "y_label": "Speedup over Strassen",
                "series": ps_speedup,
            }
        )

    strassen_base_runtime = {}
    for size in sizes:
        points = []
        for base_case in base_cases:
            row = lookup.get(("Strassen", size, 1, base_case))
            if row is not None:
                points.append((base_case, float(row["avg_runtime_seconds"])))
        if points:
            strassen_base_runtime[f"n={size}"] = points
    if strassen_base_runtime:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "strassen_runtime_vs_base_case.png",
                "title": "Strassen Runtime vs Base Case",
                "x_label": "Base case",
                "y_label": "Average runtime (seconds)",
                "series": strassen_base_runtime,
            }
        )

    par_strassen_base_runtime = {}
    for size in sizes:
        thread = best_parallel_thread(
            [row for row in aggregated_rows if row["algorithm"] == "ParStrassen"],
            "ParStrassen",
            size,
        )
        if thread is None:
            continue
        points = []
        for base_case in base_cases:
            row = lookup.get(("ParStrassen", size, thread, base_case))
            if row is not None:
                points.append((base_case, float(row["avg_runtime_seconds"])))
        if points:
            par_strassen_base_runtime[f"n={size}, t={thread}"] = points
    if par_strassen_base_runtime:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "parstrassen_runtime_vs_base_case.png",
                "title": "ParStrassen Runtime vs Base Case",
                "x_label": "Base case",
                "y_label": "Average runtime (seconds)",
                "series": par_strassen_base_runtime,
            }
        )

    efficiency = {}
    for size in sizes:
        seq = lookup.get(("Sequential", size, 1, comparison_base_case))
        if seq is not None:
            points = []
            for thread in threads:
                row = lookup.get(("ParMtrixMult", size, thread, comparison_base_case))
                if row is None:
                    continue
                runtime = float(row["avg_runtime_seconds"])
                if runtime > 0 and thread > 0:
                    speedup = float(seq["avg_runtime_seconds"]) / runtime
                    points.append((thread, speedup / thread))
            if points:
                efficiency[f"ParMtrixMult n={size}"] = points

        strassen = lookup.get(("Strassen", size, 1, comparison_base_case))
        if strassen is not None:
            points = []
            for thread in threads:
                row = lookup.get(("ParStrassen", size, thread, comparison_base_case))
                if row is None:
                    continue
                runtime = float(row["avg_runtime_seconds"])
                if runtime > 0 and thread > 0:
                    speedup = float(strassen["avg_runtime_seconds"]) / runtime
                    points.append((thread, speedup / thread))
            if points:
                efficiency[f"ParStrassen n={size}"] = points
    if efficiency:
        plot_specs.append(
            {
                "kind": "line",
                "filename": "parallel_efficiency_vs_threads.png",
                "title": "Parallel Efficiency vs Threads",
                "x_label": "Threads",
                "y_label": "Efficiency",
                "series": efficiency,
            }
        )

    for size in sizes:
        pm_bars: list[tuple[str, float]] = []
        for thread in threads:
            row = lookup.get(("ParMtrixMult", size, thread, comparison_base_case))
            if row is not None:
                pm_bars.append((f"t={thread}", float(row["avg_runtime_seconds"])))
        if pm_bars:
            plot_specs.append(
                {
                    "kind": "bar",
                    "filename": (
                        f"{algorithm_slug('ParMtrixMult')}_threads_n{size}_"
                        f"b{comparison_base_case}.png"
                    ),
                    "title": f"ParMtrixMult Thread Comparison at n={size}",
                    "x_label": "Threads",
                    "y_label": "Average runtime (seconds)",
                    "bars": pm_bars,
                }
            )

        for base_case in base_cases:
            ps_bars: list[tuple[str, float]] = []
            for thread in threads:
                row = lookup.get(("ParStrassen", size, thread, base_case))
                if row is not None:
                    ps_bars.append((f"t={thread}", float(row["avg_runtime_seconds"])))
            if ps_bars:
                plot_specs.append(
                    {
                        "kind": "bar",
                        "filename": (
                            f"{algorithm_slug('ParStrassen')}_threads_n{size}_"
                            f"b{base_case}.png"
                        ),
                        "title": f"ParStrassen Thread Comparison at n={size}, b={base_case}",
                        "x_label": "Threads",
                        "y_label": "Average runtime (seconds)",
                        "bars": ps_bars,
                    }
                )

    return plot_specs


def make_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (PNG_WIDTH, PNG_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((PLOT_MARGIN_LEFT, 20), title, fill="black", font=font)
    return image, draw, font


def value_bounds(series: dict[str, list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    x_values = [point[0] for points in series.values() for point in points]
    y_values = [point[1] for points in series.values() for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min = 0
        y_max += 1
    else:
        y_min = min(0.0, y_min)
        y_max *= 1.1
    return x_min, x_max, y_min, y_max


def scale_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int]:
    plot_width = PNG_WIDTH - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT
    plot_height = PNG_HEIGHT - PLOT_MARGIN_TOP - PLOT_MARGIN_BOTTOM
    px = PLOT_MARGIN_LEFT + int((x - x_min) / (x_max - x_min) * plot_width)
    py = PLOT_MARGIN_TOP + plot_height - int((y - y_min) / (y_max - y_min) * plot_height)
    return px, py


def draw_axes(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    left = PLOT_MARGIN_LEFT
    right = PNG_WIDTH - PLOT_MARGIN_RIGHT
    top = PLOT_MARGIN_TOP
    bottom = PNG_HEIGHT - PLOT_MARGIN_BOTTOM
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    for index in range(6):
        x_value = x_min + (x_max - x_min) * index / 5
        x_pos, _ = scale_point(x_value, y_min, x_min, x_max, y_min, y_max)
        draw.line((x_pos, bottom, x_pos, bottom + 6), fill="black", width=1)
        label = f"{int(x_value) if x_value.is_integer() else round(x_value, 2)}"
        draw.text((x_pos - 10, bottom + 10), label, fill="black", font=font)

    for index in range(6):
        y_value = y_min + (y_max - y_min) * index / 5
        _, y_pos = scale_point(x_min, y_value, x_min, x_max, y_min, y_max)
        draw.line((left - 6, y_pos, left, y_pos), fill="black", width=1)
        label = f"{round(y_value, 2)}"
        draw.text((8, y_pos - 6), label, fill="black", font=font)

    draw.text((PNG_WIDTH // 2 - 50, PNG_HEIGHT - 36), x_label, fill="black", font=font)
    draw.text((10, 36), y_label, fill="black", font=font)


def draw_line_plot(
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    image, draw, font = make_canvas(title)
    x_min, x_max, y_min, y_max = value_bounds(series)
    draw_axes(draw, font, x_label, y_label, x_min, x_max, y_min, y_max)

    legend_x = PNG_WIDTH - 260
    legend_y = 20
    for index, (name, points) in enumerate(sorted(series.items())):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        sorted_points = sorted(points)
        scaled = [
            scale_point(x, y, x_min, x_max, y_min, y_max)
            for x, y in sorted_points
        ]
        if len(scaled) >= 2:
            draw.line(scaled, fill=color, width=3)
        for px, py in scaled:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=color)

        ly = legend_y + index * 20
        draw.line((legend_x, ly + 7, legend_x + 18, ly + 7), fill=color, width=3)
        draw.text((legend_x + 24, ly), name, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def draw_bar_plot(
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    bars: list[tuple[str, float]],
) -> None:
    image, draw, font = make_canvas(title)
    left = PLOT_MARGIN_LEFT
    right = PNG_WIDTH - PLOT_MARGIN_RIGHT
    top = PLOT_MARGIN_TOP
    bottom = PNG_HEIGHT - PLOT_MARGIN_BOTTOM
    plot_width = right - left
    plot_height = bottom - top

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    max_value = max(value for _, value in bars)
    padded_max = max(1.0, max_value * 1.1)
    for index in range(6):
        value = padded_max * index / 5
        y_pos = bottom - int((value / padded_max) * plot_height)
        draw.line((left - 6, y_pos, left, y_pos), fill="black", width=1)
        draw.text((8, y_pos - 6), f"{round(value, 2)}", fill="black", font=font)

    bar_width = max(32, int(plot_width / max(len(bars), 1) * 0.5))
    gap = int((plot_width - bar_width * len(bars)) / max(len(bars) + 1, 1))
    x_pos = left + gap

    for index, (label, value) in enumerate(bars):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        bar_height = 0 if padded_max == 0 else int((value / padded_max) * plot_height)
        draw.rectangle((x_pos, bottom - bar_height, x_pos + bar_width, bottom), fill=color, outline=color)
        draw.text((x_pos, bottom + 10), label, fill="black", font=font)
        x_pos += bar_width + gap

    draw.text((PNG_WIDTH // 2 - 50, PNG_HEIGHT - 36), x_label, fill="black", font=font)
    draw.text((10, 36), y_label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def write_summary_markdown(
    output_path: Path,
    records: list[RunRecord],
    aggregated_rows: list[dict[str, object]],
    args: argparse.Namespace,
    configs: dict[str, AlgorithmRunConfig],
) -> None:
    ok_records = [record for record in records if record.status == "ok"]
    failed_records = [record for record in records if record.status != "ok"]
    sizes = sorted({record.matrix_size for record in ok_records})
    algorithms = sorted({record.algorithm for record in ok_records})
    lines = [
        "# Experiment Summary",
        "",
        "## Configuration",
        "",
        f"- Comparison base case: {args.comparison_base_case}",
        f"- Global exponent filter: min={args.min_exp if args.min_exp is not None else 'none'}, max={args.max_exp if args.max_exp is not None else 'none'}",
        f"- Parallel-only cutoff override: {args.parallel_only_from_exp if args.parallel_only_from_exp is not None else 'none'}",
        "",
        "### Effective Algorithm Configs",
        "",
        "## Outcome",
        "",
        f"- Successful runs: {len(ok_records)}",
        f"- Failed or timed out runs: {len(failed_records)}",
        f"- Algorithms covered: {', '.join(algorithms) if algorithms else 'none'}",
        f"- Successful matrix sizes: {', '.join(str(value) for value in sizes) if sizes else 'none'}",
        "",
        "## Notes",
        "",
        "- Raw per-run data is stored in `raw_results.csv`.",
        "- Aggregated averages, min/max, and standard deviation are stored in `aggregated_results.csv`.",
        "- PNG charts are written to the plots directory.",
    ]
    config_lines = format_algorithm_config_summary(configs)
    lines[10:10] = config_lines + [""]

    if failed_records:
        lines.extend(
            [
                "",
                "## Failures",
                "",
            ]
        )
        for record in failed_records[:20]:
            lines.append(
                f"- {record.algorithm} n={record.matrix_size} t={record.threads} "
                f"b={record.base_case} run={record.run_index}: {record.status}"
            )

    if aggregated_rows:
        lines.extend(
            [
                "",
                "## Representative Configurations",
                "",
            ]
        )
        for row in aggregated_rows[:12]:
            lines.append(
                f"- {row['algorithm']} n={row['matrix_size']} t={row['threads']} "
                f"b={row['base_case']} avg={row['avg_runtime_seconds']}s"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_experiment_artifacts(
    records: list[RunRecord],
    results_dir: Path,
    plots_dir: Path,
    args: argparse.Namespace,
    configs: dict[str, AlgorithmRunConfig],
) -> None:
    raw_results_path = results_dir / "raw_results.csv"
    aggregated_path = results_dir / "aggregated_results.csv"
    summary_path = results_dir / "summary.md"

    raw_rows = [asdict(record) for record in records]
    write_csv(
        raw_results_path,
        list(raw_rows[0].keys()) if raw_rows else list(RunRecord.__annotations__.keys()),
        raw_rows,
    )

    aggregated_rows = aggregate_records(records)
    aggregated_fieldnames = [
        "algorithm",
        "matrix_size",
        "threads",
        "base_case",
        "runs",
        "avg_runtime_seconds",
        "min_runtime_seconds",
        "max_runtime_seconds",
        "stdev_runtime_seconds",
        "mismatch_warnings",
    ]
    write_csv(aggregated_path, aggregated_fieldnames, aggregated_rows)

    for plot_spec in build_plot_specs(aggregated_rows, args.comparison_base_case):
        output_path = plots_dir / str(plot_spec["filename"])
        if plot_spec["kind"] == "line":
            draw_line_plot(
                output_path,
                str(plot_spec["title"]),
                str(plot_spec["x_label"]),
                str(plot_spec["y_label"]),
                dict(plot_spec["series"]),
            )
        else:
            draw_bar_plot(
                output_path,
                str(plot_spec["title"]),
                str(plot_spec["x_label"]),
                str(plot_spec["y_label"]),
                list(plot_spec["bars"]),
            )

    write_summary_markdown(summary_path, records, aggregated_rows, args, configs)


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions is not None and args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.min_value > args.max_value:
        raise SystemExit("--min-value cannot be greater than --max-value")
    if args.min_exp is not None and args.max_exp is not None and args.min_exp > args.max_exp:
        raise SystemExit("--min-exp cannot be greater than --max-exp")
    if args.parallel_only_from_exp is not None and args.parallel_only_from_exp < 0:
        raise SystemExit("--parallel-only-from-exp cannot be negative")
    if args.comparison_base_case <= 0:
        raise SystemExit("--comparison-base-case must be positive")


def run_experiments(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    executable = ensure_built_executable(repo_root, args)
    validate_args(args)
    configs = normalize_algorithm_configs(args)
    validate_algorithm_configs(configs)

    input_dir = (repo_root / args.input_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()
    plots_dir = (repo_root / args.plots_dir).resolve()
    ensure_directories(input_dir, results_dir, plots_dir)

    records: list[RunRecord] = []
    size_failure_cutoff: int | None = None
    matrix_sizes = sorted(
        {
            matrix_size
            for config in configs.values()
            if config.enabled
            for matrix_size in config.matrix_sizes
        }
    )

    for matrix_size in matrix_sizes:
        if size_failure_cutoff is not None and matrix_size > size_failure_cutoff:
            print(f"Skipping n={matrix_size} because n={size_failure_cutoff} already failed.")
            continue

        print(f"\n=== Matrix size n={matrix_size} ===")
        size_failed = False

        try:
            stop_current_size = False
            for algorithm_name in ALGORITHMS:
                config = configs[algorithm_name]
                if not config.enabled or matrix_size not in config.matrix_sizes:
                    continue

                for run_index in range(1, config.repetitions + 1):
                    seed = derive_seed(args.seed, matrix_size, run_index)
                    for base_case in config.base_cases:
                        for thread in config.threads:
                            input_stem = stem_for_run(
                                algorithm_name,
                                matrix_size,
                                thread,
                                base_case,
                                run_index,
                            )
                            input_path = input_dir / f"{input_stem}.txt"
                            write_assignment_input(
                                input_path,
                                matrix_size,
                                seed,
                                args.min_value,
                                args.max_value,
                            )
                            print(
                                f"Running {algorithm_name} for n={matrix_size}, "
                                f"t={thread}, b={base_case}, run={run_index}"
                            )
                            record = run_algorithm(
                                executable,
                                repo_root,
                                results_dir,
                                input_path,
                                algorithm_name,
                                matrix_size,
                                thread,
                                base_case,
                                run_index,
                                seed,
                                args.timeout_seconds,
                            )
                            records.append(record)
                            refresh_experiment_artifacts(records, results_dir, plots_dir, args, configs)
                            if record.status != "ok":
                                size_failed = True
                                if not args.keep_going_after_size_failure:
                                    stop_current_size = True
                                    break
                        if stop_current_size:
                            break
                    if stop_current_size:
                        break
                if stop_current_size:
                    break
        except (OSError, ValueError) as error:
            failure_log = results_dir / f"size_{matrix_size}_failure.log"
            failure_log.write_text(str(error), encoding="utf-8")
            size_failed = True

        if size_failed and not args.keep_going_after_size_failure:
            size_failure_cutoff = matrix_size

    refresh_experiment_artifacts(records, results_dir, plots_dir, args, configs)

    raw_results_path = results_dir / "raw_results.csv"
    aggregated_path = results_dir / "aggregated_results.csv"
    summary_path = results_dir / "summary.md"

    print("\nExperiment run completed.")
    print(f"Raw results: {raw_results_path}")
    print(f"Aggregated results: {aggregated_path}")
    print(f"Summary: {summary_path}")
    print(f"Plots directory: {plots_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_experiments(args)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
