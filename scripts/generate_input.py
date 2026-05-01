#!/usr/bin/env python3
"""Generate matrix input files for the ICS 507 matrix multiplication project."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two random n by n matrices in the assignment input format."
    )
    parser.add_argument("dimension", type=int, help="matrix dimension n; must be a power of 2")
    parser.add_argument("--output", type=Path, default=None, help="output text file path")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible files")
    parser.add_argument("--min", dest="min_value", type=int, default=-9, help="minimum matrix value")
    parser.add_argument("--max", dest="max_value", type=int, default=9, help="maximum matrix value")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"tests/input_{args.dimension}.txt")
    return args


def generate_matrix(dimension: int, min_value: int, max_value: int) -> list[list[int]]:
    return [
        [random.randint(min_value, max_value) for _ in range(dimension)]
        for _ in range(dimension)
    ]


def write_matrix_rows(output_file, matrix: list[list[int]]) -> None:
    for row in matrix:
        output_file.write(" ".join(str(value) for value in row))
        output_file.write("\n")


def main() -> int:
    args = parse_args()

    if not is_power_of_two(args.dimension):
        raise SystemExit("Error: dimension must be a positive power of 2.")

    if args.min_value > args.max_value:
        raise SystemExit("Error: --min cannot be greater than --max.")

    if args.seed is not None:
        random.seed(args.seed)

    matrix_a = generate_matrix(args.dimension, args.min_value, args.max_value)
    matrix_b = generate_matrix(args.dimension, args.min_value, args.max_value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        output_file.write(f"{args.dimension}\n")
        write_matrix_rows(output_file, matrix_a)
        write_matrix_rows(output_file, matrix_b)

    print(f"Wrote {args.dimension}x{args.dimension} matrices to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
