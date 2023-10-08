#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import evaluation
from solvers.pysat import PySAT

if TYPE_CHECKING:
    from solvers.base import Solver


class Args(argparse.Namespace):
    output: Path
    num_vars: list[int]
    alphas: list[float]
    runs: int
    solvers: list[str]
    num_cpus: int
    seed: int
    force: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default="ksat_eval.parquet")
    parser.add_argument("-n", "--num_vars", type=int, action="append", default=[])
    parser.add_argument(
        "-a", "--alpha", type=float, action="append", dest="alphas", default=[]
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--solver", type=str, action="append", dest="solvers")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args(namespace=Args())

    if not args.num_vars:
        args.num_vars = [100]
    if not args.alphas:
        args.alphas = [4.2]

    return args


def main():
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f'Output file "{args.output}" exists, use --force to run anyway')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    solvers: list[Solver]
    if args.solvers:
        solvers = [PySAT(name) for name in args.solvers]
    else:
        solvers = [PySAT("m22")]

    results = evaluation.generate_and_eval_ksat_par(
        solvers,
        args.num_vars,
        args.alphas,
        args.runs,
        args.num_cpus,
        args.seed,
    )

    if args.output.suffix.lower() == ".parquet":
        results.to_parquet(args.output)
    else:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
