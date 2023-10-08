#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import evaluation
import pandas as pd
from solvers.pysat import PySAT

if TYPE_CHECKING:
    from solvers.base import Solver


class Args(argparse.Namespace):
    results_dir: Path
    output: str
    num_vars: list[int]
    alphas: list[float]
    checkpoints: list[int]
    runs: int
    num_sampled_pairs: list[int]
    solvers: list[str]
    num_cpus: int
    seed: int
    device: str | None
    force: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=str, default="eval.parquet")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        action="append",
        dest="checkpoints",
        required=True,
        default=[],
    )
    parser.add_argument("-n", "--num_vars", type=int, action="append", default=[])
    parser.add_argument(
        "-a", "--alpha", type=float, action="append", dest="alphas", default=[]
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--num_sampled_pairs", type=int, action="append")
    parser.add_argument("--solver", type=str, action="append", dest="solvers")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args(namespace=Args())

    if not args.num_vars:
        args.num_vars = [100]
    if not args.alphas:
        args.alphas = [4.2]

    return args


def main():
    args = parse_args()
    outfile = args.results_dir / args.output
    if outfile.exists() and not args.force:
        print(f'Output file "{outfile}" exists, use --force to run anyway')
        return

    results = []
    if args.device is not None and "," in args.device:
        devices = [d.strip() for d in args.device.split(",")]
    else:
        devices = [args.device]

    solvers: list[Solver]
    if args.solvers:
        solvers = [PySAT(name) for name in args.solvers]
    else:
        solvers = [PySAT("m22")]

    for ckpt in args.checkpoints:
        model_path = args.results_dir / f"checkpoints/{ckpt}.pt"
        policies = [evaluation.load_policy(str(model_path), device=d) for d in devices]
        if not args.num_sampled_pairs:
            args.num_sampled_pairs = [policies[0].num_sampled_pairs]

        for n in args.num_sampled_pairs:
            for p in policies:
                p.num_sampled_pairs = n

            r = evaluation.generate_and_eval_par(
                policies,
                solvers,
                args.num_vars,
                args.alphas,
                args.runs,
                args.num_cpus,
                args.seed,
                desc=f"checkpoint={ckpt}, num_sampled_pairs={n}",
            )
            r["episode"] = ckpt
            results.append(r)

    results = pd.concat(results)
    if outfile.suffix.lower() == ".parquet":
        results.to_parquet(outfile)
    else:
        results.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
