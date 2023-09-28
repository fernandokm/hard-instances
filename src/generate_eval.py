#!/usr/bin/env python

import argparse
from pathlib import Path

import evaluation
import pandas as pd
from solvers.pysat import PySAT
from tqdm.auto import tqdm


class Args(argparse.Namespace):
    results_dir: Path
    output: str
    num_vars: list[int]
    alphas: list[float]
    checkpoints: list[int]
    runs: int
    num_cpus: int
    seed: int
    device: str | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=str, default="eval.parquet")
    parser.add_argument(
        "-c", "--checkpoint", type=int, action="append", dest="checkpoints", default=[]
    )
    parser.add_argument("-n", "--num_vars", type=int, action="append", default=[])
    parser.add_argument(
        "-a", "--alpha", type=float, action="append", dest="alphas", default=[]
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
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
        print(f'Output file "{outfile}" exists, use --force to overwrite it')
        return

    results = []
    for ckpt in tqdm(args.checkpoints, desc="checkpoints", position=1):
        model_path = args.results_dir / f"checkpoints/{ckpt}.pt"
        policy = evaluation.load_policy(str(model_path), args.device)
        r = evaluation.generate_and_eval_par(
            policy,
            PySAT("m22"),
            args.num_vars,
            args.alphas,
            args.runs,
            args.num_cpus,
            args.seed,
            desc=f"checkpoint={ckpt}",
        )
        r["episode"] = ckpt
        results.append(r)

    results = pd.concat(results)
    if outfile.suffix.lower() == ".parquet":
        results.to_parquet(outfile)
    else:
        results.to_csv(outfile)


if __name__ == "__main__":
    main()
