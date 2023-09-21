#!/usr/bin/env python

import argparse
import gzip
from pathlib import Path

import numpy as np
import utils
from tqdm.auto import trange


class Args(argparse.Namespace):
    out_dir: Path
    num_vars: int
    num_clauses: int
    k: int
    num_instances: int
    seed: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--num_vars", type=int, default=100)
    parser.add_argument("--num_clauses", type=int, default=420)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    filename = (
        f"{args.num_vars}x{args.num_clauses}x{args.k}_"
        f"{args.num_instances}_seed{args.seed}.txt.gz"
    )
    out_file = args.out_dir / filename
    print("Writing instances to file:", out_file)

    with gzip.open(out_file, mode="wt") as f:
        for _ in trange(args.num_instances, unit="templates"):
            instance = utils.random_k_sat(
                rng, n_vars=args.num_vars, n_clauses=args.num_clauses, k=args.k
            )
            print(instance.clauses, file=f)


if __name__ == "__main__":
    main()
