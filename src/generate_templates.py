#!/usr/bin/env python

import argparse
import gzip
from pathlib import Path

import numpy as np
import utils
from generation.graph import SATGraph, SplittableCNF
from tqdm.auto import trange


class Args(argparse.Namespace):
    out_dir: Path
    num_vars: int
    num_clauses: int
    k: int
    num_templates: int
    seed: int
    complexify_min: float | None
    complexify_max: float | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="templates", type=Path)
    parser.add_argument("--num_vars", default=100, type=int)
    parser.add_argument("--num_clauses", default=420, type=int)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--num_templates", default=50000, type=int)
    parser.add_argument("--complexify_min", type=float, default=None)
    parser.add_argument("--complexify_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.complexify_min is None and args.complexify_max is None:
        complexify = None
        complexify_str = ""
    else:
        cmin = 0.0 if args.complexify_min is None else args.complexify_min
        cmax = 1.0 if args.complexify_max is None else args.complexify_max
        complexify = (cmin, cmax)
        complexify_str = f"({cmin},{cmax})"

    filename = (
        f"{args.num_vars}x{args.num_clauses}x{args.k}{complexify_str}_"
        f"{args.num_templates}_seed{args.seed}.txt.gz"
    )
    out_file = args.out_dir / filename
    print("Writing templates to file:", out_file)

    with gzip.open(out_file, mode="wt") as f:
        for _ in trange(args.num_templates, unit="templates"):
            if complexify is None:
                t = SATGraph.sample_template(
                    args.num_vars, args.num_clauses * args.k, seed=rng
                )
                print(t.tolist(), file=f)
            else:
                total_splits = args.num_clauses * (args.k - 1)
                min_splits = max(1, int(total_splits * complexify[0]))
                max_splits = int(total_splits * complexify[1])
                actual_splits = rng.integers(min_splits, max_splits, endpoint=True)

                clauses = utils.random_k_sat(
                    rng, args.num_vars, args.num_clauses, args.k
                ).clauses
                splittable = SplittableCNF(clauses, rng)
                for _ in range(actual_splits):
                    splittable.random_split()

                print(splittable.clauses, file=f)

    print("Done")


if __name__ == "__main__":
    main()
