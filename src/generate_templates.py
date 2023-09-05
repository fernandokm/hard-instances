#!/usr/bin/env python

import argparse
import gzip
from pathlib import Path

import numpy as np
from generation.graph import SATGraph
from tqdm.auto import trange


class Args(argparse.Namespace):
    out_dir: Path
    num_vars: int
    num_clauses: int
    k: int
    num_templates: int
    seed: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="templates", type=Path)
    parser.add_argument("--num_vars", default=100, type=int)
    parser.add_argument("--num_clauses", default=420, type=int)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--num_templates", default=50000, type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    filename = (
        f"{args.num_vars}x{args.num_clauses}x{args.k}_"
        f"{args.num_templates}_seed{args.seed}.txt.gz"
    )
    out_file = args.out_dir / filename
    print("Writing templates to file:", out_file)

    with gzip.open(out_file, mode="wt") as f:
        for _ in trange(args.num_templates, unit="templates"):
            t = SATGraph.sample_template(
                args.num_vars, args.num_clauses * args.k, seed=rng
            )
            print(t.tolist(), file=f)

    print("Done")


if __name__ == "__main__":
    main()
