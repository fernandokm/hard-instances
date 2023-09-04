#!/usr/bin/env python

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from pysat.formula import CNF
from solvers.pysat import PySAT
from tqdm.auto import tqdm, trange


class Args(argparse.Namespace):
    outdir: Path
    num_vars: int
    num_clauses: int
    num_instances: int
    runs: int
    num_cpus: int
    seed: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--num_vars", type=int, default=100)
    parser.add_argument("--num_clauses", type=int, default=420)
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tasks = []
    skipped = 0
    for i in trange(args.num_instances, desc="generating instances"):
        file = args.outdir / f"{i}.cnf"
        if file.exists():
            skipped += 1
        else:
            instance = random_k_sat(
                rng, n_vars=args.num_vars, n_clauses=args.num_clauses, k=3
            )
            instance.to_file(str(file))

        for run in range(args.runs):
            tasks.append((str(args.outdir), i, run))

    if skipped > 0:
        print(f"Skipped generation of {skipped} existing instances")

    results = []
    with multiprocessing.Pool(args.num_cpus) as pool:
        it = tqdm(
            pool.imap_unordered(solve, tasks),
            total=len(tasks),
            desc="solving instances",
        )
        for result in it:
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(str(args.outdir / "metrics.csv"), index=False)


def random_k_sat(
    rng: np.random.Generator,
    n_vars: int,
    n_clauses: int,
    k: int = 3,
) -> CNF:
    variables = rng.choice(1 + np.arange(n_vars), size=(n_clauses, k))
    polarities = rng.choice([1, -1], size=(n_clauses, k))
    literals = variables * polarities
    return CNF(from_clauses=literals.tolist())


def solve(data: tuple[str, int, int]):
    outdir, instance, run = data
    file = outdir + f"/{instance}.cnf"
    cnf = CNF(from_file=file)
    metrics = PySAT("minisat22").solve_instance(cnf)

    return {
        "instance": instance,
        "run": run,
        **metrics,
    }


if __name__ == "__main__":
    main()
