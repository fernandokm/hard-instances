#!/usr/bin/env python

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import utils
from pysat.formula import CNF
from rich_argparse import RichHelpFormatter
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
    parser = argparse.ArgumentParser(
        description=(
            "An older alternative to `generate_eval_ksat.py` which was used "
            "earlier on in the project. This script has been left here "
            "because it is needed to generate the data for the notebook "
            "metrics.ipynb. This script does not support repeatable options. "
            "It saves the generated instances as cnf files."
        ),
        formatter_class=lambda *args, **kwargs: RichHelpFormatter(
            *args, **kwargs, max_help_position=28, width=90
        ),
        add_help=False,
    )
    parser.add_argument(
        "outdir",
        type=Path,
        help="the directory in which to store the generated instances and metrics",
    )
    parser.add_argument(
        "--num_vars",
        type=int,
        default=100,
        metavar="INT",
        help="number of variables in the generated instances \\[default: 100]",
    )
    parser.add_argument(
        "--num_clauses",
        type=int,
        default=420,
        metavar="INT",
        help="number of clauses in the generated instances \\[default: 420]",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1000,
        metavar="INT",
        help="number of instances to generate \\[default: 1000]",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="INT",
        help="number of times to solve/evaluate each instance \\[default: 3]",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        metavar="INT",
        help="number of solver processes to run in parallel \\[default: 1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="seed for all random number generators \\[default: 0]",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )
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
            instance = utils.random_k_sat(
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


def solve(data: tuple[str, int, int]):
    outdir, instance, run = data
    file = outdir + f"/{instance}.cnf"
    cnf = CNF(from_file=file)
    metrics = PySAT("minisat22").solve_instance(cnf.clauses)

    return {
        "instance": instance,
        "run": run,
        **metrics,
    }


if __name__ == "__main__":
    main()
