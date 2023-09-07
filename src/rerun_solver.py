#!/usr/bin/env python

import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
from history import History
from solvers.pysat import PySAT
from tqdm.auto import tqdm


class Args(argparse.Namespace):
    results_dir: Path
    output: str
    num_cpus: int
    num_repetitions: int
    solver: str
    force: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=str, default="rerun.csv")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_repetitions", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="minisat22")
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    outfile = args.results_dir / args.output
    if outfile.exists() and not args.force:
        print(f'Output file "{outfile}" exists, use --force to overwrite it')
        return

    history = History.load(args.results_dir, keep_full_steps=True)

    tasks = list(history.episode.index)[:100]
    results = []
    with multiprocessing.Pool(args.num_cpus, worker_init, (args,)) as pool:
        it = tqdm(
            pool.imap_unordered(worker_solve, tasks),
            total=len(tasks),
            desc=str(args.results_dir),
            unit="episodes",
        )
        for partial_results in it:
            results += partial_results

    results_df = pd.DataFrame(results).set_index(["episode", "run"])  # .sort_index()
    if outfile.suffix.lower() == ".parquet":
        results_df.to_parquet(outfile)
    else:
        results_df.to_csv(outfile)


def worker_init(args: Args):
    global num_repetitions, solver, history
    num_repetitions = args.num_repetitions
    solver = PySAT(args.solver)
    history = History.load(args.results_dir, keep_full_steps=True)


def worker_solve(episode: int):
    graph = history.get_graph(episode)
    clauses = graph.to_clauses()
    results = []
    for i in range(num_repetitions):
        r = solver.solve_instance(clauses)
        r["episode"] = episode
        r["run"] = i
        results.append(r)

    return results


if __name__ == "__main__":
    main()
