#!/usr/bin/env python

import argparse
import multiprocessing
import multiprocessing.synchronize
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

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
    noise_start: float
    noise_end: float
    noise_cpus: int
    force: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=str, default="rerun.parquet")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_repetitions", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="minisat22")
    parser.add_argument("--noise_start", type=float, default=1)
    parser.add_argument("--noise_end", type=float, default=0)
    parser.add_argument("--noise_cpus", type=int, default=1)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    outfile = args.results_dir / args.output
    if outfile.exists() and not args.force:
        print(f'Output file "{outfile}" exists, use --force to overwrite it')
        return

    history = History.load(args.results_dir, load_step=True)

    tasks = list(history.episode.index.unique("episode").sort_values())
    results = []
    with multiprocessing.Pool(args.num_cpus, worker_init, (args,)) as pool:
        with Spinner(args.noise_cpus) as spinner:
            it = tqdm(
                pool.imap_unordered(worker_solve, tasks),
                total=len(tasks),
                desc=str(args.results_dir),
                unit="episodes",
            )
            for partial_results in it:
                progress = it.n / it.total
                spinner.set_spin(args.noise_start < progress < args.noise_end)

                results += partial_results

    results_df = pd.DataFrame(results).set_index(["episode", "run"]).sort_index()
    if outfile.suffix.lower() == ".parquet":
        results_df.to_parquet(outfile)
    else:
        results_df.to_csv(outfile)


def worker_init(args: Args):
    global num_repetitions, solver, history
    num_repetitions = args.num_repetitions
    solver = PySAT(args.solver)
    history = History.load(args.results_dir, load_step=True)

    # Only keep the required columns (minimize memory usage)
    history.step = history.step[["action_0", "action_1"]].copy()
    for col in history.step.columns:
        history.step[col] = pd.to_numeric(history.step[col], downcast="integer")


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


class Spinner(AbstractContextManager):
    def __init__(self, num_cpus: int) -> None:
        self.num_cpus = num_cpus
        self.processes: list[multiprocessing.Process] = []
        self.run_event = multiprocessing.Event()
        for _ in range(self.num_cpus):
            p = multiprocessing.Process(
                target=Spinner.spin, args=(self.run_event,), daemon=True
            )
            p.start()
            self.processes.append(p)

    def set_spin(self, status: bool) -> None:
        if status:
            self.run_event.set()
        else:
            self.run_event.clear()

    def __enter__(self) -> "Spinner":
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        for p in self.processes:
            p.kill()
            p.join()
            p.close()

    @staticmethod
    def spin(run: multiprocessing.synchronize.Event):
        while True:
            run.wait()
            while run.is_set():
                for _ in range(10000):
                    pass


if __name__ == "__main__":
    main()
