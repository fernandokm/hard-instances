#!/usr/bin/env python

import argparse
import multiprocessing
import multiprocessing.synchronize
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

import pandas as pd
from history import History
from rich_argparse import RichHelpFormatter
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
    RichHelpFormatter.highlights.append(r"(?P<args>results_dir)")
    parser = argparse.ArgumentParser(
        description=(
            "Recomputes the training metrics for a model. This makes it possible "
            "to recompute the cpu_time metric on a quieter CPU, with less noise "
            "than during training. This script also supports the intentional addition "
            "of a CPU load during the process for analysis purposes."
        ),
        formatter_class=lambda *args, **kwargs: RichHelpFormatter(
            *args, **kwargs, max_help_position=28, width=90
        ),
        add_help=False,
    )

    group = parser.add_argument_group("Main options")
    group.add_argument(
        "results_dir",
        type=Path,
        help=(
            "the directory with the training results, "
            "e.g., runs/SAGE/1970-01-01T00:00:00"
        ),
    )
    group.add_argument(
        "-o",
        "--output",
        type=str,
        default="rerun.parquet",
        metavar="STR",
        help=(
            "name of the output file, which will be saved in the results_dir "
            "\\[default: rerun.parquet]"
        ),
    )
    group.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        metavar="INT",
        help=(
            "number of solver processes to run in parallel; each solver process "
            "solves a different instance \\[default: 1]"
        ),
    )
    group.add_argument(
        "--num_repetitions",
        type=int,
        default=1000,
        metavar="INT",
        help="how many times to solve each instance \\[default: 1000]",
    )
    group.add_argument(
        "--solver",
        type=str,
        default="minisat22",
        metavar="SOLVER",
        help=(
            "which solver to use; can be any solver accepted by PySAT "
            "(see `pysat.solvers.SolverNames` for a full list of accepted "
            "solver names) \\[default: minisat22]"
        ),
    )
    group.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite output files",
    )
    group.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )

    ##################
    # NOISE OPTIONS #
    ##################
    group = parser.add_argument_group(
        title="Noise",
        description=(
            "This script can simulate the effects of other processes during training. "
            "To do this, the instances generated during training are re-evaluated "
            "in the same order as during training (some variations are possible due to "
            "parallelism), and a CPU load is introduced. The load consists of a "
            "configurable number of processes, each of which consumes 100% CPU "
            "(spin loop)."
        ),
    )
    group.add_argument(
        "--noise_start",
        type=float,
        default=1,
        metavar="FLOAT",
        help=(
            "approximate episode at which to start the CPU load, re-scaled to the "
            "interval \\[0, 1] \\[default: 1.0]"
        ),
    )
    group.add_argument(
        "--noise_end",
        type=float,
        default=0,
        metavar="FLOAT",
        help=(
            "approximate episode at which to end the CPU load, re-scaled to the "
            "interval \\[0, 1] \\[default: 0.0]"
        ),
    )
    group.add_argument(
        "--noise_cpus",
        type=int,
        default=1,
        metavar="INT",
        help="how many CPUs should be used \\[default: 1]",
    )

    # TODO: add groups?

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
