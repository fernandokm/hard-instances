#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import evaluation
import pandas as pd
from rich_argparse import RichHelpFormatter
from solvers.pysat import PySAT

if TYPE_CHECKING:
    from solvers.base import Solver


class Args(argparse.Namespace):
    results_dir: Path
    output: str
    checkpoints: list[int]
    num_vars: list[int]
    alphas: list[float]
    complexify: list[float]
    runs: int
    num_sampled_pairs: list[int]
    solvers: list[str]
    num_cpus: int
    save_instances: bool
    multinomial_templates: bool
    seed: int
    device: str | None
    force: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluates a G2SAT model adapted for the generation of hard 3-SAT instances"
        ),
        formatter_class=lambda *args, **kwargs: RichHelpFormatter(
            *args, **kwargs, max_help_position=28, width=90
        ),
        add_help=False,
    )

    ############
    # REQUIRED #
    ############
    group = parser.add_argument_group("Required arguments")
    group.add_argument(
        "results_dir",
        type=Path,
        help=(
            "the directory with the training results, "
            "e.g., runs/SAGE/1970-01-01T00:00:00"
        ),
    )
    group.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        action="append",
        dest="checkpoints",
        required=True,
        default=[],
        metavar="INT",
        help=("checkpoint at which to evaluate the model \\[repeatable]"),
    )

    #####################
    # INFERENCE OPTIONS #
    #####################
    group = parser.add_argument_group(
        title="Inference options",
        description=(
            "Options marked as repeatable may be specified multiple times and "
            "will cause the evaluation to be executed once for each specified value. "
            "For repeatable options, the default value applies only if no value is "
            "provided."
        ),
    )
    group.add_argument(
        "-n",
        "--num_vars",
        type=int,
        action="append",
        default=[],
        metavar="INT",
        help=(
            "number of variables in the evaluation instances "
            "\\[repeatable, default: 100]"
        ),
    )
    group.add_argument(
        "-a",
        "--alpha",
        type=float,
        action="append",
        dest="alphas",
        default=[],
        metavar="FLOAT",
        help=(
            "ratio of clauses to variables in the evaluation instances "
            "\\[repeatable, default: 4.2]"
        ),
    )
    group.add_argument(
        "--complexify",
        type=float,
        action="append",
        help=(
            "instead of generating new instances from templates, augment random k-sat "
            "instances; the value specified corresponds to the percentage of splits "
            "which should be performed \\[repeatable, default: do not complexify]"
        ),
    )
    group.add_argument(
        "--num_sampled_pairs",
        type=int,
        action="append",
        metavar="INT",
        help=(
            "number of clause pairs sampled by the oracle "
            "\\[repeatable, default: the same value used during training]"
        ),
    )
    group.add_argument(
        "--multinomial_templates",
        action="store_true",
        help=(
            "sample the templates from a multinomial distribution instead of a "
            "triangular distribution"
        ),
    )

    ################
    # EVAL OPTIONS #
    ################
    group = parser.add_argument_group(
        title="Evaluation options",
    )

    group.add_argument(
        "--runs",
        type=int,
        default=100,
        metavar="INT",
        help=(
            "how many instances to generate and evaluate; if any of the "
            "repeatable parameters are specified, this is the number of "
            "instances generated for each combination of the repeatable "
            "parameters \\[default: 100]"
        ),
    )
    group.add_argument(
        "--solver",
        type=str,
        action="append",
        dest="solvers",
        metavar="SOLVER",
        help=(
            "which solver to use; can be any solver accepted by PySAT "
            "(see `pysat.solvers.SolverNames` for a full list of accepted "
            "solver names) \\[repeatable, default: minisat22]"
        ),
    )

    #################
    # OTHER OPTIONS #
    #################
    group = parser.add_argument_group(
        title="Other options",
    )

    group.add_argument(
        "-o",
        "--output",
        type=str,
        default="eval.parquet",
        metavar="STR",
        help=(
            "name of the output file, which will be saved in the results_dir "
            "\\[default: eval.parquet]"
        ),
    )
    group.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        metavar="INT",
        help="number of evaluation processes to run in parallel [default: 1]",
    )
    group.add_argument(
        "--save_instances",
        action="store_true",
        help=(
            "save the generated instances in the output file; in augmentation mode, "
            "also save the original instances"
        ),
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="seed for all random number generators \\[default: 0]",
    )
    group.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "device in which to run the neural network model "
            "\\[default: the same device used during training]"
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

    args = parser.parse_args(namespace=Args())

    if not args.num_vars:
        args.num_vars = [100]
    if not args.alphas:
        args.alphas = [4.2]
    if not args.solvers:
        args.solvers = ["minisat22"]

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

    solvers: list["Solver"] = [PySAT(name) for name in args.solvers]

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
                args.complexify,
                args.runs,
                args.num_cpus,
                args.save_instances,
                args.multinomial_templates,
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
