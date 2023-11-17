#!/usr/bin/env python

import argparse
import gzip
from pathlib import Path

import numpy as np
import utils
from generation.graph import SATGraph, SplittableCNF
from rich_argparse import RichHelpFormatter
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
    multinomial: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "Generates templates and/or partial instances which can be used train "
            "a G2SAT model for the generation of hard 3-SAT instances "
            "(see the documentation of train_g2sat.py for more information)"
        ),
        formatter_class=lambda *args, **kwargs: RichHelpFormatter(
            *args, **kwargs, max_help_position=28, width=90
        ),
        add_help=False,
    )
    group = parser.add_argument_group("Main options")
    group.add_argument(
        "--out_dir",
        default="templates",
        type=Path,
        metavar="PATH",
        help=(
            "directory in which the template file should be saved "
            "\\[default: templates]"
        ),
    )
    group.add_argument(
        "--num_vars",
        type=int,
        default=100,
        metavar="INT",
        help=(
            "number of variables in the generated instances "
            "(i.e. after the templates are merged) \\[default: 100]"
        ),
    )
    group.add_argument(
        "--num_clauses",
        type=int,
        default=420,
        metavar="INT",
        help=(
            "number of clauses in the generated instances "
            "(i.e. after the templates are merged) \\[default: 100]"
        ),
    )
    group.add_argument(
        "--k",
        type=int,
        default=3,
        metavar="INT",
        help=(
            "number of literals in each clause in the generated instances "
            "(i.e. after the templates are merged) \\[default: 3]"
        ),
    )
    group.add_argument(
        "--num_templates",
        type=int,
        default=50000,
        metavar="INT",
        help="number of templates to generate \\[default: 50_000]",
    )
    group.add_argument(
        "--multinomial",
        action="store_true",
        help=(
            "sample the templates from a multinomial distribution instead of a "
            "triangular distribution"
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
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )

    group = parser.add_argument_group(
        title="Instance augmentation",
        description=(
            "By default this script generates random templates for the task of "
            "instance generation. However, if either of the flags --complexify_min or "
            "--complexify_max are specified, this script switchs to generating partial "
            "instances, for the task of instance augmentation/complexification. "
            "To this, a set of random k-sat instances is generated. For each "
            "instance, a random number of splits is performed. The percentage of "
            "splits performed in each instance, relative the the maximum number "
            "of splits possible, is sampled from the interval \\[complexify_min, "
            "complexify_max]."
        ),
    )
    group.add_argument(
        "--complexify_min",
        type=float,
        default=None,
        help=(
            "Perform instance augmentation and ensure that the fraction of splits "
            "performed is at least --complexify_min \\[default: 0.0]"
        ),
    )
    group.add_argument(
        "--complexify_max",
        type=float,
        default=None,
        help=(
            "Perform instance augmentation and ensure that the fraction of splits "
            "performed is at least --complexify_max \\[default: 1.0]"
        ),
    )
    args = parser.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.complexify_min is None and args.complexify_max is None:
        complexify = None
        note = ""
    else:
        cmin = 0.0 if args.complexify_min is None else args.complexify_min
        cmax = 1.0 if args.complexify_max is None else args.complexify_max
        complexify = (cmin, cmax)
        note = f"({cmin},{cmax})"

    if complexify is None and args.multinomial:
        note = "(multinomial)"

    filename = (
        f"{args.num_vars}x{args.num_clauses}x{args.k}{note}_"
        f"{args.num_templates}_seed{args.seed}.txt.gz"
    )
    out_file = args.out_dir / filename
    print("Writing templates to file:", out_file)

    with gzip.open(out_file, mode="wt") as f:
        for _ in trange(args.num_templates, unit="templates"):
            if complexify is None:
                t = SATGraph.sample_template(
                    args.num_vars,
                    args.num_clauses * args.k,
                    multinomial=args.multinomial,
                    seed=rng,
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
