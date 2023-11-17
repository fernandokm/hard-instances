#!/usr/bin/env python

import argparse
import socket
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torch_geometric as tg
import utils
from generation.envs import G2SATEnv
from generation.generators import G2SATPolicy, ReinforceTrainer, callbacks
from generation.graph import SamplingMethod
from gnn_models.sage import SAGE
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter
from solvers.pysat import PySAT
from tensorboardX import SummaryWriter
from torch import optim


class Args(argparse.Namespace):
    gpu: bool
    gpu_device: str
    num_vars: int
    num_clauses: int
    num_layers: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    gamma: float
    lr: float
    lr_decay: float
    num_episodes: int
    num_sampled_pairs: int
    solve_repetitions: int
    solve_agg: str
    intermediate_rewards: bool
    sampling_method: SamplingMethod
    allow_overlaps: bool
    compress_observations: bool
    action_mode: Literal["sample", "argmax"]
    template_file: Path | None
    eval_freq: int
    eval_files: list[Path]
    eval_repetitions: int
    eval_agg: str
    reference_instances: Path | None
    normalize_by_reference: bool
    checkpoint_freq: int | None
    metric: str
    seed: int
    tensorboard: bool


def parse_args() -> Args:
    RichHelpFormatter.highlights.append(r"(?P<groups>Input Files)")
    parser = argparse.ArgumentParser(
        description=(
            "Trains a G2SAT model adapted for the generation of hard 3-SAT instances"
        ),
        formatter_class=lambda *args, **kwargs: RichHelpFormatter(
            *args, **kwargs, max_help_position=28, width=90
        ),
        add_help=False,
    )

    #############
    # INFERENCE #
    #############
    group = parser.add_argument_group(
        title="Inference Configuration",
        description=(
            "Configure how the instances are generated (inference). "
            "These hyperparameters can be changed during inference."
        ),
    )
    group.add_argument(
        "--num_vars",
        default=50,
        type=int,
        metavar="INT",
        help=(
            "number of variables in the generated instances, used to "
            "sample templates when --template_file is not specified \\[default: 50]"
        ),
    )
    group.add_argument(
        "--num_clauses",
        default=210,
        type=int,
        metavar="INT",
        help=(
            "number of clauses in the generated instances, used to "
            "sample templates when --template_file is not specified \\[default: 210]"
        ),
    )
    group.add_argument(
        "--num_sampled_pairs",
        default=2_000,
        type=int,
        metavar="INT",
        help="number of clause pairs sampled by the oracle \\[default: 2000]",
    )
    group.add_argument(
        "--allow_overlaps",
        action="store_true",
        help="allow overlapping clauses in the oracle \\[default: false]",
    )
    group.add_argument(
        "--action_mode",
        choices=["sample", "argmax"],
        default="argmax",
        metavar="MODE",
        help=(
            "how the gnn logits should be converted into a single action, "
            "either `sample` or `argmax`"
            "\\[default: `argmax`]"
        ),
    )
    group.add_argument(
        "--sampling_method",
        choices=["g2sat", "uniform"],
        default="g2sat",
        metavar="METHOD",
        help=(
            "sampling method used in the oracle: `uniform` uses uniform sampling, "
            "`g2sat` uses the optimized algorithm found in the g2sat code "
            "\\[default: `g2sat`]"
        ),
    )

    #########
    # MODEL #
    #########
    group = parser.add_argument_group(
        title="Model Configuration",
        description="Neural network architecture (hyperparameters for the SAGE GNN).",
    )
    group.add_argument(
        "--num_layers",
        default=3,
        type=int,
        metavar="INT",
        help="number of GNN layers \\[default: 3]",
    )
    group.add_argument(
        "--feature_dim",
        default=32,
        type=int,
        metavar="INT",
        help="dimension of the node features after encoding \\[default: 32]",
    )
    group.add_argument(
        "--hidden_dim",
        default=32,
        type=int,
        metavar="INT",
        help="output dimension of the hidden GNN layers \\[default: 32]",
    )
    group.add_argument(
        "--output_dim",
        default=32,
        type=int,
        metavar="INT",
        help="output dimension of final GNN layer \\[default: 32]",
    )
    group.add_argument(
        "--compress_observations",
        action="store_true",
        help=(
            "encode the node features (0/1/2 indicating positive/negative literals and "
            "clauses) using an embedding layer, instead of the dense layer used "
            "by default in SAGE \\[default: false]"
        ),
    )

    ############
    # TRAINING #
    ############
    group = parser.add_argument_group(
        title="Training Configuration",
        description=(
            "Configure how the model is trained. "
            "These hyperparameters have no effect during inference."
        ),
    )
    group.add_argument(
        "--gamma",
        default=0.999,
        type=float,
        metavar="FLOAT",
        help="discount factor \\[default: 0.999]",
    )
    group.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        metavar="FLOAT",
        help="learning rate \\[default: 1e-3]",
    )
    group.add_argument(
        "--lr_decay",
        default=1,
        type=float,
        metavar="FLOAT",
        help="learning rate exponential decay factor, per episode \\[default: 1]",
    )
    group.add_argument(
        "--num_episodes",
        default=20_000,
        type=int,
        metavar="INT",
        help="number of training episodes \\[default: 20_000]",
    )
    group.add_argument(
        "--solve_repetitions",
        default=1,
        type=int,
        metavar="INT",
        help=(
            "how many times to solve each generated instance, this is useful "
            "when running over time \\[default: 1]"
        ),
    )
    group.add_argument(
        "--solve_agg",
        choices=["mean", "median", "min"],
        default="median",
        metavar="AGG_FN",
        help=(
            "when --solve_repetitions > 1, how the metrics should be aggregated; "
            "either `mean`, `median` or `min`"
            "\\[default: `median`]"
        ),
    )
    group.add_argument(
        "--intermediate_rewards",
        action="store_true",
        help=(
            "give additional non-zero rewards before the final step "
            "\\[default: false]"
        ),
    )
    group.add_argument(
        "--reference_instances",
        type=Path,
        default=None,
        metavar="INSTANCE_FILE",
        help=(
            "whenever a generated instance is evaluated, evaluate also the specified "
            "reference instances (see the Input Files section for more information on "
            "file formats) \\[default: none]"
        ),
    )
    group.add_argument(
        "--normalize_by_reference",
        action="store_true",
        help=(
            "divide the training and evaluation metrics by the reference metrics; "
            "this can be useful when using time as a metric, in order to disregard "
            "low frequency effects due to other running processes, though it's "
            "unlikely to help with high frequency noise (default: false)"
        ),
    )
    group.add_argument(
        "--metric",
        choices=["time_cpu", "decisions", "conflicts", "restarts", "propagations"],
        default="time_cpu",
        help=(
            "optimisation metric / hardness measure reward used when training "
            "the model; one of: `time_cpu`, `decisions`, `conflicts`, `restarts` "
            " and `propagations` \\[default: time_cpu]"
        ),
    )
    group.add_argument(
        "--template_file",
        type=Path,
        default=None,
        metavar="TEMPLATE_FILE|INSTANCE_FILE",
        help=(
            "use these templates or instances during training instead of using "
            "random templates (see the Input Files section for more information on "
            "file formats) \\[default: none]"
        ),
    )

    ##############
    # EVALUATION #
    ##############
    group = parser.add_argument_group(
        title="Evaluation Configuration",
        description="Configure evaluation epochs during the training process.",
    )
    group.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        metavar="INT",
        help="how many training episodes to run between evaluation epochs",
    )
    group.add_argument(
        "--eval_file",
        type=Path,
        action="append",
        dest="eval_files",
        default=[],
        metavar="TEMPLATE_FILE|INSTANCE_FILE",
        help=(
            "evaluate the model on these templates or instances; if --eval_file is "
            "specified multiple times, all of the templates/instances will be used "
            "(see the Input Files section for more information on file formats) "
            "\\[default: none]"
        ),
    )
    group.add_argument(
        "--eval_repetitions",
        type=int,
        default=-1,
        metavar="INT",
        help="use a different number of --solve_repetitions during evaluation epochs",
    )
    group.add_argument(
        "--eval_agg",
        choices=["mean", "median", "min"],
        default="",
        metavar="AGG",
        help="use a different --solve_agg during evaluation epochs",
    )

    ##########
    # DEVICE #
    ##########
    group = parser.add_argument_group(
        title="Device Configuration",
        description="Configure where to execute the deep learning model (CPU / GPU)",
    )
    exclusive = group.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="use gpu \\[default: use gpu if cuda is available, otherwise use cpu]",
    )
    exclusive.add_argument(
        "--cpu",
        dest="gpu",
        action="store_false",
        help=(
            "do not use gpu \\[default: use gpu if cuda is available, "
            "otherwise use cpu]"
        ),
    )
    group.add_argument(
        "--gpu_device",
        type=str,
        default="cuda",
        metavar="DEVICE",
        help="use the specified device if running with gpu \\[default: cuda]",
    )

    ##########
    # OTHERS #
    ##########
    group = parser.add_argument_group(
        title="Others",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="seed for all random number generators \\[default: 0]",
    )
    group.add_argument(
        "--tensorboard",
        action="store_true",
        help="save results to tensorboard \\[default: false]",
    )
    group.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "save frequently to checkpoint the current model weights, in episodes "
            "\\[default: never]"
        ),
    )
    group.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )

    ###############
    # INPUT FILES #
    ###############
    input_files_description = """
        Two kinds of input files are accepted depending on the flag:

        - `TEMPLATE_FILE`: a file which stores multiple templates
            in compact form. Each line of the file contains a single
            JSON-encoded list of ints **[count(v1), count(v2), ..., count(-v1),
            count(-v2), ...]** with the number of occurrences of each
            literal in the template. These files are generated with the scripts
            `src/generate_templates.py`.
            For example, the template **x1 OR x1 OR x1 OR -x1 OR x2 or -x2 or -x2**
            corresponds to the line **[3, 1, 1, 2]**.

        - `INSTANCE_FILE`: a file which stores multiple instances
            (not necessarily templates). Each line of the file contains
            a single JSON-encoded list of lists of ints representing the
            instance. These files are generated with the script
            `src/generate_instances.py`.
            For example, the instance **(x1 AND x2) OR (NOT x1 and x3)**
            corresponds to the line **[[1, 2], [-1, 3]]**.

        Both types of files may be given in plaintext (.txt) or compressed
        plaintext (.txt.gz) format.
        """
    group = parser.add_argument_group(
        title="Input Files",
        description=Markdown(
            textwrap.dedent(input_files_description), style="argparse.text"
        ),  # type: ignore
    )

    parser.set_defaults(gpu=torch.cuda.is_available())
    args = parser.parse_args(namespace=Args())

    if args.eval_repetitions == -1:
        args.eval_repetitions = args.solve_repetitions
    if args.eval_agg == "":
        args.eval_agg = args.solve_agg

    return args


def create_logdir() -> Path:
    now = datetime.now().isoformat(timespec="seconds")
    logdir = Path("runs/SAGE") / now
    # The following loop ensures that the logdir as new by appending a
    # suffix to the directory name if it already exists (this is necessary
    # to prevent experiments executed at the same time from overwriting each other).
    i = 1
    while True:
        try:
            # we rely on the file system implementation to prevent race conditions
            logdir.mkdir(parents=True)
            break
        except FileExistsError:
            logdir = logdir.with_name(now + "-" + str(i))
            i += 1
    return logdir


def main():
    logdir = create_logdir()
    utils.Tee.save_stdout_stderr(str(logdir / "stdout.txt"))

    args = parse_args()
    print(
        "Running with args:",
        *[f"{k} = {getattr(args, k)}" for k in vars(args)],
        sep="\n  ",
    )
    print("Running on:", socket.gethostname())

    tg.seed_everything(args.seed)

    print("Logdir:", logdir)

    fixed_templates = None
    if args.template_file is not None:
        fixed_templates = utils.parse_template_file(args.template_file)
    eval_templates = []
    for file in args.eval_files:
        eval_templates += utils.parse_template_file(file)

    reference_instance = None
    if args.reference_instances:
        reference_instance = utils.parse_instance_file(args.reference_instances)
    elif args.normalize_by_reference:
        print(
            "Error: --normalize_by_reference speccified without any reference instance"
        )
        sys.exit(1)

    env_config = {
        "num_vars": args.num_vars,
        "num_clauses": args.num_clauses,
        "solver": PySAT("minisat22"),
        "reward_metric": args.metric,
        "compress_observations": args.compress_observations,
        "num_sampled_pairs": args.num_sampled_pairs,
        "allow_overlaps": args.allow_overlaps,
        "sampling_method": args.sampling_method,
        "reference_instances": reference_instance,
        "normalize_by_reference": args.normalize_by_reference,
    }
    env = G2SATEnv(
        **env_config,
        intermediate_rewards=args.intermediate_rewards,
        fixed_templates=fixed_templates,
        solve_repetitions=args.solve_repetitions,
        solve_agg=args.solve_agg,
    )
    if eval_templates:
        eval_env = G2SATEnv(
            **env_config,
            fixed_templates=eval_templates,
            solve_repetitions=args.eval_repetitions,
            solve_agg=args.eval_agg,
        )
    else:
        eval_env = None
    model = SAGE(
        input_dim=1 if env.compress_observations else 3,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_embeddings=3 if env.compress_observations else None,
        num_layers=args.num_layers,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    if args.gpu:
        model = model.to(args.gpu_device)
    policy = G2SATPolicy(model, args.num_sampled_pairs, args.compress_observations)

    cbs: list[callbacks.Callback] = [callbacks.HistoryWriter(str(logdir))]
    if args.tensorboard:
        writer = SummaryWriter(str(logdir))
        cbs.append(callbacks.Tensorboard(writer))
    if args.checkpoint_freq:
        cbs.append(
            callbacks.ModelCheckpoint(
                model,
                optimizer,
                args.checkpoint_freq,
                vars(args),
                logdir / "checkpoints",
            )
        )

    trainer = ReinforceTrainer(
        env,
        policy,
        optimizer=optimizer,
        scheduler=scheduler,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        callbacks_list=cbs,
        action_mode=args.action_mode,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()
