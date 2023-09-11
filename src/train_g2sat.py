#!/usr/bin/env python

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torch_geometric as tg
import utils
from generation.envs import G2SATEnv
from generation.generators import G2SATPolicy, ReinforceTrainer, logging
from generation.graph import SamplingMethod
from gnn_models.sage import SAGE
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
    metric: str
    seed: int
    tensorboard: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="use gpu (default: True if cuda is found)",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="cuda",
        help="use the specified device if running with gpu (default: cuda)",
    )
    parser.add_argument(
        "--cpu",
        dest="gpu",
        action="store_false",
        help="use cpu (default: False unless cuda is found)",
    )

    parser.add_argument("--num_vars", default=100, type=int)
    parser.add_argument("--num_clauses", default=420, type=int)

    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--feature_dim", default=32, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--output_dim", default=32, type=int)
    parser.add_argument("--gamma", default=0.999, type=float)

    parser.add_argument("--lr", dest="lr", default=1e-3, type=float)
    parser.add_argument("--num_episodes", default=20_000, type=int)
    parser.add_argument("--num_sampled_pairs", default=2_000, type=int)
    parser.add_argument("--solve_repetitions", default=1, type=int)
    parser.add_argument(
        "--solve_agg", choices=["mean", "median", "min"], default="median"
    )
    parser.add_argument("--intermediate_rewards", action="store_true")
    parser.add_argument("--allow_overlaps", action="store_true")
    parser.add_argument(
        "--sampling_method", choices=["g2sat", "uniform"], default="g2sat"
    )
    parser.add_argument("--action_mode", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--template_file", type=Path, default=None)
    parser.add_argument("--compress_observations", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument(
        "--eval_file", type=Path, action="append", dest="eval_files", default=[]
    )
    parser.add_argument("--eval_repetitions", type=int, default=-1)
    parser.add_argument("--eval_agg", choices=["mean", "median", "min"], default="")
    parser.add_argument(
        "--metric",
        choices=["time_cpu", "decisions", "conflicts", "restarts", "propagations"],
        default="time_cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")

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

    tg.seed_everything(args.seed)

    print("Logdir:", logdir)

    fixed_templates = None
    if args.template_file is not None:
        fixed_templates = utils.parse_template_file(args.template_file)
    eval_templates = []
    for file in args.eval_files:
        eval_templates += utils.parse_template_file(file)

    env_config = {
        "num_vars": args.num_vars,
        "num_clauses": args.num_clauses,
        "solver": PySAT("minisat22"),
        "reward_metric": args.metric,
        "compress_observations": args.compress_observations,
        "num_sampled_pairs": args.num_sampled_pairs,
        "intermediate_rewards": args.intermediate_rewards,
        "allow_overlaps": args.allow_overlaps,
        "sampling_method": args.sampling_method,
    }
    env = G2SATEnv(
        **env_config,
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
    if args.gpu:
        model = model.to(args.gpu_device)
    policy = G2SATPolicy(env, model)

    loggers: list[logging.Logger] = [logging.FileLogger(str(logdir))]
    if args.tensorboard:
        writer = SummaryWriter(str(logdir))
        loggers.append(logging.TensorboardLogger(writer))

    trainer = ReinforceTrainer(
        policy,
        optimizer=optim.AdamW(model.parameters(), lr=args.lr),
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        loggers=loggers,
        action_mode=args.action_mode,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()
