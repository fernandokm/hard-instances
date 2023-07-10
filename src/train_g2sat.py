#!/usr/bin/env python

import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import utils
from generation.envs import G2SATEnv
from generation.generators import G2SATPolicy, train_reinforce
from gnn_models.sage import SAGE
from solvers.pysat import PySAT
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter


class Args(argparse.Namespace):
    gpu: bool
    num_layers: int
    feature_dim: int
    hidden_dim: int
    output_dim: int
    gamma: float
    lr: float
    num_episodes: int
    num_sampled_pairs: int


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
    parser.add_argument("--gamma", default=0.99, type=float)

    parser.add_argument("--lr", dest="lr", default=1e-3, type=float)
    parser.add_argument("--num_episodes", default=20_000, type=int)
    parser.add_argument("--num_sampled_pairs", default=2_000, type=int)

    parser.set_defaults(gpu=torch.cuda.is_available())
    args = parser.parse_args(namespace=Args())
    return args


def main():
    logdir = Path("runs/SAGE") / datetime.now().isoformat(timespec="seconds")
    logdir.mkdir(parents=True, exist_ok=True)
    utils.Tee.save_stdout_stderr(str(logdir / "stdout.txt"))

    args = parse_args()
    print(
        "Running with args:",
        *[f"{k} = {getattr(args, k)}" for k in vars(args)],
        sep="\n  ",
    )

    print("Logdir:", logdir)
    writer = SummaryWriter(str(logdir))

    env = G2SATEnv(
        args.num_vars,
        args.num_clauses,
        PySAT("minisat22"),
        "time_cpu",
        compress_observations=True,
        num_sampled_pairs=args.num_sampled_pairs,
    )
    model = SAGE(
        input_dim=1 if env.compress_observations else 3,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_embeddings=3 if env.compress_observations else None,
        num_layers=args.num_layers,
    )
    if args.gpu:
        model = model.to("cuda")
    policy = G2SATPolicy(env, model, random_state=42)

    history = train_reinforce(
        policy,
        optimizer=optim.AdamW(model.parameters(), lr=args.lr),
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        writer=writer,
    )

    json.dump(history, (logdir / "history.json").open("w"))


if __name__ == "__main__":
    main()