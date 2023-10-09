import multiprocessing
import queue
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import utils
from generation.generators import G2SATPolicy
from generation.graph import SATGraph
from gnn_models.sage import SAGE
from solvers.base import Solver
from tqdm.auto import tqdm


def load_policy(path: str, device: str | None = None) -> G2SATPolicy:
    data = torch.load(path, map_location=device)

    model = SAGE(
        input_dim=3,
        feature_dim=data["feature_dim"],
        hidden_dim=data["hidden_dim"],
        output_dim=data["output_dim"],
        num_layers=data["num_layers"],
    )
    if device:
        model = model.to(device)
    model.load_state_dict(data["model_state_dict"])
    return G2SATPolicy(
        model,
        num_sampled_pairs=data["num_sampled_pairs"],
        compress_observations=data["compress_observations"],
        allow_overlaps=data["allow_overlaps"]
    )


def generate(
    policy: G2SATPolicy, num_vars: int, alpha: float, seed: utils.Seed = None
) -> SATGraph:
    template = SATGraph.sample_template(num_vars, int(num_vars * 3 * alpha), seed=seed)
    instance = policy.generate(template, seed=seed)
    return instance


def generate_and_eval(
    policy: G2SATPolicy,
    solvers: list[Solver],
    num_vars: int,
    alpha: float,
    seed: utils.Seed = None,
) -> Iterable[dict[str, Any]]:
    instance = generate(policy, num_vars, alpha, seed=seed)
    clauses = instance.to_clauses()
    for solver in solvers:
        r = solver.solve_instance(clauses)
        r["num_vars"] = num_vars
        r["alpha"] = alpha
        r["solver"] = solver.name
        r["num_sampled_pairs"] = policy.num_sampled_pairs
        yield r


def _handle_eval_queue(
    worker_idx: int,
    policies: list[G2SATPolicy],
    solvers: list[Solver],
    in_queue: torch.multiprocessing.Queue,
    out_queue: torch.multiprocessing.Queue,
) -> None:
    policy = policies[worker_idx % len(policies)]
    # policy.num_sampled_pairs = 2_000
    while True:
        try:
            num_vars, alpha, run, seed = in_queue.get_nowait()
        except queue.Empty:
            return
        for r in generate_and_eval(policy, solvers, num_vars, alpha, seed):
            r["run"] = run
            out_queue.put(r)


def generate_and_eval_par(
    policies: list[G2SATPolicy],
    solvers: list[Solver],
    num_vars: list[int],
    alphas: list[float],
    runs: int,
    num_cpus: int = 1,
    seed: utils.Seed = None,
    desc="generating and solving instances",
) -> pd.DataFrame:
    rng_factory = utils.RngFactory(seed)
    spawn_ctx = torch.multiprocessing.get_context("spawn")
    tasks = []
    for n in num_vars:
        for a in alphas:
            # Use the same sequence of rngs for each (n,a) pair
            rng = rng_factory.make()
            for run in range(runs):
                tasks.append((n, a, run, rng.spawn(1)[0]))

    # We shuffle the tasks to ensure that the harder tasks (those with larger n,a)
    # are not all at the end of the tasks (this gives us better tqdm time estimates)
    random.shuffle(tasks)
    in_queue = spawn_ctx.Queue()
    out_queue = spawn_ctx.Queue()
    for t in tasks:
        in_queue.put_nowait(t)

    results = []
    num_tasks = in_queue.qsize()
    ctx = torch.multiprocessing.spawn(
        _handle_eval_queue,
        (policies, solvers, in_queue, out_queue),
        nprocs=num_cpus,
        join=False,
    )
    in_queue.close()
    pbar = tqdm(total=num_tasks * len(solvers), desc=desc, smoothing=0.005)
    while pbar.n < pbar.total:
        r = out_queue.get()
        results.append(r)
        pbar.update()
    pbar.close()

    ctx.join()
    return pd.DataFrame(results).sort_values(["num_vars", "alpha", "run"])


def _random_ksat_init(solvers_):
    global solvers
    solvers = solvers_


def _random_ksat_solve(args: tuple[int, float, int, utils.Seed]):
    num_vars, alpha, run, seed = args
    cnf = utils.random_k_sat(
        rng=np.random.default_rng(seed),
        n_vars=num_vars,
        n_clauses=int(num_vars * alpha),
    )
    results = []
    for solver in solvers:
        r = solver.solve_instance(cnf.clauses)
        r["num_vars"] = num_vars
        r["alpha"] = alpha
        r["solver"] = solver.name
        r["run"] = run
        results.append(r)
    return results


def generate_and_eval_ksat_par(
    solvers: list[Solver],
    num_vars: list[int],
    alphas: list[float],
    runs: int,
    num_cpus: int = 1,
    seed: utils.Seed = None,
    desc="generating and solving instances",
) -> pd.DataFrame:
    rng_factory = utils.RngFactory(seed)
    tasks = []
    for n in num_vars:
        for a in alphas:
            # Use the same sequence of rngs for each (n,a) pair
            rng = rng_factory.make()
            for run in range(runs):
                tasks.append((n, a, run, rng.spawn(1)[0]))

    # We shuffle the tasks to ensure that the harder tasks (those with larger n,a)
    # are not all at the end of the tasks (this gives us better tqdm time estimates)
    random.shuffle(tasks)

    results = []
    with multiprocessing.Pool(num_cpus, _random_ksat_init, (solvers,)) as pool:
        it = tqdm(
            pool.imap_unordered(_random_ksat_solve, tasks),
            total=len(tasks),
            desc=desc,
            smoothing=0.001,
        )
        for result in it:
            results += result

    return pd.DataFrame(results).sort_values(["num_vars", "alpha", "run"])
