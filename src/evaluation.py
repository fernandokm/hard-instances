import queue
import random

import pandas as pd
import torch
import utils
from generation.generators import G2SATPolicy
from generation.graph import SATGraph
from gnn_models.sage import SAGE
from solvers.base import Solver
from torch import multiprocessing
from tqdm.auto import tqdm


def load_policy(path: str, device: str | None = None) -> G2SATPolicy:
    data = torch.load(path)

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
    )


def generate(
    policy: G2SATPolicy, num_vars: int, alpha: float, seed: utils.Seed = None
) -> SATGraph:
    template = SATGraph.sample_template(num_vars, int(num_vars * 3 * alpha), seed=seed)
    instance = policy.generate(template, seed=seed)
    return instance


def generate_and_eval(
    policy: G2SATPolicy,
    solver: Solver,
    num_vars: int,
    alpha: float,
    seed: utils.Seed = None,
):
    instance = generate(policy, num_vars, alpha, seed=seed)
    r = solver.solve_instance(instance.to_clauses())
    r["num_vars"] = num_vars
    r["alpha"] = alpha
    return r


def _handle_eval_queue(
    worker_idx: int,
    policy: G2SATPolicy,
    solver: Solver,
    in_queue: multiprocessing.Queue,
    out_queue: multiprocessing.Queue,
) -> None:
    # print(f"[{worker_idx}] Start", flush=True)
    while True:
        # print(f"[{worker_idx}] Looking (size={in_queue.qsize()})", flush=True)
        try:
            num_vars, alpha, run, seed = in_queue.get_nowait()
        except queue.Empty:
            # print(f"[{worker_idx}] Done", flush=True)
            return
        # print(f"[{worker_idx}] Got task ({num_vars}, {alpha})", flush=True)
        r = generate_and_eval(policy, solver, num_vars, alpha, seed)
        r["run"] = run
        out_queue.put(r)


def generate_and_eval_par(
    policy: G2SATPolicy,
    solver: Solver,
    num_vars: list[int],
    alphas: list[float],
    runs: int,
    num_cpus: int = 1,
    seed: utils.Seed = None,
    desc="generating and solving instances",
) -> pd.DataFrame:
    rng_factory = utils.RngFactory(seed)
    spawn_ctx = multiprocessing.get_context("spawn")
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
    ctx = multiprocessing.spawn(
        _handle_eval_queue,
        (policy, solver, in_queue, out_queue),
        nprocs=num_cpus,
        join=False,
    )
    in_queue.close()
    pbar = tqdm(total=num_tasks, desc="generating and solving instances")
    while pbar.n < pbar.total:
        r = out_queue.get()
        results.append(r)
        pbar.update()
    pbar.refresh()

    ctx.join()
    return pd.DataFrame(results).sort_values(["num_vars", "alpha", "run"])
