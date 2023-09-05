from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from wagner import InstanceSampler, Wagner

from .base import Solver
from .pysat import PySAT
from .utils import decode_array, encode_array

sns.set()


def generate_metric_data(solver: Solver, metric: str, size: tuple[int, int]=(25, 5)):
    cache_file = (
        f".cache/test-metrics_{solver.name}_{metric}_{size[0]}x{size[1]}.parquet"
    )
    if Path(cache_file).exists():
        df = pd.read_parquet(cache_file)
        if "instance" in df.columns:
            df["instance"] = df["instance"].map(decode_array)
        return df

    w = Wagner(
        size,
        sampler=InstanceSampler(values=[-1, 0, 1]),
        lr=1e-3,
        frac_train=0.07,
        frac_survival=0.06,
        device="cuda",
    )
    results = w.train(
        solver,
        metric,
        generations=250,
        return_results=True,
        return_instances=True,
        instances_per_generation=1000,
    )
    results = pd.DataFrame(results)

    results_save = results.copy(deep=False)
    if "instance" in results_save.columns:
        results_save["instance"] = results_save["instance"].map(encode_array)
    results_save.to_parquet(cache_file, index=False)

    return results


def generate_data(solver: Solver, metrics: list[str], size: tuple[int, int]=(25, 5)):
    results = []
    for metric in metrics:
        print(f"Generating/loading data for {solver.name}/{metric}")
        r = generate_metric_data(solver, metric, size)
        r["optimized_metric"] = metric
        r["iter"] = np.arange(r.shape[0]) // 1000
        results.append(r)

    return pd.concat(results, axis=0)



minisat = PySAT("minisat22")
minisat_metrics = ["time_cpu", "restarts", "conflicts", "decisions", "propagations"]
results_small = generate_data(minisat, minisat_metrics, (25, 5))
results_large = generate_data(minisat, minisat_metrics, (1000, 200))
