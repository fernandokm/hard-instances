import itertools
import time
from collections import defaultdict
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from solvers.base import Solver
from utils import Seed

from .graph import SamplingMethod, SATGraph


class G2SATObservation(TypedDict):
    graph: spaces.GraphInstance
    valid_actions: list[tuple[int, int]]
    total_valid_actions: int


class G2SATEnv(gym.Env[dict, npt.NDArray[np.integer]]):
    def __init__(
        self,
        num_vars: int,
        num_clauses: int,
        solver: Solver,
        reward_metric: str,
        num_sampled_pairs: int = 2_000,
        compress_observations: bool = True,
        intermediate_rewards: bool = False,
        intermediate_rewards_coeff: float = 0.1,
        sampling_method: SamplingMethod = "g2sat",
        allow_overlaps: bool = False,
        fixed_templates: list[npt.NDArray[np.int64]] | None = None,
        solve_repetitions: int = 1,
        solve_agg: str = "mean",
    ) -> None:
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.solver = solver
        self.reward_metric = reward_metric
        self.num_sampled_pairs = num_sampled_pairs
        self.compress_observations = compress_observations
        self.intermediate_rewards = intermediate_rewards
        self.intermediate_rewards_coeff = intermediate_rewards_coeff
        self.sampling_method = sampling_method
        self.allow_overlaps = allow_overlaps
        self.solve_repetitions = solve_repetitions
        self.solve_agg = solve_agg

        self.fixed_templates = fixed_templates
        if fixed_templates is None:
            self.fixed_templates_iter = None
        else:
            self.fixed_templates_iter = iter(itertools.cycle(fixed_templates))

        max_num_nodes = num_vars * 2 + num_clauses * 3
        self.observation_space = spaces.Dict(
            {
                "graph": spaces.Graph(
                    node_space=spaces.Discrete(2),
                    edge_space=None,
                ),
                "valid_pairs": spaces.Sequence(
                    spaces.MultiDiscrete([max_num_nodes, max_num_nodes]),
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([max_num_nodes, max_num_nodes])

    def step(
        self,
        action: npt.NDArray[np.integer] | tuple[int, int],
    ) -> tuple[G2SATObservation, float, bool, bool, dict[str, Any]]:
        t0 = time.monotonic()
        clause1, clause2 = action

        if tuple(action) not in self.valid_actions:
            msg = f"Invalid action: {action}"
            raise ValueError(msg)

        self.graph.merge(clause1, clause2)

        obs, sample_time = self.get_obs()
        self.valid_actions = obs["valid_actions"]

        is_3sat = self.graph.is_3sat()
        terminated = is_3sat or len(obs["valid_actions"]) == 0
        truncated = False

        if terminated or self.intermediate_rewards:
            metrics_raw = defaultdict(list)
            for _ in range(self.solve_repetitions):
                for k, v in self.solver.solve_instance(self.graph).items():
                    metrics_raw[k].append(v)
            agg_fn = getattr(np, self.solve_agg)
            metrics = {}
            for k in list(metrics_raw.keys()):
                metrics[k] = agg_fn(metrics_raw[k])

            reward = metrics[self.reward_metric]
            if not terminated:
                reward *= self.intermediate_rewards_coeff
        else:
            metrics = {}
            reward = 0

        t1 = time.monotonic()
        info = {
            "is_3sat": is_3sat,
            "metrics": metrics,
            "timing": {
                "step": t1 - t0,
                "sample_pairs": sample_time,
            },
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        template: npt.NDArray[np.int64] | None = None,
        seed: Seed = None,
    ) -> tuple[G2SATObservation, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        if template is not None:
            # ok, already have template
            pass
        elif self.fixed_templates_iter is not None:
            # get next fixed template
            template = next(self.fixed_templates_iter)
        else:
            # sample a new template
            template = SATGraph.sample_template(
                self.num_vars, self.num_clauses * 3, seed=rng
            )

        self.graph = SATGraph.from_template(
            template,
            sampling_method=self.sampling_method,
            allow_overlaps=self.allow_overlaps,
            seed=rng,
        )
        obs, sample_time = self.get_obs()
        self.valid_actions = obs["valid_actions"]

        info = {
            "template": template,
            "timing": {
                "sample_pairs": sample_time,
            },
        }
        return obs, info

    def render(self) -> None:
        self.graph.plot_nx()

    def get_obs(self) -> tuple[G2SATObservation, float]:
        t0 = time.monotonic()
        valid, count = self.graph.sample_valid_merges_with_count(self.num_sampled_pairs)
        t1 = time.monotonic()
        return {
            "graph": self.graph.to_graph_instance(self.compress_observations),
            "valid_actions": valid,
            "total_valid_actions": count,
        }, t1 - t0
