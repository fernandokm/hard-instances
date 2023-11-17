import itertools
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
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
        fixed_templates: Sequence[npt.NDArray[np.int64] | list[list[int]]]
        | None = None,
        solve_repetitions: int = 1,
        solve_agg: str = "mean",
        reference_instances: list[list[list[int]]] | None = None,
        normalize_by_reference: bool = False,
    ) -> None:
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.solver = solver
        self.reward_metric = reward_metric
        self.num_sampled_pairs = num_sampled_pairs
        self.compress_observations = compress_observations
        self.intermediate_rewards = intermediate_rewards
        self.intermediate_rewards_coeff = intermediate_rewards_coeff
        self.sampling_method: SamplingMethod = sampling_method
        self.allow_overlaps = allow_overlaps
        self.solve_repetitions = solve_repetitions
        self.solve_agg = solve_agg
        self.reference_instances = reference_instances
        self.normalize_by_reference = normalize_by_reference

        self.fixed_templates = fixed_templates
        if fixed_templates is None:
            self.fixed_templates_iter = None
        else:
            self.fixed_templates_iter = iter(itertools.cycle(fixed_templates))

        max_num_nodes = self._compute_max_node_count()
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

    def _compute_max_node_count(self) -> int:
        max_num_nodes = self.num_vars * 2 + self.num_clauses * 3
        if self.fixed_templates is None:
            return max_num_nodes

        for template in self.fixed_templates:
            if isinstance(template, list):
                num_vars = max(
                    abs(literal) for clause in template for literal in clause
                )
                num_literals = 2 * num_vars
                num_clauses = sum(len(clause) for clause in template)
            else:
                num_literals = len(template)
                num_clauses = template.sum()
            num_nodes = num_literals + num_clauses
            max_num_nodes = max(max_num_nodes, num_nodes)
        return max_num_nodes

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
            metrics = self._get_metrics()
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

    def _solve_instance(self, instance: list[list[int]]) -> dict[str, Any]:
        return self.solver.solve_instance_agg(
            instance,
            repetitions=self.solve_repetitions,
            agg_fn=self.solve_agg,
        )

    def _get_metrics(self) -> dict[str, Any]:
        metrics = self._solve_instance(self.graph.to_clauses())
        if self.reference_instances:
            ref_lists = defaultdict(list)
            for instance in self.reference_instances:
                ref = self._solve_instance(instance)
                for k, v_ref in ref.items():
                    ref_lists[k].append(v_ref)
            fn = getattr(np, self.solve_agg)
            for k, v_refs in ref_lists.items():
                metrics[f"ref/{k}"] = fn(v_refs)
                if self.normalize_by_reference:
                    # Add epsilon to avoid divide-by-zero warnings
                    metrics[k] /= metrics[f"ref/{k}"] + sys.float_info.epsilon
        return metrics

    def reset(
        self,
        template: npt.NDArray[np.int64] | list[list[int]] | None = None,
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

        if isinstance(template, list):
            self.graph = SATGraph.from_clauses(
                template,
                sampling_method=self.sampling_method,
                allow_overlaps=self.allow_overlaps,
                seed=rng,
            )
        else:
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
