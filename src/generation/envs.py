from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from solvers.base import Solver

from .graph import SATGraph


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
    ) -> None:
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.solver = solver
        self.reward_metric = reward_metric
        self.num_sampled_pairs = num_sampled_pairs
        self.compress_observations = compress_observations
        self.intermediate_rewards = intermediate_rewards
        self.intermediate_rewards_coeff = intermediate_rewards_coeff

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
        action: npt.NDArray[np.integer],
    ) -> tuple[G2SATObservation, float, bool, bool, dict[str, Any]]:
        clause1, clause2 = action

        if self.graph.node_type[clause1] != 2 or self.graph.node_type[clause2] != 2:
            msg = f"Invalid action {action}: one of the merged nodes is not a clause"
            raise ValueError(msg)
        if self.graph.have_overlapping_vars(clause1, clause2):
            msg = f"Invalid action {action}: merged nodes have repeated variables"
            raise ValueError(msg)

        self.graph.merge(clause1, clause2)

        obs = self.get_obs()

        is_3sat = self.graph.is_3sat()
        if is_3sat or self.intermediate_rewards:
            info = self.solver.solve_instance(self.graph)
            reward = info[self.reward_metric]
            if not is_3sat:
                reward *= self.intermediate_rewards_coeff
        else:
            info = {}
            reward = 0

        terminated = is_3sat or len(obs["valid_actions"]) == 0
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        template: npt.NDArray[np.int64] | None = None,
        seed: int | None = None,
    ) -> tuple[G2SATObservation, dict[str, Any]]:
        super().reset(seed=seed)
        if template is None:
            template = SATGraph.sample_template(
                self.num_vars, self.num_clauses * 3, self.np_random
            )

        self.graph = SATGraph.from_template(template)

        return self.get_obs(), {}

    def render(self) -> None:
        self.graph.plot_nx()

    def get_obs(self) -> G2SATObservation:
        return {
            "graph": self.graph.to_graph_instance(self.compress_observations),
            "valid_actions": self.graph.sample_valid_merges(self.num_sampled_pairs),
            # "valid_actions": self.graph.get_valid_merges(),
            "total_valid_actions": self.graph.count_valid_merges(),
        }
