import time
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from generation import callbacks
from gymnasium.spaces import GraphInstance
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from utils import RngFactory, Seed

from .envs import G2SATEnv, G2SATObservation
from .graph import SATGraph


class G2SATPolicy:
    def __init__(
        self,
        model: nn.Module,
        num_sampled_pairs: int = 2_000,
        compress_observations: bool = True,
        allow_overlaps: bool = False,
    ) -> None:
        self.model = model
        self.num_sampled_pairs = num_sampled_pairs
        self.compress_observations = compress_observations
        self.allow_overlaps = allow_overlaps

    @property
    def device(self):
        for param in self.model.parameters():
            return param.device

    def generate(
        self,
        template: npt.NDArray[np.int64] | list[list[int]] | SATGraph,
        num_sampled_pairs: int | None = None,
        compress_observations: bool | None = None,
        allow_overlaps: bool | None = None,
        seed: Seed = None,
    ) -> SATGraph:
        if num_sampled_pairs is None:
            num_sampled_pairs = self.num_sampled_pairs
        if compress_observations is None:
            compress_observations = self.compress_observations
        if allow_overlaps is None:
            allow_overlaps = self.allow_overlaps

        if isinstance(template, SATGraph):
            g = template
        elif isinstance(template, list):
            g = SATGraph.from_clauses(
                template,
                allow_overlaps=allow_overlaps,
                seed=seed,
            )
        else:
            g = SATGraph.from_template(
                template,
                allow_overlaps=allow_overlaps,
                seed=seed,
            )
        while not g.is_3sat():
            pairs = g.sample_valid_merges(num_sampled_pairs)
            if not pairs:
                break
            logits = self.predict_logits(
                g.to_graph_instance(compress_observations),
                torch.as_tensor(np.array(pairs), device=self.device),
            )
            i, j = pairs[torch.argmax(logits)]
            g.merge(i, j)

        return g

    def predict_logits(
        self, instance: GraphInstance, pairs: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.model(
            torch.as_tensor(instance.nodes, device=self.device),
            edge_index=torch.as_tensor(instance.edge_links, device=self.device),
        )
        logits = torch.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], dim=-1)
        return logits

    def sample_action(
        self,
        obs: G2SATObservation,
        action_mode: Literal["sample", "argmax"] = "argmax",
    ) -> tuple[tuple[int, int], torch.Tensor]:
        logits = self.predict_logits(
            obs["graph"],
            torch.as_tensor(obs["valid_actions"], device=self.device),
        )

        if action_mode == "sample":
            dist = Categorical(logits=logits)
            action_idx = dist.sample()

            # Approximate the probability of choosing this pair
            # as the probability that the pair was in the samples
            # and the pair was chosen from the samples
            sample_size = len(obs["valid_actions"])
            total_pairs = obs["total_valid_actions"]
            if sample_size == total_pairs:
                log_prob_pair_proposed = 0  # log(1)
            else:
                log_prob_pair_proposed = np.log(sample_size) - np.log(total_pairs)
            log_prob_pair_chosen = dist.log_prob(action_idx)
            log_prob = log_prob_pair_proposed + log_prob_pair_chosen
        else:
            action_idx = torch.argmax(logits)

            # This is the log probability of the original model, in which
            # pairs are proposed sequentially
            log_prob_pair_proposed = -np.log(obs["total_valid_actions"])
            log_prob_pair_chosen = nn.functional.logsigmoid(logits[action_idx])
            log_prob = log_prob_pair_proposed + log_prob_pair_chosen

        action = obs["valid_actions"][action_idx]

        return action, log_prob


class ReinforceTrainer:
    def __init__(
        self,
        env: G2SATEnv,
        policy: G2SATPolicy,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        num_episodes: int = 50_000,
        gamma: float = 0.99,
        eval_env: G2SATEnv | None = None,
        eval_freq: int = 1,
        action_mode: Literal["sample", "argmax"] = "argmax",
        callbacks_list: list[callbacks.Callback] | None = None,
        seed: Seed = None,
    ):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.action_mode: Literal["sample", "argmax"] = action_mode

        self.callback = callbacks.setup_callbacks(
            callbacks_list,
            num_episodes,
        )

        self.train_rng, eval_rng = np.random.default_rng(seed).spawn(2)

        # Re-use the same rng at every eval loop
        self.eval_rng_factory = RngFactory(eval_rng)

    def train(self):
        for episode in range(self.num_episodes):
            if episode % self.eval_freq == 0:
                self.evaluate()
            self.run_episode(self.env, self.train_rng, evaluation=False)
        self.evaluate()
        self.callback.close()
        self.callback = callbacks.CallbackList([])  # Dummy callback

    def evaluate(self):
        if self.eval_env is None:
            return
        if self.eval_env.fixed_templates:
            num_episodes = len(self.eval_env.fixed_templates)
        else:
            num_episodes = 1

        for _ in range(num_episodes):
            self.run_episode(
                self.eval_env, self.eval_rng_factory.make(), evaluation=True
            )

    def run_episode(
        self,
        env: G2SATEnv,
        base_rng: np.random.Generator,
        evaluation: bool = True,
    ) -> None:
        self.callback.on_episode_start(evaluation)

        log_probs = []
        rewards = []

        # We spawn a new child rng per episode in order to ensure that
        # the initial states depend only on the seed (and not on what happened during
        # the previous episodes). In particular, this ensures that the initial templates
        # used are the same across all experiments.
        obs, episode_info = env.reset(seed=base_rng.spawn(1)[0])

        terminated = False
        num_steps = 0
        while not terminated:
            t0 = time.monotonic()
            action, log_prob = self.policy.sample_action(obs, self.action_mode)
            t1 = time.monotonic()

            obs, reward, terminated, truncated, info = env.step(np.asarray(action))

            info["action"] = action
            info["reward"] = reward
            info["log_prob"] = log_prob.item()
            info["timing"]["predict"] = t1 - t0

            if terminated:
                episode_info["metrics"] = info["metrics"]

            self.callback.on_step(info)
            log_probs.append(log_prob)
            rewards.append(reward)
            num_steps += 1

        losses = []
        returns = compute_returns(rewards, self.gamma, device=self.policy.device)
        for log_prob, ret in zip(log_probs, returns, strict=True):
            losses.append(-log_prob.view(1) * ret)

        loss = torch.cat(losses).sum()
        if evaluation:
            episode_info["timing"]["train"] = 0
        else:
            t0 = time.monotonic()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t1 = time.monotonic()
            episode_info["timing"]["train"] = t1 - t0

        if self.scheduler:
            episode_info["lr"] = self.scheduler.get_last_lr()
            if isinstance(episode_info["lr"], list):
                episode_info["lr"] = episode_info["lr"][0]
            if not evaluation:
                self.scheduler.step()

        episode_info |= {
            "loss": loss.item(),
            "return/shaped": returns[0],
            "return/original": rewards[-1] * self.gamma ** (num_steps - 1),
        }
        self.callback.on_episode_end(episode_info)


def compute_returns(
    rewards: list[float], gamma: float, device: torch.device | None = None
):
    returns_rev = []
    ret = 0
    for r in rewards[::-1]:
        ret = r + gamma * ret
        returns_rev.append(ret)
    return returns_rev[::-1]
