from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm, trange

from solvers.base import Solver


class Wagner:
    net: nn.Module

    def __init__(
        self,
        instance_shape: tuple[int, ...],
        instances_per_epoch: int = 1000,
        frac_train: float = 0.93,
        frac_survival: float = 0.94,
        epochs: int = 1000,
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.instance_shape = instance_shape
        self.instances_per_epoch = instances_per_epoch
        self.frac_train = frac_train
        self.frac_survival = frac_survival
        self.epochs = epochs
        self.lr = lr
        self.memory = []
        self.scores_per_generation = []

        self.instance_size_flat = 1
        for dim in self.instance_shape:
            self.instance_size_flat *= dim

        self.reset()

    def generate(self, n: int = 1) -> torch.Tensor:
        obs = torch.zeros((n, self.instance_size_flat * 2)).to(self.device)
        with torch.no_grad():
            for i in range(self.instance_size_flat):
                obs[:, self.instance_size_flat + i] = 1
                logits = self.net(obs)
                obs[:, self.instance_size_flat + i] = 0

                dist = Bernoulli(logits=logits.view(-1))
                obs[:, i] = dist.sample()

        return obs[:, : self.instance_size_flat].view((-1, *self.instance_shape))

    def train(
        self,
        solver: Solver,
        metric: str,
        *,
        generations: int | None = None,
        instances_per_generation: int | None = None
    ):
        if generations is None:
            generations = self.epochs
        if instances_per_generation is None:
            instances_per_generation = self.instances_per_epoch

        it = trange(generations, desc="Generations")
        for _ in it:
            pop = self.generate(instances_per_generation)
            scores = torch.as_tensor(
                [
                    solver.solve_instance(pop[i].cpu().numpy())[metric]
                    for i in range(instances_per_generation)
                ]
            )
            self.scores_per_generation.append(scores)
            self.train_on_generation(pop, scores)
            scores = [x[1] for x in self.memory]
            it.set_postfix(
                max_score=max(scores),
                moving_avg_score=np.mean(scores),
                pop_size=len(scores),
            )

    def train_on_generation(
        self, new_instances: torch.Tensor, new_scores: torch.Tensor
    ):
        assert new_instances.shape[0] == new_scores.shape[0]
        num_instances = new_instances.shape[0]
        new_instances = new_instances.view(num_instances, -1)

        for i in range(num_instances):
            self.memory.append((new_instances[i].clone(), new_scores[i].item()))

        self.memory.sort(key=lambda x: x[1], reverse=True)
        train_size = int(self.frac_train * len(self.memory))
        survival_size = int(self.frac_survival * len(self.memory))

        training_population = torch.vstack(
            [inst for inst, score in self.memory[:train_size]]
        )
        self._fit_crossentropy(training_population)

        self.memory = self.memory[:survival_size]

    def _fit_crossentropy(self, population: torch.Tensor):
        for i in range(self.instance_size_flat):
            idxs = torch.arange(self.instance_size_flat).view(1, -1).to(self.device)
            states = torch.hstack(
                [
                    population.where(idxs < i, 0),
                    (idxs == i).to(torch.float32).repeat(population.shape[0], 1),
                ]
            )
            preds = self.net(states)
            target = population[:, i]

            loss = F.binary_cross_entropy_with_logits(preds.view(-1), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def reset(self) -> None:
        self.net = nn.Sequential(
            nn.Linear(self.instance_size_flat * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        ).to(self.device)

        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)
