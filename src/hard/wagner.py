from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm, trange


class Wagner:
    net: nn.Module

    def __init__(
        self,
        individual_size: int,
        individuals_per_generation: int = 1000,
        frac_train: float = 0.93,
        frac_survival: float = 0.94,
        generations: int = 1000,
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.individual_size = individual_size
        self.individuals_per_generation = individuals_per_generation
        self.frac_train = frac_train
        self.frac_survival = frac_survival
        self.generations = generations
        self.lr = lr
        self.memory = []
        self.scores_per_generation = []

        self.reset()

    def generate(self, n: int = 1) -> torch.Tensor:
        obs = torch.zeros((n, self.individual_size * 2)).to(self.device)
        with torch.no_grad():
            for i in range(self.individual_size):
                obs[:, self.individual_size + i] = 1
                logits = self.net(obs)
                obs[:, self.individual_size + i] = 0

                dist = Bernoulli(logits=logits.view(-1))
                obs[:, i] = dist.sample()

        return obs[:, : self.individual_size]

    def train(
        self,
        score: Callable[[torch.Tensor], float],
        *,
        generations: int | None = None,
        individuals_per_generation: int | None = None
    ):
        if generations is None:
            generations = self.generations
        if individuals_per_generation is None:
            individuals_per_generation = self.individuals_per_generation

        it = trange(generations, desc="Generations")
        for _ in it:
            pop = self.generate(individuals_per_generation)
            scores = torch.as_tensor(
                [score(pop[i]) for i in range(individuals_per_generation)]
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
        self, new_individuals: torch.Tensor, new_scores: torch.Tensor
    ):
        assert new_individuals.shape[0] == new_scores.shape[0]

        for i in range(new_individuals.shape[0]):
            self.memory.append((new_individuals[i].clone(), new_scores[i].item()))

        self.memory.sort(key=lambda x: x[1], reverse=True)
        train_size = int(self.frac_train * len(self.memory))
        survival_size = int(self.frac_survival * len(self.memory))

        training_population = torch.vstack(
            [indiv for indiv, score in self.memory[:train_size]]
        )
        self._fit_crossentropy(training_population)

        self.memory = self.memory[:survival_size]

    def _fit_crossentropy(self, population: torch.Tensor):
        for i in range(self.individual_size):
            idxs = torch.arange(self.individual_size).view(1, -1).to(self.device)
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
            nn.Linear(self.individual_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        ).to(self.device)

        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)
