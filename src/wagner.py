from typing import Any, Literal, overload

import numpy as np
import torch
from solvers.legacy.base import Solver
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import trange


class InstanceSampler:
    shape: tuple[int, ...]
    flat_size: int
    num_values: int
    values: torch.Tensor | None
    logits_dim: int
    activation: Literal["sigmoid", "softmax"]

    def __init__(
        self,
        values: list[int] | int = 2,
        activation: Literal["sigmoid", "softmax"] | None = None,
        device: str | None = None,
    ) -> None:
        if isinstance(values, int):
            self.num_values = values
            self.values = None
        else:
            self.num_values = len(values)
            self.values = torch.as_tensor(values)

        if activation == "sigmoid" and self.num_values > 2:
            msg = (
                f"Cannot use activation 'sigmoid' with more than "
                f"two instance_values: {values}"
            )
            raise ValueError(msg)

        if activation == "softmax" or self.num_values > 2:
            self.activation = "softmax"
            self.logits_dim = self.num_values
        else:
            self.activation = "sigmoid"
            self.logits_dim = 1

        if device:
            self.set_device(device)

    def set_device(self, device: str):
        if self.values is not None:
            self.values = self.values.to(device)

    def sample_logits(
        self, logits: torch.Tensor, transform_values=True
    ) -> torch.Tensor:
        if self.activation == "sigmoid":
            dist = Bernoulli(logits=logits)
        else:
            dist = Categorical(logits=logits)
        # Convert to long because bernoulli and categorical have different
        # return types
        raw_values = dist.sample().long()

        if transform_values:
            return self.transform(raw_values)
        return raw_values

    def transform(self, raw_values: torch.Tensor) -> torch.Tensor:
        if self.values is not None:
            return self.values[raw_values]
        return raw_values

    def cross_entropy(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return F.binary_cross_entropy_with_logits(preds, target)
        else:
            return F.cross_entropy(preds, target)


class Wagner:
    net: nn.Module

    def __init__(
        self,
        shape: tuple[int, ...],
        sampler: InstanceSampler | None = None,
        instances_per_epoch: int = 1000,
        frac_train: float = 0.07,
        frac_survival: float = 0.06,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.instance_shape = shape
        self.instance_shape_flat = int(np.prod(shape))
        self.sampler = sampler or InstanceSampler()
        self.instances_per_epoch = instances_per_epoch
        self.frac_train = frac_train
        self.frac_survival = frac_survival
        self.lr = lr
        self.device = device

        self.sampler.set_device(device)

        self.memory = []
        self.scores_per_generation = []

        self.reset()

    def generate(self, n: int = 1, transform_values: bool = True) -> torch.Tensor:
        with torch.no_grad():
            obs = torch.zeros((n, self.instance_shape_flat * 2), device=self.device)
            output = torch.zeros(
                (n, self.instance_shape_flat), dtype=torch.long, device=self.device
            )
            for i in range(self.instance_shape_flat):
                # Temporarily update the mask for sampling
                obs[:, self.instance_shape_flat + i] = 1
                logits = self.net(obs)
                obs[:, self.instance_shape_flat + i] = 0

                sampled = self.sampler.sample_logits(
                    logits.view(n, -1), transform_values=transform_values
                )
                obs[:, i] = sampled
                output[:, i] = sampled

            # return a copy of the tensor in order not to leak
            # the right half of obs (the mask)
            return output.view(n, *self.instance_shape)

    @overload
    def train(
        self,
        solver: Solver,
        metric: str,
        *,
        return_results: Literal[True] = True,
        return_instances: bool = True,
        generations: int = 1000,
        instances_per_generation: int | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @overload
    def train(
        self,
        solver: Solver,
        metric: str,
        *,
        return_results: Literal[False] = False,
        return_instances: bool = True,
        generations: int = 1000,
        instances_per_generation: int | None = None,
    ) -> None:
        pass

    def train(
        self,
        solver: Solver,
        metric: str,
        *,
        return_results: bool = False,
        return_instances: bool = True,
        generations: int = 1000,
        instances_per_generation: int | None = None,
    ):
        if instances_per_generation is None:
            instances_per_generation = self.instances_per_epoch
        results = [] if return_results else None

        it = trange(generations, desc="Generations")
        for _ in it:
            pop = self.generate(instances_per_generation, transform_values=False)
            scores = []
            for i in range(instances_per_generation):
                inst = self.sampler.transform(pop[i]).cpu().numpy()
                res = solver.solve_instance(inst)
                scores.append(res[metric])
                if results is not None:
                    if return_instances:
                        res["instance"] = inst
                    results.append(res)

            self.scores_per_generation.append(scores)
            self.train_on_generation(pop, scores)
            all_scores = [x[1] for x in self.memory]
            it.set_postfix(
                avg_iter_score=np.mean(scores),
                avg_pop_score=np.mean(all_scores),
                max_pop_score=max(all_scores),
                pop_size=len(all_scores),
            )

        return results

    def train_on_generation(self, new_instances: torch.Tensor, new_scores: list[float]):
        assert new_instances.shape[0] == len(new_scores)
        num_instances = new_instances.shape[0]
        new_instances = new_instances.view(num_instances, -1)

        if not ((0 <= new_instances) & (new_instances < self.sampler.num_values)).all():
            msg = (
                "The new_instances should be untransformed (i.e. their elements should"
                "be in the range [0, sampler.num_values))"
            )
            raise ValueError(msg)

        for i in range(num_instances):
            self.memory.append((new_instances[i].clone(), new_scores[i]))

        self.memory.sort(key=lambda x: x[1], reverse=True)
        train_size = int(self.frac_train * len(self.memory))
        survival_size = int(self.frac_survival * len(self.memory))

        training_population = torch.vstack(
            [inst for inst, score in self.memory[:train_size]]
        )
        self._fit_crossentropy(training_population)

        self.memory = self.memory[:survival_size]

    def _fit_crossentropy(self, population: torch.Tensor):
        pop_size = population.shape[0]
        inst_size = self.instance_shape_flat
        states = torch.zeros(
            (pop_size * inst_size, 2 * inst_size),
            device=self.device,
        )
        target = torch.zeros(
            pop_size * inst_size, dtype=population.dtype, device=self.device
        )
        for i in range(inst_size):
            start = i * pop_size
            end = (i + 1) * pop_size
            states[start:end, inst_size + i] = 1
            states[start:end, :i] = population[:, :i]
            target[start:end] = population[:, i]

        preds = self.net(states)
        loss = self.sampler.cross_entropy(preds, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def reset(self) -> None:
        self.net = nn.Sequential(
            nn.Linear(self.instance_shape_flat * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, self.sampler.logits_dim),
        ).to(self.device)

        self.net = torch.jit.script(self.net)  # type: ignore

        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)
