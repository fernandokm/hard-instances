from typing import Any, Literal, overload

import numpy as np
import torch
from solvers.base import Solver
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import trange


class Wagner:
    net: nn.Module

    def __init__(
        self,
        instance_shape: tuple[int, ...],
        instance_values: list[int] | int = 2,
        instances_per_epoch: int = 1000,
        frac_train: float = 0.07,
        frac_survival: float = 0.06,
        epochs: int = 1000,
        lr: float = 1e-4,
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

        if isinstance(instance_values, int):
            self.instance_values_dim = instance_values
            self.instance_values = None
        else:
            self.instance_values_dim = len(instance_values)
            self.instance_values = torch.as_tensor(instance_values).to(self.device)

        self.instance_size_flat = 1
        for dim in self.instance_shape:
            self.instance_size_flat *= dim

        self.reset()

    def generate(self, n: int = 1, transform_values: bool = True) -> torch.Tensor:
        instances = []
        with torch.no_grad():
            obs = torch.zeros((n, self.instance_size_flat * 2)).to(self.device)
            for i in range(self.instance_size_flat):
                obs[:, self.instance_size_flat + i] = 1
                logits = self.net(obs)
                obs[:, self.instance_size_flat + i] = 0

                dist = Categorical(logits=logits.view(n, -1))
                sampled = dist.sample()
                if transform_values and self.instance_values is not None:
                    sampled = self.instance_values[sampled]
                instances.append(sampled)

        return torch.cat(instances).view((n, *self.instance_shape))

    @overload
    def train(
        self,
        solver: Solver,
        metric: str,
        *,
        return_results: Literal[True] = True,
        return_instances: bool = True,
        generations: int | None = None,
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
        generations: int | None = None,
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
        generations: int | None = None,
        instances_per_generation: int | None = None,
    ):
        if generations is None:
            generations = self.epochs
        if instances_per_generation is None:
            instances_per_generation = self.instances_per_epoch
        results = [] if return_results else None

        it = trange(generations, desc="Generations")
        for _ in it:
            pop = self.generate(instances_per_generation, transform_values=False)
            scores_list = []
            for i in range(instances_per_generation):
                inst = pop[i]
                if self.instance_values is not None:
                    inst = self.instance_values[inst.to(torch.int32)]
                inst_np = inst.cpu().numpy()
                res = solver.solve_instance(inst_np)
                scores_list.append(res[metric])
                if results is not None:
                    if return_instances:
                        res["instance"] = inst_np
                    results.append(res)

            scores = torch.as_tensor(scores_list)
            self.scores_per_generation.append(scores)
            self.train_on_generation(pop, scores)
            all_scores = [x[1] for x in self.memory]
            it.set_postfix(
                avg_iter_score=scores.mean(dtype=torch.float32).cpu().item(),
                avg_pop_score=np.mean(all_scores),
                max_pop_score=max(all_scores),
                pop_size=len(all_scores),
            )

        return results

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
        pop_size = population.shape[0]
        inst_size = self.instance_size_flat
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
            states[start:end, : i + 1] = population[:, : i + 1]
            target[start:end] = population[:, i]

        preds = self.net(states)
        loss = F.cross_entropy(preds, target)

        self.optimizer.zero_grad(set_to_none=True)
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
            nn.Linear(4, self.instance_values_dim),
        ).to(self.device)

        self.net = torch.jit.script(self.net)

        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)
