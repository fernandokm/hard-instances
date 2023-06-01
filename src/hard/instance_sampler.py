from typing import Literal

import numpy as np
import torch
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F


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
