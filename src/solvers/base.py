from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

Key = TypeVar("Key")
Value = TypeVar("Value")
AggregatedValue = TypeVar("AggregatedValue")


class Solver(ABC, Generic[Value]):
    @abstractmethod
    def solve_instance(
        self,
        instance: list[list[int]],
    ) -> dict[str, Value]:
        pass

    def solve_instance_agg(
        self,
        instance: list[list[int]],
        repetitions: int = 1,
        agg_fn: str | Callable[[list[Value]], AggregatedValue] = "mean",
    ) -> dict[str, AggregatedValue]:
        results = [self.solve_instance(instance) for _ in range(repetitions)]

        # convert the list[dict[str, Value]] to a dict[str, list[Value]]
        transposed = defaultdict(list)
        for r in results:
            for k, v in r.items():
                transposed[k].append(v)

        # apply the aggregation function to each list[Value]
        fn = getattr(np, agg_fn) if isinstance(agg_fn, str) else agg_fn
        aggregated = {}
        for k, v in transposed.items():
            aggregated[k] = fn(v)

        return aggregated

    @property
    def name(self):
        return self.__class__.__name__
