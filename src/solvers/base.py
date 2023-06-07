import os
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class Solver(ABC):
    @abstractmethod
    def solve_instance(
        self,
        instance: "np.ndarray",
    ) -> dict[str, Any]:
        pass

    @property
    def name(self):
        return self.__class__.__name__
