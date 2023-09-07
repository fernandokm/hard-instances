from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from generation.graph import SATGraph

Value = TypeVar("Value")


class Solver(ABC, Generic[Value]):
    @abstractmethod
    def solve_instance(
        self,
        instance: "SATGraph",
    ) -> dict[str, Value]:
        pass

    @property
    def name(self):
        return self.__class__.__name__
