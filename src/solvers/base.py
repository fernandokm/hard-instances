from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Value = TypeVar("Value")


class Solver(ABC, Generic[Value]):
    @abstractmethod
    def solve_instance(
        self,
        instance: list[list[int]],
    ) -> dict[str, Value]:
        pass

    @property
    def name(self):
        return self.__class__.__name__
