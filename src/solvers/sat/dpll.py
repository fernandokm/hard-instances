import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from solvers.base import Solver


class Result(TypedDict):
    feasible: bool
    wallclock_time: float
    cpu_time: float
    solution: list[bool | None]
    num_backtracks: int


@dataclass(slots=True)
class Variable:
    num: int
    assignment: bool | None
    watched_pos: list["Clause"]
    watched_neg: list["Clause"]


@dataclass(slots=True)
class Literal:
    var: Variable
    polarity: bool

    def is_false(self):
        return self.var.assignment == (not self.polarity)

    def assign(self) -> bool:
        if self.var.assignment == self.polarity:
            return True
        elif self.var.assignment == (not self.polarity):
            return False

        self.var.assignment = self.polarity
        falsified = self.var.watched_neg if self.polarity else self.var.watched_pos
        for c in falsified:
            c.update_watched_literals()
            if c.is_false():
                self.var.assignment = None
                return False
        return True


@dataclass(slots=True)
class Clause:
    literals: list[Literal]
    watched_1: Literal
    watched_2: Literal

    def __init__(self, literals):
        self.literals = literals
        self.watched_1 = self._watch_new(literals[0])
        self.watched_2 = self._watch_new(literals[-1], excluded=[self.watched_1])

    def is_false(self):
        return self.watched_1.is_false() and self.watched_2.is_false()

    def get_unit_literal(self) -> Literal | None:
        w1_false = self.watched_1.is_false()
        w2_false = self.watched_2.is_false()
        if w1_false and not w2_false:
            return self.watched_2
        if w2_false and not w1_false:
            return self.watched_1
        if not w1_false and self.watched_1 == self.watched_2:
            return self.watched_1

    def update_watched_literals(self):
        if self.watched_1.is_false():
            self.watched_1 = self._watch_new(
                excluded=(self.watched_1, self.watched_2), default=self.watched_1
            )
        if self.watched_2.is_false():
            self.watched_2 = self._watch_new(
                excluded=(self.watched_1, self.watched_2), default=self.watched_2
            )

    def size(self) -> int:
        return sum(lit.var.assignment is None for lit in self.literals)

    def _watch_new(self, default: Literal, excluded: Sequence[Literal] = []) -> Literal:
        for lit in self.literals:
            if lit in excluded or lit.is_false():
                continue
            if lit.polarity:
                lit.var.watched_pos.append(self)
            else:
                lit.var.watched_neg.append(self)
            return lit
        return default


class DPLL(Solver):
    def solve_instance(
        self,
        instance: np.ndarray,
    ) -> Result:
        variables = [Variable(i, None, [], []) for i in range(instance.shape[1])]
        clauses = []
        for i in range(instance.shape[0]):
            literals = []
            for j in instance[i].nonzero()[0]:
                lit = Literal(variables[j], polarity=instance[i, j] > 0)
                literals.append(lit)
            if literals:
                clauses.append(Clause(literals))

        self._start = time.perf_counter()
        self._start_cpu = time.process_time()
        self._backtracks = 0
        return self._dpll(clauses, variables)

    def _make_result(self, feasible: bool, variables: list[Variable]):
        now = time.perf_counter()
        now_cpu = time.process_time()

        return Result(
            feasible=feasible,
            wallclock_time=now - self._start,
            cpu_time=now_cpu - self._start_cpu,
            solution=[var.assignment for var in variables],
            num_backtracks=self._backtracks,
        )

    def _dpll(
        self,
        clauses: list[Clause],
        variables: list[Variable],
        depth=0,
    ) -> Result:
        # Unit propagation
        found_unit = True
        while found_unit:
            found_unit = False
            for c in clauses:
                lit = c.get_unit_literal()
                if lit is not None and lit.var.assignment is None:
                    if not lit.assign():
                        return self._make_result(False, variables)
                    found_unit = True
        if all(var.assignment is not None for var in variables):
            return self._make_result(True, variables)

        chosen_var = variables[0]
        for var in variables:
            if var.assignment is None:
                chosen_var = var
                break

        if Literal(chosen_var, True).assign():
            assignments = [var.assignment for var in variables]
            result_new = self._dpll(clauses, variables, depth + 1)
            if result_new["feasible"]:
                return result_new
            for var, assign in zip(variables, assignments):
                var.assignment = assign
            chosen_var.assignment = None
            self._backtracks += 1

        if Literal(chosen_var, False).assign():
            return self._dpll(clauses, variables, depth + 1)

        return self._make_result(False, variables)
