from typing import TYPE_CHECKING

import pysat.formula
import pysat.solvers

from .base import Solver

if TYPE_CHECKING:
    from generation.graph import SATGraph


class PySAT(Solver[int | float | bool]):
    def __init__(self, solver_name: str):
        self.solver_name = solver_name

    def make_pysat_solver(self, clauses: list[list[int]]) -> pysat.solvers.Solver:
        return pysat.solvers.Solver(
            self.solver_name,
            use_timer=True,
            bootstrap_with=clauses,
        )

    def solve_instance(
        self,
        instance: "SATGraph | pysat.formula.CNF",
    ) -> dict[str, int | float | bool]:
        if isinstance(instance, pysat.formula.CNF):
            clauses = instance.clauses
        else:
            clauses = instance.to_clauses()
        with self.make_pysat_solver(clauses) as solver:
            feasible = solver.solve()
            return {
                "feasible": feasible,
                "time_cpu": solver.time(),
                **solver.accum_stats(),  # type: ignore
            }

    @property
    def name(self):
        base_name = super().name
        return f"{base_name}({self.solver_name})"
