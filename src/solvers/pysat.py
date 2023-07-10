from typing import TYPE_CHECKING, TypedDict

import pysat.solvers

from .base import Solver

if TYPE_CHECKING:
    from generation.graph import SATGraph


class Result(TypedDict):
    feasible: bool
    time_cpu: float
    restarts: int
    conflicts: int
    decisions: int
    propagations: int


class PySAT(Solver):
    def __init__(self, solver_name: str):
        self.solver_name = solver_name

    def make_pysat_solver(self, instance: "SATGraph") -> pysat.solvers.Solver:
        clauses = instance.to_clauses()
        return pysat.solvers.Solver(
            self.solver_name,
            use_timer=True,
            bootstrap_with=clauses,
        )

    def solve_instance(
        self,
        instance: "SATGraph",
    ) -> Result:
        with self.make_pysat_solver(instance) as solver:
            feasible = solver.solve()
            return Result(
                feasible=feasible,
                time_cpu=solver.time(),
                **solver.accum_stats(),  # type: ignore
            )

    @property
    def name(self):
        base_name = super().name
        return f"{base_name}({self.solver_name})"
