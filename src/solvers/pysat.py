import pysat.formula
import pysat.solvers

from .base import Solver


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
        clauses: list[list[int]],
    ) -> dict[str, int | float | bool]:
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
