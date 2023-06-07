from typing import TypedDict

import numpy as np
import pysat.solvers
from solvers.base import Solver


class Result(TypedDict):
    feasible: bool
    time_cpu: float
    restarts: int
    conflicts: int
    decisions: int
    propagations: int


class PySAT(Solver):
    def __init__(self, solver_name: str, **solver_kwargs):
        solver_kwargs.setdefault("use_timer", True)
        self.use_timer: bool = solver_kwargs["use_timer"]
        self.solver_name = solver_name
        self.solver_kwargs = solver_kwargs

    def make_pysat_solver(self, clauses: np.ndarray) -> pysat.solvers.Solver:
        cnf = []
        num_clauses, num_vars = clauses.shape
        for i in range(num_clauses):
            (variable_idxs,) = np.nonzero(clauses[i])
            if len(variable_idxs) == 0:
                continue
            # Add one because variables are 1-indexed in cnf files
            variables = (variable_idxs + 1) * clauses[i, variable_idxs].astype(np.int32)
            variables = variables.tolist()
            cnf.append(variables)
        return pysat.solvers.Solver(
            self.solver_name, **self.solver_kwargs, bootstrap_with=cnf
        )

    def solve_instance(
        self,
        clauses: np.ndarray,
    ) -> Result:
        with self.make_pysat_solver(clauses) as solver:
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
