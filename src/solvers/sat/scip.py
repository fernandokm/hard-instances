import tempfile
from typing import TextIO, TypedDict

import numpy as np
from pyscipopt import SCIP_PARAMEMPHASIS, Model, quicksum
from solvers.base import Solver


class Result(TypedDict):
    feasible: bool
    runtime: float
    model: Model


class SCIP(Solver):
    @staticmethod
    def convert_to_cnf(clauses: np.ndarray, out: TextIO):
        num_clauses, num_vars = clauses.shape
        out.write(f"p cnf {num_vars} {num_clauses}\n")
        for i in range(num_clauses):
            (variable_idxs,) = np.nonzero(clauses[i])
            # Add one because variables are 1-indexed in cnf files
            variables = (variable_idxs + 1) * clauses[i, variable_idxs]
            for v in variables:
                out.write(f"{v} ")
            out.write("0\n")

    def solve_instance(
        self,
        clauses: np.ndarray,
    ) -> Result:
        model = Model()
        model.hideOutput()
        with tempfile.NamedTemporaryFile("w+", suffix=".cnf") as f:
            SCIP.convert_to_cnf(clauses, out=f)
            f.flush()
            model.readProblem(f.name)

        model.optimize()

        return Result(
            feasible=model.getStatus() == "optimal",
            runtime=model.getSolvingTime(),
            model=model,
        )
