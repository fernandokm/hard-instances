import tempfile
from typing import TextIO, TypedDict

import numpy as np
from pyscipopt import Model
from solvers.base import Solver


class _ResultNonTotal(TypedDict, total=False):
    model: Model


class Result(_ResultNonTotal, total=True):
    feasible: bool
    runtime: float
    num_nodes: int
    num_total_nodes: int
    num_leaves: int


class SCIP(Solver):
    @staticmethod
    def convert_to_cnf(clauses: np.ndarray, out: TextIO):
        num_clauses, num_vars = clauses.shape
        out.write(f"p cnf {num_vars} {num_clauses}\n")
        for i in range(num_clauses):
            (variable_idxs,) = np.nonzero(clauses[i])
            # Add one because variables are 1-indexed in cnf files
            variables = (variable_idxs + 1) * clauses[i, variable_idxs].astype(np.int32)
            for v in variables:
                out.write(f"{v} ")
            out.write("0\n")

    @staticmethod
    def solve_from_cnf(path: str, return_model: bool = False):
        model = Model()
        model.hideOutput()
        model.readProblem(path)
        model.optimize()

        res = Result(
            feasible=model.getStatus() == "optimal",
            runtime=model.getSolvingTime(),
            num_nodes=model.getNNodes(),
            num_total_nodes=model.getNTotalNodes(),
            num_leaves=model.getNLeaves(),
        )
        if return_model:
            res["model"] = model
        return res

    def solve_instance(
        self,
        clauses: np.ndarray,
        return_model: bool = False,
    ) -> Result:
        with tempfile.NamedTemporaryFile("w+", suffix=".cnf") as f:
            SCIP.convert_to_cnf(clauses, out=f)
            f.flush()
            return SCIP.solve_from_cnf(f.name, return_model=return_model)
