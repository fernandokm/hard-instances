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

    @staticmethod
    def _convert_to_model_sat(clauses: np.ndarray) -> Model:
        model = Model()
        model.hideOutput()
        with tempfile.NamedTemporaryFile("w+", suffix=".cnf") as f:
            SCIP.convert_to_cnf(clauses, out=f)
            f.flush()
            model.readProblem(f.name)

        return model

    @staticmethod
    def _convert_to_model_lp(clauses: np.ndarray) -> Model:
        num_clauses, num_vars = clauses.shape

        model = Model()
        model.hideOutput()
        x = [model.addVar(f"x{var_idx}", "B") for var_idx in range(num_vars)]

        clause_literals = [[] for _ in range(num_clauses)]
        for clause_idx, var_idx in zip(*clauses.nonzero()):  # type: ignore
            if clauses[clause_idx, var_idx] > 0:
                lit = x[var_idx]
            else:
                lit = 1 - x[var_idx]
            clause_literals[clause_idx].append(lit)

        for literals in clause_literals:
            if literals:
                model.addCons(quicksum(literals) >= 1)

        return model

    def solve_instance(
        self,
        clauses: np.ndarray,
        pure_sat: bool = True,
    ) -> Result:
        if pure_sat:
            model = self._convert_to_model_sat(clauses)
            model.setEmphasis(SCIP_PARAMEMPHASIS.CPSOLVER)
        else:
            model = self._convert_to_model_lp(clauses)
        model.optimize()

        return Result(
            feasible=model.getStatus() == "optimal",
            runtime=model.getSolvingTime(),
            model=model,
        )
