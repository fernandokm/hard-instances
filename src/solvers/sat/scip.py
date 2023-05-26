from os import environ
from typing import TypedDict

import numpy as np
from pyscipopt import SCIP_PARAMEMPHASIS, Model, quicksum
from solvers.base import Solver

environ["LDFLAGS"] = "-Wl,-rpath,/usr/lib"


class Result(TypedDict):
    feasible: bool
    runtime: float
    model: Model


class SCIP(Solver):
    @staticmethod
    def _convert_to_model_sat(clauses: np.ndarray) -> Model:
        num_clauses, num_vars = clauses.shape

        model = Model()
        x = [model.addVar(f"x{var_idx}", "B") for var_idx in range(num_vars)]

        clause_literals = [[] for _ in range(num_clauses)]
        signs = [[] for _ in range(num_clauses)]
        for clause_idx, var_idx in zip(*clauses.nonzero()):  # type: ignore
            clause_literals[clause_idx].append(x[var_idx])
            signs[clause_idx].append(clauses[clause_idx, var_idx])

        for literals, s in zip(clause_literals, signs):
            if literals:
                model.addConsLogicor(literals, s)

        return model

    @staticmethod
    def _convert_to_model_lp(clauses: np.ndarray) -> Model:
        num_clauses, num_vars = clauses.shape

        model = Model()
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
        pure_sat: bool = False,
    ) -> Result:
        if pure_sat:
            model = self._convert_to_model_sat(clauses)
            model.setEmphasis(SCIP_PARAMEMPHASIS.CPSOLVER)
        else:
            model = self._convert_to_model_lp(clauses)
        model.hideOutput()
        model.optimize()

        return Result(
            feasible=model.getStatus() == "optimal",
            runtime=model.getSolvingTime(),
            model=model,
        )
