from typing import TextIO
import numpy as np


def read_cnf(path: str) -> np.ndarray:
    clauses = np.array([])
    i = 0
    with open(path) as f:
        for line in f:
            if line.startswith("c "):
                continue
            elif line.startswith("p cnf "):
                _, _, num_vars, num_clauses = line.split()
                clauses = np.zeros((int(num_clauses), int(num_vars)), dtype=int)
            else:
                for j in line.split()[:-1]:
                    j = int(j)
                    clauses[i, abs(j) - 1] = j // abs(j)
                i += 1
    return clauses


def write_cnf(clauses: np.ndarray, out: TextIO | str):
    if isinstance(out, str):
        out = open(out, "w")
    num_clauses, num_vars = clauses.shape
    out.write(f"p cnf {num_vars} {num_clauses}\n")
    for i in range(num_clauses):
        (variable_idxs,) = np.nonzero(clauses[i])
        if len(variable_idxs) == 0:
            continue
        # Add one because variables are 1-indexed in cnf files
        variables = (variable_idxs + 1) * clauses[i, variable_idxs].astype(np.int32)
        for v in variables:
            out.write(f"{v} ")
        out.write("0\n")
