#!/usr/bin/env python

import numpy as np
import scipy
from solvers.sat.dpll import DPLL

s = DPLL()

def read_cnf(path: str):
    data = []
    indices = []
    indptr = [0]
    num_vars = num_clauses = 0
    with open(path) as f:
        for line in f:
            if line.startswith("c "):
                continue
            elif line.startswith("p cnf "):
                _, _, num_vars, num_clauses = line.split()
                num_vars = int(num_vars)
                num_clauses = int(num_clauses)
            else:
                for j in line.split()[:-1]:
                    j = int(j)
                    indices.append(abs(j) - 1)
                    data.append(j // abs(j))
                indptr.append(len(indices))
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(num_clauses, num_vars)
    )



res = s.solve_instance(np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]]))
print(res)

inst = read_cnf("1829.cnf")
res = s.solve_instance(inst)
print(res)
