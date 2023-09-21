import gzip
import json
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt
from pysat.formula import CNF

Seed = int | np.random.Generator | None


class Tee:
    def __init__(self, dst1, dst2):
        self.dst1 = dst1
        self.dst2 = dst2

    def write(self, data):
        self.dst1.write(data)
        self.dst2.write(data)

    def flush(self):
        self.dst1.flush()
        self.dst2.flush()

    @staticmethod
    def save_stdout_stderr(filename: str, mode="w+"):
        f = open(filename, mode)
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)


def parse_template(raw_template: str) -> npt.NDArray[np.int64]:
    return np.asarray(json.loads(raw_template), dtype=np.int64)


def _open_and_decompress(file: Path) -> TextIO:
    if file.suffix.lower() == ".gz":
        return gzip.open(file, "rt")
    else:
        return file.open("r")


def parse_template_file(file: Path) -> list[npt.NDArray[np.int64]]:
    templates = []
    with _open_and_decompress(file) as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(parse_template(line))
    return templates


def parse_instance_file(file: Path) -> list[list[list[int]]]:
    instances = []
    with _open_and_decompress(file) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def random_k_sat(
    rng: np.random.Generator,
    n_vars: int,
    n_clauses: int,
    k: int = 3,
) -> CNF:
    variables = rng.choice(1 + np.arange(n_vars), size=(n_clauses, k))
    polarities = rng.choice([1, -1], size=(n_clauses, k))
    literals = variables * polarities
    return CNF(from_clauses=literals.tolist())
