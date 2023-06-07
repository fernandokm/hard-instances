import contextlib
import io
import re
import tempfile
import time
from collections.abc import Callable, Collection
from typing import TypedDict

import numpy as np
from pyscipopt import Model
from solvers.base import Solver

from . import convert_to_cnf

StatsDict = dict[str, int | float]
StatsTable = dict[str, dict[str, int | float]]

SolverStats = TypedDict(
    "SolverStats",
    {
        "Total Time": StatsDict,
        "Original Problem": StatsDict,
        "Presolved Problem": StatsDict,
        "Presolvers": StatsTable,
        "Constraints": StatsTable,
        "Constraint Timings": StatsTable,
        "Propagators": StatsTable,
        "Propagator Timings": StatsTable,
        "Conflict Analysis": StatsTable,
        "Separators": StatsTable,
        "Cutselectors": StatsTable,
        "Pricers": StatsTable,
        "Branching Rules": StatsTable,
        "Primal Heuristics": StatsTable,
        "LP": StatsTable,
        "B&B Tree": StatsDict,
        "Root Node": StatsDict,
        "Solution": StatsDict,
        "Integrals": StatsTable,
    },
    total=False,
)

GROUPS = set(SolverStats.__annotations__.keys())


def get_stats_parser(
    group_name: str,
) -> Callable[[str], StatsDict | StatsTable]:
    solvers = {
        StatsDict: _parse_numeric,
        StatsTable: _parse_table,
    }
    return solvers[SolverStats.__annotations__[group_name]]


def parse_stats(stats: str, groups: Collection[str] = GROUPS) -> SolverStats:
    raw_groups = re.split(r"\n(?! )", stats.strip())[1:]

    result = SolverStats()
    for group in raw_groups:
        group_name = group.split(":", maxsplit=1)[0].strip()
        if group_name in groups:
            result[group_name] = get_stats_parser(group_name)(group)
    return result


class _ResultNonTotal(SolverStats, total=False):
    model: Model


class Result(_ResultNonTotal, total=True):
    feasible: bool
    time_internal: float
    time_wallclock: float
    time_cpu: float


class SCIP(Solver):
    def __init__(
        self, return_model: bool = False, stats_groups: Collection[str] = GROUPS
    ):
        self.return_model = return_model
        self.stats_groups = stats_groups

    def solve_from_cnf(self, path: str) -> Result:
        model = Model()
        model.redirectOutput()
        model.hideOutput()
        model.readProblem(path)

        start = time.perf_counter()
        start_cpu = time.process_time()
        model.optimize()
        end = time.perf_counter()
        end_cpu = time.process_time()

        if len(self.stats_groups) > 0:
            model.hideOutput(False)
            s = io.StringIO()
            with contextlib.redirect_stdout(s):
                model.printStatistics()
            stats = s.getvalue()
        else:
            stats = ""

        res = Result(
            feasible=model.getStatus() == "optimal",
            time_internal=model.getSolvingTime(),
            time_wallclock=end - start,
            time_cpu=end_cpu - start_cpu,
            **parse_stats(stats, self.stats_groups),
        )
        if self.return_model:
            res["model"] = model
        return res

    def solve_instance(self, clauses: np.ndarray) -> Result:
        with tempfile.NamedTemporaryFile("w+", suffix=".cnf") as f:
            convert_to_cnf(clauses, out=f)  # type: ignore
            f.flush()
            return self.solve_from_cnf(f.name)


def _parse_number(raw: str) -> int | float:
    if raw == "-":
        return np.nan
    if "." in raw:
        return float(raw)
    return int(raw.removesuffix("+"))


def _parse_numeric(raw: str) -> StatsDict:
    parsed = {}
    for line in raw.splitlines():
        key, val = line.split(":")
        val_match = re.search(r"^\d+(?:\.\d+)?", val.strip())
        if val_match is None:
            continue

        key = key.strip()
        val = _parse_number(val_match.group(0))
        parsed[key] = val
    return parsed


def _parse_table(raw: str) -> StatsTable:
    def clean_line(line: str) -> str:
        line = line.strip()
        # Strip parentheticals at the end of the line
        if line.endswith(")"):
            return line[: line.rindex("(")].strip()
        return line

    lines = [clean_line(line) for line in raw.splitlines()]

    _, raw_columns = lines[0].split(":")
    if raw.lstrip().startswith("Conflict Analysis"):
        raw_columns = raw_columns.replace("LP Iters", "LPIters")
    columns = raw_columns.split()
    table = {col: {} for col in columns}

    for line in lines[1:]:
        idx, vals = line.split(":")
        idx = idx.strip()
        for col, val in zip(columns, vals.split()):
            col = col.strip()
            table[col][idx] = _parse_number(val)

    return table
