import gzip
import json
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt

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
