import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import utils
from generation.graph import SATGraph


@dataclass
class History:
    directory: Path
    step: pd.DataFrame
    last_step: pd.DataFrame
    rerun: pd.DataFrame
    episode: pd.DataFrame

    def get_graph(self, episode: int, max_steps: int = sys.maxsize) -> SATGraph:
        raw_template: str = self.episode.loc[episode, "template"]  # type: ignore
        template = utils.parse_template(raw_template)
        graph = SATGraph.from_template(template)

        actions = self.step.loc[(episode, slice(None)), ["action_0", "action_1"]]
        assert (np.diff(actions.index.get_level_values("step")) == 1).all()

        for (_, step), a0, a1 in actions.itertuples(name=None):
            if step >= max_steps:
                break
            graph.merge(a0, a1)
        return graph

    @staticmethod
    def load(directory: str | Path, keep_full_steps: bool = False) -> "History":
        if not isinstance(directory, Path):
            directory = Path(directory)

        step = _read_sorted(directory / "history_step", index_cols=["episode", "step"])
        episode = _read_sorted(directory / "history_episode", index_cols=["episode"])

        last_step = step.reset_index(level="step")
        last_step = last_step.loc[~last_step.index.duplicated(keep="last")]

        episode["num_steps"] = last_step["step"] + 1

        try:
            rerun = _read_sorted(directory / "rerun", index_cols=["episode", "run"])
        except FileNotFoundError:
            rerun = pd.DataFrame()

        if not keep_full_steps:
            step = pd.DataFrame(columns=step.columns)

        return History(
            directory=directory,
            step=step,
            last_step=last_step,
            rerun=rerun,
            episode=episode,
        )


def _read_sorted(path: Path, index_cols: str | list[str]):
    full_path = path.with_suffix(".parquet")
    if full_path.exists():
        df = pd.read_parquet(full_path)
    else:
        full_path = full_path.with_suffix(".csv")
        df = pd.read_csv(full_path)

    if set(df.index.names) != set(index_cols):
        df.set_index(index_cols, inplace=True)
    if not df.index.is_monotonic_increasing:
        print(f"File {full_path} is not sorted")
        # Check first if the values are already sorted
        # Since all history files are supposed to be already sorted,
        # we should always be able to avoid sorting
        df.sort_index(inplace=True)

    return df
