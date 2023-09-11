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
    rerun: pd.DataFrame
    episode: pd.DataFrame

    def get_graph(
        self,
        episode: int,
        eval_episode: float = np.nan,
        max_steps: int = sys.maxsize,
    ) -> SATGraph:
        raw_template: str = self.episode.loc[(episode, eval_episode), "template"]  # type: ignore
        template = utils.parse_template(raw_template)
        graph = SATGraph.from_template(template)

        actions = self.step.loc[
            (episode, eval_episode, slice(None)), ["action_0", "action_1"]
        ]
        assert (np.diff(actions.index.get_level_values("step")) == 1).all()

        for (_, _, step), a0, a1 in actions.itertuples(name=None):
            if step >= max_steps:
                break
            graph.merge(a0, a1)
        return graph

    @staticmethod
    def load(directory: str | Path, load_step: bool = False) -> "History":
        if not isinstance(directory, Path):
            directory = Path(directory)

        episode = _read(
            directory / "history_episode", index_cols=["episode", "eval_episode"]
        )
        have_episode_metrics = any(
            col.startswith("metrics/") for col in episode.columns
        )

        if load_step or not have_episode_metrics:
            step = _read(
                directory / "history_step",
                index_cols=["episode", "eval_episode", "step"],
            )
        else:
            step = pd.DataFrame()

        if not have_episode_metrics:
            last_step = step.reset_index(level="step")
            last_step = last_step.loc[~last_step.index.duplicated(keep="last")]
            metric_cols = [
                col for col in last_step.columns if col.startswith("metrics/")
            ]
            episode.loc[:, metric_cols] = last_step.loc[:, metric_cols]

        try:
            rerun = _read(directory / "rerun", index_cols=["episode", "run"])
        except FileNotFoundError:
            rerun = pd.DataFrame()

        if not load_step:
            step = pd.DataFrame(columns=step.columns)

        return History(
            directory=directory,
            step=step,
            rerun=rerun,
            episode=episode,
        )


def _read(path: Path, index_cols: list[str]):
    full_path = path.with_suffix(".parquet")
    if full_path.exists():
        df = pd.read_parquet(full_path)
    else:
        full_path = path.with_suffix(".csv")
        if full_path.exists():
            df = pd.read_csv(full_path)
        else:
            msg = str(path.with_suffix(".{parquet,csv}"))
            raise FileNotFoundError(msg)

    if set(df.index.names) != {None}:
        df.reset_index(inplace=True)

    for col in index_cols:
        if col not in df.columns:
            df[col] = np.nan

    df.set_index(index_cols, inplace=True)

    return df
