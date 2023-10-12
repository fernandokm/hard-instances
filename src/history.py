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
    reruns: dict[str, pd.DataFrame]
    episode: pd.DataFrame
    evaluation: pd.DataFrame

    def get_graph(
        self,
        episode: int,
        eval_episode: float = np.nan,
        max_steps: int = sys.maxsize,
    ) -> SATGraph:
        raw_template = self.episode.loc[(episode, eval_episode), "template"]
        template = utils.parse_template(raw_template)  # type: ignore
        if isinstance(template, list):
            graph = SATGraph.from_clauses(template)
        else:
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
    def load(
        directory: str | Path,
        load_step: bool = False,
        load_reruns: bool = False,
        load_eval: bool = False,
    ) -> "History":
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

        if not load_step:
            step = pd.DataFrame(columns=step.columns)

        reruns = {}
        if load_reruns:
            rerun_dirs = {
                *directory.glob("rerun*.parquet"),
                *directory.glob("rerun*.csv"),
            }
            for rerun_dir in rerun_dirs:
                reruns[rerun_dir.stem] = _read(rerun_dir, index_cols=["episode", "run"])

        if load_eval:
            eval_ = _read(directory / "eval", index_cols=[], allow_missing=True)
        else:
            eval_ = pd.DataFrame()

        # Update old column names:
        episode.rename(
            columns={"timing/logger": "timing/callbacks"}, inplace=True, errors="ignore"
        )
        step.rename(
            columns={"timing/logger": "timing/callbacks"}, inplace=True, errors="ignore"
        )

        return History(
            directory=directory,
            step=step,
            reruns=reruns,
            episode=episode,
            evaluation=eval_,
        )


def _read(path: Path, index_cols: list[str], allow_missing: bool=False):
    full_path = path.with_suffix(".parquet")
    if full_path.exists():
        df = pd.read_parquet(full_path)
    else:
        full_path = path.with_suffix(".csv")
        if full_path.exists():
            df = pd.read_csv(full_path)
        elif allow_missing:
            return pd.DataFrame()
        else:
            msg = str(path.with_suffix(".{parquet,csv}"))
            raise FileNotFoundError(msg)

    if not index_cols:
        return df

    if set(df.index.names) != {None}:
        df.reset_index(inplace=True)

    for col in index_cols:
        if col not in df.columns:
            df[col] = np.nan

    df.set_index(index_cols, inplace=True)

    return df
