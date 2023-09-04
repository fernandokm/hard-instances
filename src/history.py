from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class History:
    directory: Path
    step: pd.DataFrame
    last_step: pd.DataFrame
    episode: pd.DataFrame

    def limit_episodes(self, num_episodes: int):
        self.step = self.step[self.step["episode"] < num_episodes]
        self.last_step = self.last_step[self.last_step["episode"] < num_episodes]
        self.episode = self.episode[self.episode["episode"] < num_episodes]

    @staticmethod
    def load(directory: str | Path, keep_full_steps: bool = False) -> "History":
        if not isinstance(directory, Path):
            directory = Path(directory)

        step = _read_sorted(directory / "history_step", index_cols=["episode", "step"])
        episode = _read_sorted(directory / "history_episode", index_cols=["episode"])

        last_step = step.reset_index(level="step")
        last_step = last_step.loc[~last_step.index.duplicated(keep="last")]

        episode["num_steps"] = last_step["step"] + 1

        if not keep_full_steps:
            step = pd.DataFrame(columns=step.columns)

        return History(
            directory=directory,
            step=step,
            last_step=last_step,
            episode=episode,
        )


def _read_sorted(path: Path, index_cols: str | list[str]):
    if path.with_suffix(".parquet").exists():
        df = pd.read_parquet(path.with_suffix(".parquet"))
    else:
        df = pd.read_csv(path.with_suffix(".csv"))

    df.set_index(index_cols, inplace=True)
    if not df.index.is_monotonic_increasing:
        # Check first if the values are already sorted
        # Since all history files are supposed to be already sorted,
        # we should always be able to avoid sorting
        df.sort_index(inplace=True)

    return df
