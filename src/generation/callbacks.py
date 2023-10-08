import copy
import csv
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm.auto import tqdm


class Callback:
    def __init__(self) -> None:
        pass

    def on_step(self, info: dict) -> None:
        pass

    def on_episode_start(self, evaluation: bool = False) -> None:
        pass

    def on_episode_end(self, info: dict) -> None:
        pass

    def close(self) -> None:
        pass


class WithTiming(Callback):
    def __init__(self, inner: Callback):
        self.inner = inner
        self.time = defaultdict(float)

    def on_episode_start(self, evaluation: bool) -> None:
        return self.inner.on_episode_start(evaluation)

    def _with_time(self, fn, info: dict):
        t0 = time.monotonic()
        result = fn(info)
        t1 = time.monotonic()
        self.time["callbacks"] += t1 - t0

        return result

    def _update_episode_time(self, info: dict):
        for k, v in info.get("timing", {}).items():
            self.time[k] += v

    def on_step(self, info: dict) -> None:
        self._update_episode_time(info)
        return self._with_time(self.inner.on_step, info)

    def on_episode_end(self, info: dict) -> None:
        self._update_episode_time(info)
        info["timing"] = dict(self.time)
        self.time.clear()

        return self._with_time(self.inner.on_episode_end, info)

    def close(self) -> None:
        self.inner.close()


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def on_episode_start(self, evaluation: bool = False) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(evaluation)

    def on_step(self, info: dict) -> None:
        for callback in self.callbacks:
            callback.on_step(info)

    def on_episode_end(self, info: dict) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(info)

    def close(self):
        for callback in self.callbacks:
            callback.close()

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def has_type(self, callback_type: type[Callback]) -> bool:
        return any(isinstance(callback, callback_type) for callback in self.callbacks)


class Tqdm(Callback):
    def __init__(
        self,
        metrics: list[str] | None = None,
        num_episodes: float | None = None,
        **tqdm_kwargs,
    ) -> None:
        if metrics is None:
            metrics = []
        tqdm_kwargs.setdefault("desc", "Training")
        tqdm_kwargs.setdefault("unit", "episodes")
        tqdm_kwargs.setdefault("smoothing", 0.0002)
        self.pbar = tqdm(total=num_episodes, **tqdm_kwargs)
        self.metrics = metrics.copy()
        self._postfix = {}
        self._eval_episodes = 0
        self._evaluation = False

    def on_episode_start(self, evaluation: bool = False) -> None:
        self._evaluation = evaluation

    def on_step(self, info: dict) -> None:
        pass

    def on_episode_end(self, info: dict) -> None:
        if self._evaluation:
            self._eval_episodes += 1
            self._postfix["eval_episodes"] = self._eval_episodes
            self._evaluation = False
            self.pbar.set_postfix(self._postfix, refresh=True)
            return
        for k in self.metrics:
            if k in info:
                self._postfix[k] = info[k]
        self.pbar.set_postfix(self._postfix, refresh=False)
        self.pbar.update()
        self._postfix["eval_episodes"] = 0

    def close(self):
        self.pbar.close()


class Tensorboard(Callback):
    def __init__(self, writer: SummaryWriter | str) -> None:
        if isinstance(writer, SummaryWriter):
            self.writer = writer
        else:
            self.writer = SummaryWriter(writer)

        self._num_steps = 0
        self._num_episodes = 0
        self._num_eval_episodes = 0
        self._num_eval_steps = 0
        self._evaluation = False

    def on_episode_start(self, evaluation: bool = False) -> None:
        self._evaluation = evaluation

    def on_step(self, info: dict) -> None:
        self._write_scalars(info, is_step=True)
        if self._evaluation:
            self._num_eval_steps += 1
        else:
            self._num_steps += 1

    def on_episode_end(self, info: dict) -> None:
        self._write_scalars(info, is_step=False)
        if self._evaluation:
            self._num_eval_episodes += 1
        else:
            self._num_episodes += 1

    def _write_scalars(self, data: dict, is_step: bool) -> None:
        data = _flatten_dict(data)
        if is_step:
            global_step = self._num_eval_steps if self._evaluation else self._num_steps
            suffix = "_step"
        else:
            global_step = (
                self._num_eval_episodes if self._evaluation else self._num_episodes
            )
            suffix = "_ep"

        for k, v in data.items():
            if isinstance(v, int | float | np.number):
                if self._evaluation:
                    k = "eval/" + k
                k += suffix
                self.writer.add_scalar(k, v, global_step)


class HistoryWriter(Callback):
    def __init__(self, outdir: str, convert_to_parquet: bool = True) -> None:
        self.outdir = outdir
        self.out_step_raw = open(outdir + "/history_step.csv", "w")
        self.out_step = _AutoDictWriter(self.out_step_raw)
        self.out_episode_raw = open(outdir + "/history_episode.csv", "w")
        self.out_episode = _AutoDictWriter(self.out_episode_raw)
        self.convert_to_parquet = convert_to_parquet

        self._steps = []
        self._episode = 0
        self._eval_episode = 0
        self._evaluation = False

    def on_episode_start(self, evaluation: bool = False) -> None:
        self._evaluation = evaluation

    def on_step(self, info: dict) -> None:
        info = _flatten_dict(info, normalize_types=True)
        info["step"] = len(self._steps)
        info["episode"] = self._episode
        info["eval_episode"] = self._eval_episode if self._evaluation else None

        self._steps.append(info)

    def on_episode_end(self, info: dict) -> None:
        info = _flatten_dict(info, normalize_types=True)
        info["episode"] = self._episode
        info["eval_episode"] = self._eval_episode if self._evaluation else None

        self.out_episode.writerows([info])
        self.out_step.writerows(self._steps)
        self.out_episode_raw.flush()
        self.out_step_raw.flush()

        self._steps.clear()
        if self._evaluation:
            self._eval_episode += 1  # type: ignore
        else:
            self._episode += 1
            self._eval_episode = 0

    @staticmethod
    def _convert_to_parquet(name: str):
        path_csv = name + ".csv"
        path_parquet = name + ".parquet"

        data = pd.read_csv(path_csv)
        data.to_parquet(path_parquet)
        os.remove(path_csv)

    def close(self):
        self.out_step_raw.close()
        self.out_episode_raw.close()
        if self.convert_to_parquet:
            self._convert_to_parquet(self.outdir + "/history_step")
            self._convert_to_parquet(self.outdir + "/history_episode")


class _AutoDictWriter:
    def __init__(self, fp):
        self.fp = fp
        self.inner = None
        self.fieldnames = set()

    def writerows(self, rowdicts: list[dict]):
        keys = set().union(*(d.keys() for d in rowdicts))
        if self.inner is None:
            self.fieldnames = keys
            self.inner = csv.DictWriter(
                self.fp, fieldnames=sorted(keys), extrasaction="ignore"
            )
            self.inner.writeheader()
        else:
            extra_keys = keys - self.fieldnames
            if extra_keys:
                extra_keys_str = ", ".join(extra_keys)
                print(
                    "Warning: keys not present in initial iteration will not be "
                    f"saved ({extra_keys_str})"
                )
        self.inner.writerows(rowdicts)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        freq: int,
        hparams: dict[str, Any],
        outdir: str | Path,
    ) -> None:
        hparams = copy.deepcopy(hparams)
        if not _picklable(hparams):
            msg = f"Hparams object not picklable: {hparams}"
            raise ValueError(msg)

        self.model = model
        self.optimizer = optimizer
        self.checkpoint_freq = freq
        self.hparams = hparams
        self.outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

        self._evaluation = False
        self._episode = 0
        self._latest_checkpoint = None

    def on_episode_start(self, evaluation: bool = False) -> None:
        self._evaluation = evaluation
        if not evaluation and self._episode % self.checkpoint_freq == 0:
            self.save_checkpoint()

    def on_episode_end(self, info: dict) -> None:
        if not self._evaluation:
            self._episode += 1

    def save_checkpoint(self):
        if self._latest_checkpoint == self._episode:
            return
        self._latest_checkpoint = self._episode
        data = {
            "episode": self._episode,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            **self.hparams,
        }
        ckpt_file = self.outdir / f"{self._episode}.pt"
        torch.save(data, ckpt_file)

    def close(self):
        self.save_checkpoint()


def setup_callbacks(
    callbacks: list[Callback] | None,
    num_episodes: int,
    default_tqdm_metrics: list[str] | None = None,
    with_timing: bool = True,
) -> Callback:
    cb = CallbackList(callbacks or [])

    if not cb.has_type(Tqdm):
        cb.append(Tqdm(default_tqdm_metrics, num_episodes=num_episodes))

    if with_timing:
        cb = WithTiming(cb)
    return cb


def _flatten_dict(
    d: dict, prefix: str = "", out: dict | None = None, normalize_types: bool = False
) -> dict:
    if out is None:
        out = {}
    for k, v in d.items():
        k = prefix + k
        if isinstance(v, dict):
            _flatten_dict(v, prefix=k + "/", out=out, normalize_types=normalize_types)
        elif normalize_types and isinstance(v, tuple):
            for i, vi in enumerate(v):
                out[f"{k}_{i}"] = vi
        elif normalize_types and isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif normalize_types and isinstance(v, bool | np.bool_):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _picklable(obj) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except pickle.PicklingError:
        return False
