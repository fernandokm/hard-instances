import csv
import time
from collections import defaultdict

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm


class Logger:
    def set_num_episodes(self, n: int) -> None:
        pass

    def step(self, info: dict) -> None:
        pass

    def end_episode(self, info: dict) -> None:
        pass


class WithTiming(Logger):
    def __init__(self, inner: Logger):
        self.inner = inner
        self.time = defaultdict(float)

    def set_num_episodes(self, n: int) -> None:
        return self.inner.set_num_episodes(n)

    def _with_time(self, fn, info: dict):
        t0 = time.monotonic()
        result = fn(info)
        t1 = time.monotonic()
        self.time["logger"] += t1 - t0

        return result

    def _update_episode_time(self, info: dict):
        for k, v in info.get("timing", {}).items():
            self.time[k] += v

    def step(self, info: dict) -> None:
        self._update_episode_time(info)
        return self._with_time(self.inner.step, info)

    def end_episode(self, info: dict) -> None:
        self._update_episode_time(info)
        info["timing"] = dict(self.time)
        self.time.clear()

        return self._with_time(self.inner.end_episode, info)


class LoggerList(Logger):
    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers

    def set_num_episodes(self, n: int) -> None:
        for logger in self.loggers:
            logger.set_num_episodes(n)

    def step(self, info: dict) -> None:
        for logger in self.loggers:
            logger.step(info)

    def end_episode(self, info: dict) -> None:
        for logger in self.loggers:
            logger.end_episode(info)

    def append(self, logger: Logger) -> None:
        self.loggers.append(logger)

    def has_type(self, logger_type: type[Logger]) -> bool:
        return any(isinstance(logger, logger_type) for logger in self.loggers)


class TqdmLogger(Logger):
    def __init__(self, metrics: list[str] = ["loss"], **tqdm_kwargs) -> None:
        tqdm_kwargs.setdefault("desc", "Training")
        tqdm_kwargs.setdefault("unit", "episodes")
        self.pbar = tqdm(**tqdm_kwargs)
        self.metrics = metrics.copy()
        self._postfix = {}
        self._steps = 0

    def set_num_episodes(self, n: int) -> None:
        self.pbar.total = n

    def step(self, info: dict) -> None:
        self._steps += 1
        self._postfix["steps"] = self._steps
        self.pbar.set_postfix(self._postfix, refresh=False)

    def end_episode(self, info: dict) -> None:
        self._postfix = {"step": self._steps}
        for k in self.metrics:
            if k in info:
                self._postfix[k] = info[k]
        self.pbar.set_postfix(self._postfix, refresh=False)
        self.pbar.update()

    def __del__(self):
        self.pbar.close()


class TensorboardLogger(Logger):
    def __init__(self, writer: SummaryWriter | str) -> None:
        if isinstance(writer, SummaryWriter):
            self.writer = writer
        else:
            self.writer = SummaryWriter(writer)

        self._last_step = {}
        self._num_steps = 0
        self._num_episodes = 0

    def step(self, info: dict) -> None:
        self._write_scalars(info, global_step=self._num_steps, suffix="_step")
        self._last_step_info = info
        self._num_steps += 1

    def end_episode(self, info: dict) -> None:
        self._write_scalars(info, global_step=self._num_episodes, suffix="_ep")
        self._write_scalars(
            self._last_step_info, global_step=self._num_episodes, suffix="_ep"
        )
        self._num_episodes += 1

    def _write_scalars(self, data: dict, global_step, suffix: str = "") -> None:
        data = _flatten_dict(data)
        for k, v in data.items():
            if isinstance(v, int | float | np.number):
                k += suffix
                self.writer.add_scalar(k, v, global_step)


class CsvLogger(Logger):
    class AutoDictWriter:
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
                        "Warning: keys not present in initial iteration will not be saved"
                        f" ({extra_keys_str})"
                    )
            self.inner.writerows(rowdicts)

    def __init__(self, outdir: str) -> None:
        self.outdir = outdir
        self.out_step_raw = open(outdir + "/history_step.csv", "w")
        self.out_step = self.AutoDictWriter(self.out_step_raw)
        self.out_episode_raw = open(outdir + "/history_episode.csv", "w")
        self.out_episode = self.AutoDictWriter(self.out_episode_raw)

        self._steps = []
        self._num_episodes = 0

    def step(self, info: dict) -> None:
        info = _flatten_dict(info, normalize_types=True)
        info["step"] = len(self._steps)
        info["episode"] = self._num_episodes

        self._steps.append(info)

    def end_episode(self, info: dict) -> None:
        info = _flatten_dict(info, normalize_types=True)
        info["episode"] = self._num_episodes

        self.out_episode.writerows([info])
        self.out_step.writerows(self._steps)

        self._steps.clear()
        self._num_episodes += 1

    def __close__(self):
        self.out_step_raw.close()
        self.out_episode_raw.close()


class HistoryLogger(Logger):
    def __init__(self):
        self.step_info = []
        self.history = []

    def step(self, info: dict) -> None:
        self.step_info.append(info)

    def end_episode(self, info: dict) -> None:
        episode_info = {"step_info": self.step_info, **info}
        self.history.append(episode_info)
        self.step_info = []


def setup_loggers(
    loggers: list[Logger] | None,
    num_episodes: int,
    return_history: bool,
    default_tqdm_metrics: list[str],
    with_timing: bool = True,
) -> tuple[Logger, HistoryLogger | None]:
    logger = LoggerList(loggers or [])

    if return_history:
        logger_hist = HistoryLogger()
        logger.append(logger_hist)
    else:
        logger_hist = None

    if not logger.has_type(TqdmLogger):
        logger.append(TqdmLogger(default_tqdm_metrics))

    logger.set_num_episodes(num_episodes)

    if with_timing:
        logger = WithTiming(logger)
    return logger, logger_hist


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
