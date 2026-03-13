import itertools
import os
import random
from calendar import monthrange
from collections import Counter
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from metraq_dip.data.metraq_db import metraq_db

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def sensor_group_hash(ids):
    norm = tuple(sorted(ids))
    payload = "-".join(map(str, norm))
    return payload


def get_random_sensors(*, val_number: int, test_number: int, pollutants: List[int], sensors: List[int] = None):
    sensors: pd.DataFrame = metraq_db.get_sensors(magnitudes=pollutants, sensors=sensors)
    random_sensors = sensors.sample(frac=1)["id"].to_numpy()

    train_number = len(sensors) - val_number - test_number
    train_sensors = random_sensors[:train_number]
    val_sensors = random_sensors[train_number:train_number + val_number]
    test_sensors = random_sensors[train_number + val_number:]

    return train_sensors, val_sensors, test_sensors


def _score_group_min_pairwise_dist(xy: np.ndarray) -> float:
    # xy: (4,2)
    d = xy[:, None, :] - xy[None, :, :]
    d2 = (d**2).sum(-1)
    return float(np.sqrt(d2[d2 > 0].min()))


def get_spread_test_groups(*, n_groups: int, group_size: int, max_uses_per_sensor: int, magnitudes: list[int]):
    df = metraq_db.get_sensors(magnitudes=magnitudes)
    ids = df["id"].to_numpy()
    xy = df[["utm_x", "utm_y"]].to_numpy().astype(np.float64)

    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    scored = []
    for group in itertools.combinations(ids, group_size):
        pts = xy[[id_to_idx[s] for s in group]]
        scored.append((_score_group_min_pairwise_dist(pts), tuple(sorted(group))))

    scored.sort(key=lambda x: x[0], reverse=True)

    usage = Counter()
    selected = []
    for _, group in scored:
        if all(usage[s] < max_uses_per_sensor for s in group):
            selected.append(group)
            for s in group:
                usage[s] += 1
            if len(selected) == n_groups:
                break

    if len(selected) < n_groups:
        raise RuntimeError(
            f"Could only select {len(selected)} groups with max_uses_per_sensor={max_uses_per_sensor}. "
            f"Try increasing the cap."
        )

    return selected, usage


def get_random_time_windows(*,
                            year: int,
                            windows_per_month: int = 10,
                            weekend_fraction: float = 0.4,
                            start_hours: list[int] = None
                            ):
    if start_hours is None:
        raise ValueError("start_hours must be provided")

    rng = np.random.default_rng()
    # Keep user-provided order, remove duplicates to avoid duplicated windows.
    start_hours = tuple(dict.fromkeys(int(h) for h in start_hours))
    if len(start_hours) == 0:
        raise ValueError("start_hours must contain at least one hour")
    if any(h < 0 or h > 23 for h in start_hours):
        raise ValueError("start_hours must contain values between 0 and 23")
    if windows_per_month <= 0:
        raise ValueError("windows_per_month must be > 0")
    if not 0 <= weekend_fraction <= 1:
        raise ValueError("weekend_fraction must be between 0 and 1")

    windows = []

    for month in range(1, 13):
        n_days = monthrange(year, month)[1]
        all_days = [pd.Timestamp(year=year, month=month, day=d) for d in range(1, n_days + 1)]
        weekend_days = [d for d in all_days if d.weekday() >= 5]
        weekday_days = [d for d in all_days if d.weekday() < 5]

        def build_candidates(days):
            candidates = []
            for day in days:
                for h in start_hours:
                    start = day + pd.Timedelta(hours=h)
                    # Keep existing validity criterion (handles DST-aware timestamps if ever used).
                    if len(pd.date_range(start, start + pd.Timedelta(hours=23), freq="h")) == 24:
                        candidates.append(start)
            return candidates

        weekend_candidates = build_candidates(weekend_days)
        weekday_candidates = build_candidates(weekday_days)
        max_month_candidates = len(weekend_candidates) + len(weekday_candidates)
        if max_month_candidates < windows_per_month:
            raise ValueError(
                f"Cannot sample {windows_per_month} unique windows for {year}-{month:02d} "
                f"with start_hours={start_hours}. Maximum possible is {max_month_candidates}."
            )
        rng.shuffle(weekend_candidates)
        rng.shuffle(weekday_candidates)

        weekend_target = int(round(windows_per_month * weekend_fraction))
        weekend_take = min(weekend_target, len(weekend_candidates))

        month_windows = weekend_candidates[:weekend_take]

        weekday_needed = windows_per_month - len(month_windows)
        weekday_take = min(weekday_needed, len(weekday_candidates))
        month_windows.extend(weekday_candidates[:weekday_take])

        # If one side cannot satisfy its share, fill from remaining windows on the other side.
        if len(month_windows) < windows_per_month:
            used = set(month_windows)
            remaining = [w for w in weekend_candidates[weekend_take:] if w not in used]
            remaining.extend(w for w in weekday_candidates[weekday_take:] if w not in used)
            rng.shuffle(remaining)
            month_windows.extend(remaining[:windows_per_month - len(month_windows)])

        month_windows = sorted(month_windows)
        windows.extend(month_windows)

    return sorted(windows)


if __name__=="__main__":
    # s = get_random_sensors(4, 4, [7])
    # s = get_spread_test_groups(n_groups=10, group_size=4, max_uses_per_sensor=2, magnitudes=[7])
    # s2 = get_spread_test_groups(n_groups=10, group_size=4, max_uses_per_sensor=2, magnitudes=[7])
    # print(s==s2)
    time_windows = get_random_time_windows(year=2024, windows_per_month=20, start_hours=list(range(6, 23)))
    pass
