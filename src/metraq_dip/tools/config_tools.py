from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


def _normalize_start_hours(start_hours: list[int], *, deduplicate: bool) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for hour in start_hours:
        hour = int(hour)
        if hour < 0 or hour > 23:
            raise ValueError("start_hours must contain values between 0 and 23.")
        if deduplicate and hour in seen:
            continue
        seen.add(hour)
        normalized.append(hour)

    if not normalized:
        raise ValueError("start_hours must contain at least one hour.")
    return normalized


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_channels: int = Field(gt=0)
    levels: int = Field(gt=0)
    preserve_time: bool
    neural_upscale: bool
    skip_connections: bool


class SpreadTestGroupsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_groups: int = Field(gt=0)
    group_size: int = Field(gt=0)
    max_uses_per_sensor: int = Field(gt=0)


class RandomTimeWindowsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=1900)
    windows_per_month: int = Field(gt=0)
    start_hours: list[int] = Field(min_length=1)
    weekend_fraction: float = Field(default=0.4, ge=0.0, le=1.0)

    @field_validator("start_hours")
    @classmethod
    def validate_start_hours(cls, value: list[int]) -> list[int]:
        # Keep duplicates so repeated values can be used as sampling weights.
        return _normalize_start_hours(value, deduplicate=False)


class AllTimeWindowsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=1900)
    start_hours: list[int] = Field(default_factory=lambda: list(range(24)))

    @field_validator("start_hours")
    @classmethod
    def validate_start_hours(cls, value: list[int]) -> list[int]:
        return _normalize_start_hours(value, deduplicate=True)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pollutants: list[int] = Field(min_length=1)
    hours: int = Field(gt=0)
    epochs: int = Field(gt=0)
    ensemble_size: int = Field(gt=0)
    lr: float = Field(gt=0)

    normalize: bool = False
    add_meteo: bool = False
    add_time_channels: bool = False
    add_coordinates: bool = False
    add_distance_to_sensors: bool = False
    add_traffic_data: bool = False
    k_best_n: int | None = Field(default=None, gt=0)

    model: ModelConfig

    # Runtime-injected fields for each experiment:
    date: datetime | None = None
    validation_sensors: int | None = Field(default=None, gt=0)
    test_sensors: list[int] | None = Field(default=None, min_length=1)
    data_split: list[float] | None = None

    @field_validator("pollutants")
    @classmethod
    def validate_pollutants(cls, value: list[int]) -> list[int]:
        if any(pollutant <= 0 for pollutant in value):
            raise ValueError("pollutants must contain positive sensor magnitude ids.")
        return value

    @field_validator("test_sensors")
    @classmethod
    def validate_test_sensors(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(sensor_id <= 0 for sensor_id in value):
            raise ValueError("test_sensors must contain positive sensor ids.")
        return value

    @field_validator("data_split")
    @classmethod
    def validate_data_split(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 3:
            raise ValueError("data_split must contain exactly 3 values: train, validation and test.")
        if any(part <= 0 for part in value):
            raise ValueError("data_split entries must be > 0.")
        total = sum(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("data_split entries must sum to 1.0.")
        return value


class SessionConfig(TrainerConfig):
    spread_test_groups: SpreadTestGroupsConfig
    random_time_windows: RandomTimeWindowsConfig | None = None
    all_time_windows: AllTimeWindowsConfig | None = None

    @model_validator(mode="after")
    def validate_time_windows_strategy(self) -> "SessionConfig":
        if self.random_time_windows is None and self.all_time_windows is None:
            raise ValueError("Configuration must define either 'random_time_windows' or 'all_time_windows'.")
        if self.random_time_windows is not None and self.all_time_windows is not None:
            raise ValueError("Use only one time window strategy: 'random_time_windows' or 'all_time_windows'.")
        return self

    def build_trainer_config(
        self,
        *,
        date: datetime,
        test_sensors: list[int],
        validation_sensors: int = 4,
    ) -> TrainerConfig:
        payload = self.model_dump(
            exclude={"spread_test_groups", "random_time_windows", "all_time_windows"},
        )
        payload["date"] = date
        payload["test_sensors"] = test_sensors
        payload["validation_sensors"] = validation_sensors
        return TrainerConfig.model_validate(payload)


def load_session_config(config_file: Path | str) -> SessionConfig:
    config_path = Path(config_file)
    with config_path.open("r", encoding="utf-8") as file:
        loaded: Any = yaml.safe_load(file)

    if loaded is None:
        raise ValueError("Configuration file is empty. Please provide a YAML mapping with the experiment settings.")
    if not isinstance(loaded, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")

    try:
        return SessionConfig.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration file '{config_path}':\n{exc}") from exc
