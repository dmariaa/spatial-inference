from __future__ import annotations

import pytest

from metraq_dip.tools.config_tools import SessionConfig, TrainerConfig, load_session_config


def _base_config() -> dict:
    return {
        "pollutants": [7],
        "hours": 24,
        "epochs": 250,
        "ensemble_size": 5,
        "lr": 0.01,
        "normalize": False,
        "add_meteo": False,
        "add_time_channels": False,
        "add_coordinates": False,
        "add_distance_to_sensors": True,
        "model": {
            "base_channels": 16,
            "levels": 3,
            "preserve_time": False,
            "neural_upscale": False,
            "skip_connections": True,
        },
        "spread_test_groups": {
            "n_groups": 10,
            "group_size": 4,
            "max_uses_per_sensor": 2,
        },
        "random_time_windows": {
            "year": 2024,
            "windows_per_month": 20,
            "start_hours": [8, 8, 9, 10],
            "weekend_fraction": 0.4,
        },
    }


def test_session_config_accepts_valid_random_time_windows_config():
    config = SessionConfig.model_validate(_base_config())

    assert config.pollutants == [7]
    assert config.spread_test_groups.n_groups == 10
    assert config.random_time_windows is not None
    assert config.random_time_windows.start_hours == [8, 8, 9, 10]
    assert config.all_time_windows is None


def test_session_config_accepts_valid_all_time_windows_config():
    payload = _base_config()
    payload.pop("random_time_windows")
    payload["all_time_windows"] = {"year": 2024}

    config = SessionConfig.model_validate(payload)

    assert config.all_time_windows is not None
    assert config.all_time_windows.start_hours == list(range(24))
    assert config.random_time_windows is None


def test_session_config_requires_one_time_window_strategy():
    payload = _base_config()
    payload.pop("random_time_windows")

    with pytest.raises(ValueError, match="either 'random_time_windows' or 'all_time_windows'"):
        SessionConfig.model_validate(payload)


def test_session_config_rejects_both_time_window_strategies():
    payload = _base_config()
    payload["all_time_windows"] = {"year": 2024}

    with pytest.raises(ValueError, match="Use only one time window strategy"):
        SessionConfig.model_validate(payload)


def test_session_config_rejects_unknown_fields():
    payload = _base_config()
    payload["unexpected"] = 123

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SessionConfig.model_validate(payload)


def test_build_trainer_config_injects_runtime_fields():
    session_config = SessionConfig.model_validate(_base_config())

    trainer_config = session_config.build_trainer_config(
        date="2024-01-01T08:00:00",
        test_sensors=[28079004, 28079016, 28079036, 28079050],
        validation_sensors=4,
    )

    assert isinstance(trainer_config, TrainerConfig)
    assert trainer_config.date is not None
    assert trainer_config.validation_sensors == 4
    assert trainer_config.test_sensors == [28079004, 28079016, 28079036, 28079050]


def test_load_session_config_from_yaml_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    payload = _base_config()
    config_path.write_text(
        "\n".join(
            [
                "pollutants: [7]",
                "hours: 24",
                "epochs: 250",
                "ensemble_size: 5",
                "lr: 0.01",
                "normalize: false",
                "add_meteo: false",
                "add_time_channels: false",
                "add_coordinates: false",
                "add_distance_to_sensors: true",
                "model:",
                "  base_channels: 16",
                "  levels: 3",
                "  preserve_time: false",
                "  neural_upscale: false",
                "  skip_connections: true",
                "spread_test_groups:",
                "  n_groups: 10",
                "  group_size: 4",
                "  max_uses_per_sensor: 2",
                "random_time_windows:",
                "  year: 2024",
                "  windows_per_month: 20",
                "  start_hours: [8, 9, 10]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_session_config(config_path)

    assert config.model.levels == payload["model"]["levels"]
    assert config.random_time_windows is not None
    assert config.random_time_windows.start_hours == [8, 9, 10]


def test_load_session_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_session_config(config_path)


def test_load_session_config_rejects_invalid_values(tmp_path):
    config_path = tmp_path / "config.yaml"

    lines: list[str] = []
    lines.extend(["pollutants: [0]", "hours: 24", "epochs: 250", "ensemble_size: 5", "lr: 0.01"])
    lines.extend(
        [
            "model:",
            "  base_channels: 16",
            "  levels: 3",
            "  preserve_time: false",
            "  neural_upscale: false",
            "  skip_connections: true",
            "spread_test_groups:",
            "  n_groups: 10",
            "  group_size: 4",
            "  max_uses_per_sensor: 2",
            "all_time_windows:",
            "  year: 2024",
        ]
    )
    config_path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid configuration file"):
        load_session_config(config_path)
