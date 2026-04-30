from __future__ import annotations

import numpy as np
import pandas as pd

from metraq_dip.data.aq_backends import register_aq_backend
from metraq_dip.tools.grid import prepare_grid_context
from metraq_dip.utils import plot


def _build_airparif_like_sensors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["A1", "B1", "C1", "D1"],
            "latitude": [48.9000, 48.8500, 48.8700, 48.8300],
            "longitude": [2.3000, 2.3500, 2.2800, 2.3700],
            "utm_x": [448000.0, 452000.0, 446500.0, 453500.0],
            "utm_y": [5416500.0, 5411000.0, 5413200.0, 5408700.0],
            "metric_crs": ["EPSG:32631"] * 4,
        }
    )


class DummyPlotBackend:
    dataset_name = "plotdummy"
    backend_name = "files"

    def __init__(self, sensors: pd.DataFrame):
        self._sensors = sensors.copy()

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        df = self._sensors.copy()
        if sensors is not None:
            df = df[df["id"].isin([int(sensor_id) for sensor_id in sensors])]
        return df.sort_values("id").reset_index(drop=True)

    def get_measurements(
        self,
        *,
        start_date,
        end_date,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sensor_id": pd.Series(dtype="int64"),
                "entry_date": pd.Series(dtype="datetime64[ns]"),
                "magnitude_id": pd.Series(dtype="int32"),
                "value": pd.Series(dtype="float64"),
            }
        )

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        return {int(magnitude_id): (0.0, 1.0) for magnitude_id in magnitudes}


def _write_dummy_experiment(experiment_dir, *, test_sensors: np.ndarray | None = None) -> None:
    config_path = experiment_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "add_coordinates: false",
                "add_distance_to_sensors: false",
                "add_meteo: false",
                "add_time_channels: false",
                "add_traffic_data: false",
                "normalize: false",
                "pollutants:",
                "- 1",
                "ensemble_size: 1",
                "k_best_n: 1",
                "epochs: 1",
                "hours: 1",
                "lr: 0.01",
                "model:",
                "  architecture: \"autoencoder\"",
                "  base_channels: 4",
                "  levels: 1",
                "  learned_upsampling: false",
                "  preserve_time: false",
                "  skip_connections: false",
                "aq_dataset: \"plotdummy\"",
                "aq_backend: \"files\"",
                "random_time_windows:",
                "  year: 2024",
                "  windows_per_month: 1",
                "  start_hours: [8]",
                "  weekend_fraction: 0.0",
                "spread_test_groups:",
                "  n_groups: 2",
                "  group_size: 2",
                "  max_uses_per_sensor: 1",
            ]
        ),
        encoding="utf-8",
    )

    np.savez(
        experiment_dir / "data.npz",
        test_sensors=np.array([[1, 2], [3, 4]], dtype=np.int64) if test_sensors is None else test_sensors,
        time_windows=np.array(["2024-01-01T08:00:00"], dtype="datetime64[s]"),
    )


def test_plot_sensor_groups_map_supports_metric_crs_and_default_labels():
    sensors = _build_airparif_like_sensors()
    grid_ctx = prepare_grid_context(sensors, cell_size_m=1000, margin_m_x=1000, margin_m_y=1000)

    fig = plot.plot_sensor_groups_map(
        grid_ctx=grid_ctx,
        sensor_groups=[
            {
                "train_sensors": [3, 4],
                "val_sensors": [],
                "test_sensors": [1, 2],
            }
        ],
        n_cols=1,
    )

    assert len(fig.data) == 4
    assert fig.layout.annotations[0].text == "1-2"
    assert fig.layout.map.center.lat is not None
    assert fig.layout.map.center.lon is not None


def test_load_experiment_sensor_groups_from_experiment_folder(tmp_path):
    sensors = _build_airparif_like_sensors()
    register_aq_backend(dataset="plotdummy", backend="files", implementation=DummyPlotBackend(sensors))

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    _write_dummy_experiment(experiment_dir)

    loaded = plot.load_experiment_sensor_groups(experiment_dir)
    assert loaded.group_labels == ["1-2", "3-4"]
    assert loaded.sensor_groups[0]["test_sensors"] == [1, 2]
    assert loaded.sensor_groups[0]["train_sensors"] == [3, 4]

    loaded_no_train = plot.load_experiment_sensor_groups(experiment_dir, include_train_sensors=False)
    assert loaded_no_train.sensor_groups[0]["train_sensors"] == []

    fig = plot.plot_experiment_sensor_groups_map(
        experiment_dir,
        n_cols=1,
        title="Experiment sensor groups",
    )

    assert len(fig.data) == 8
    assert fig.layout.title.text == "Experiment sensor groups"
    assert [annotation.text for annotation in fig.layout.annotations] == ["1-2", "3-4"]


def test_main_writes_map_html_for_experiment_folder(tmp_path):
    sensors = _build_airparif_like_sensors()
    register_aq_backend(dataset="plotdummy", backend="files", implementation=DummyPlotBackend(sensors))

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    _write_dummy_experiment(experiment_dir)

    outdir = tmp_path / "rendered"
    written_paths = plot.main(
        experiment_folder=experiment_dir,
        outdir=outdir,
        n_cols=1,
        title="CLI output",
    )

    assert written_paths == [(outdir / "sensor_groups_map.html").resolve()]
    assert written_paths[0].is_file()
    assert "plotly" in written_paths[0].read_text(encoding="utf-8").lower()
