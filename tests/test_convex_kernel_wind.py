"""Wind math, admitted-cache alignment and common train/evaluation integration."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from torch_geometric.data import Batch, Data

from metraq_dip.tools.grid import prepare_grid_context
from metraq_dip.tools.interpolator import IdwInterpolator
from metraq_gnn.data.cache import AQSensorCache
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import build_wind_source_cache, load_wind_source_cache
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wind_sample(vector=(3., 0.), axis=0):
    positions = torch.zeros((3, 2))
    positions[:, axis] = torch.arange(3.)
    values = torch.tensor([[10.], [0.], [30.]])
    visible = torch.tensor([[True], [False], [True]])
    return Data(pos=positions, num_nodes=3,
                x=torch.cat((values, visible.float()), dim=1).unsqueeze(1),
                context_mask=visible.unsqueeze(1), wind_uv=torch.tensor([vector] * 3),
                wind_available=torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize("axis,vector", [(0, (3., 0.)), (1, (0., 3.))])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_wind_favours_downwind_sources_and_batches_independently(axis, vector, device):
    data = wind_sample(vector, axis).to(device)
    reverse = data.clone()
    reverse.wind_uv *= -1
    kernel = ConvexGeometricKernel(mode="power", wind_mode="directional",
                                  initial_wind_strength=1.).to(device).prepare_geometry(data.pos, [0, 2])
    prediction, weights = kernel(data, return_weights=True)
    assert prediction[1] < 20
    assert kernel(reverse)[1] > 20
    assert prediction[0] == 10 and prediction[2] == 30
    torch.testing.assert_close(weights[0][-1].sum(1), torch.ones(3, device=device))
    assert ((prediction >= 10) & (prediction <= 30)).all()
    batch = Batch.from_data_list([data, reverse])
    torch.testing.assert_close(kernel(batch), torch.cat((prediction, kernel(reverse))))
    loss = kernel(batch)[1].sum()
    loss.backward()
    assert torch.isfinite(kernel.raw_wind_strength.grad)
    assert kernel.raw_wind_strength.grad.abs() > 0


def test_calm_missing_and_receiver_wind_are_neutral_and_legacy_loads_strictly():
    data = wind_sample()
    legacy = ConvexGeometricKernel(mode="power").prepare_geometry(data.pos, [0, 2])
    restored = ConvexGeometricKernel(mode="power").prepare_geometry(data.pos, [0, 2])
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert "raw_wind_strength" not in legacy.state_dict()
    kernel = ConvexGeometricKernel(mode="power", wind_mode="directional").prepare_geometry(data.pos, [0, 2])
    expected = legacy(data)
    data.wind_uv.zero_()
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)
    data.wind_available.zero_()
    data.wind_uv.fill_(float("nan"))
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)
    data.wind_available[1] = True  # Hidden AQ receiver is never a candidate source.
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)
    data.wind_available[0] = True
    with pytest.raises(ValueError, match="finite"):
        kernel(data)
    del data.wind_uv
    with pytest.raises(ValueError, match="requires wind"):
        kernel(data)


@pytest.fixture
def wind_inputs(tmp_path):
    times = pd.date_range("2015-01-01", periods=25, freq="h").append(
        pd.date_range("2023-01-01", periods=25, freq="h")).append(
        pd.date_range("2024-01-01", periods=25, freq="h")).as_unit("ns")
    sensors = pd.DataFrame(dict(id=list(range(7)) + [1000, 1001],
        utm_x=list(440500. + 1000 * np.arange(7)) + [440500., 446500.],
        utm_y=[4470500.] * 9))
    ctx = prepare_grid_context(sensors.iloc[:7], cell_size_m=1000, margin_m_x=500, margin_m_y=500)
    assert ctx["grid"].shape == (1, 7)
    aq_values = np.broadcast_to((10 + np.arange(7, dtype=np.float32))[None, :, None],
                                (len(times), 7, 1)).copy()
    aq = AQSensorCache(path=tmp_path, metadata=dict(grid_shape=[1, 7], cell_size_m=1000,
        margin_m_x=500, margin_m_y=500), timestamps_ns=times.asi8,
        sensor_ids=np.arange(7), sensor_node_indices=np.arange(7), magnitude_ids=np.array([8]),
        values=aq_values, availability=np.isfinite(aq_values), graph=build_grid_graph(ctx))
    meteo = tmp_path / "meteo"
    meteo.mkdir()
    scope = dict(clipping=False, wind="Existing admitted u/v",
                 catalog=[dict(id=81, units="m/s"), dict(id=82, units="Grd")])
    (meteo / "scope.json").write_text(json.dumps(scope))
    for month in times.to_period("M").unique():
        rows = []
        for timestamp in times[times.to_period("M") == month]:
            for station in (1000, 1001):
                for magnitude, value in ((811, 3. if station == 1000 else -1.), (812, 0.)):
                    rows.append(dict(sensor_id=station, entry_date=timestamp, magnitude_id=magnitude,
                                     value=value if timestamp.hour != 0 else 0.,
                                     is_interpolated=int(station == 1001), derived=True))
        pd.DataFrame(rows).to_parquet(meteo / f"meteo_{month}.parquet", index=False)
    return aq, sensors, meteo


def make_wind(tmp_path, inputs, **options):
    aq, sensors, meteo = inputs
    return build_wind_source_cache(path=tmp_path, aq_cache=aq, sensors=sensors,
                                   meteorology_dirs=[meteo], mapping=options.pop("mapping", "idw"), **options)


def test_cache_pairs_components_preserves_calm_and_provenance(wind_inputs, tmp_path):
    aq, _, meteo = wind_inputs
    month = meteo / "meteo_2015-01.parquet"
    frame = pd.read_parquet(month)
    # Missing v excludes the whole station pair at this hour.
    incomplete = (frame.sensor_id == 1000) & (frame.entry_date == aq.time_index[1]) & (frame.magnitude_id == 812)
    frame = frame.loc[~incomplete]
    frame.to_parquet(month, index=False)
    wind = make_wind(tmp_path / "wind", wind_inputs)
    assert isinstance(wind.values, np.memmap)
    assert wind.availability[0].all() and (wind.values[0] == 0).all()
    np.testing.assert_allclose(wind.values[1, :, 0], -1.)
    np.testing.assert_array_equal(np.load(wind.path / "station_count.npy")[:2], [2, 1])
    observed = make_wind(tmp_path / "observed", wind_inputs, observed_only=True)
    assert not observed.availability[1].any()
    assert observed.availability[0].all()  # Valid calm, observed source remains.
    values, mask = wind.align(aq, aq.time_index[-2:])
    assert values.shape == (2, 7, 2) and mask.all()
    with pytest.raises(ValueError, match="timestamps"):
        wind.align(aq, pd.DatetimeIndex(["2025-01-01"]))
    aq.sensor_node_indices[0] = 1
    with pytest.raises(ValueError, match="mapping"):
        wind.align(aq, aq.time_index[:1])


def test_cache_missing_months_need_explicit_fallback_and_detects_corruption(wind_inputs, tmp_path):
    _, _, meteo = wind_inputs
    (meteo / "meteo_2015-01.parquet").unlink()
    with pytest.raises(ValueError, match="missing meteorology months"):
        make_wind(tmp_path / "rejected", wind_inputs)
    wind = make_wind(tmp_path / "allowed", wind_inputs, allow_missing_months=True)
    assert wind.metadata["missing_months"] == ["2015-01"]
    assert not wind.availability[:25].any()
    with (wind.path / "wind_uv.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_wind_source_cache(wind.path)


def test_idw_explicit_coordinates_match_existing_grid_algorithm():
    data = np.array([[[2., 0., 8.]]])
    mask = np.array([[[True, False, True]]])
    old = IdwInterpolator(data, mask)
    point = IdwInterpolator.from_points([0., 2.], [0., 0.], [2., 8.])
    x = np.linspace(0, 2, 9)
    np.testing.assert_array_equal(old(x, np.zeros_like(x), mode="points"), point(x, np.zeros_like(x), mode="points"))


def test_wind_dataset_preserves_aq_windows_masks_and_test_isolation(wind_inputs, tmp_path):
    aq, _, _ = wind_inputs
    wind = make_wind(tmp_path / "wind", wind_inputs)
    options = dict(cache=aq, context_sensor_mask=np.array([[1, 1, 1, 1, 0, 0, 0]], bool),
        forbidden_context_mask=np.array([[0, 0, 0, 0, 1, 1, 1]], bool),
        target_sensor_mask=np.array([[0, 0, 0, 0, 1, 1, 0]], bool), hours=24,
        split_start="2023-01-01", split_end="2023-12-31 23:00",
        normalization_mean=[10.], normalization_std=[2.])
    plain = CachedKernelDataset(**options)
    windy = CachedKernelDataset(**options, wind_cache=wind)
    assert windy.end_indices == plain.end_indices
    for index in range(len(windy)):
        a, b = windy[index], plain[index]
        torch.testing.assert_close(a.x, b.x)
        torch.testing.assert_close(a.target_mask, b.target_mask)
        assert not a.wind_available[4:].any()
        aq.values[:, 6] = np.nan
        torch.testing.assert_close(CachedKernelDataset(**options, wind_cache=wind)[index].x, a.x)
    assert windy.wind_coverage == 1.


@pytest.mark.parametrize("mode", ["power", "fixed"])
def test_directional_training_and_frozen_common_evaluator(wind_inputs, tmp_path, monkeypatch, mode):
    aq, _, _ = wind_inputs
    wind = make_wind(tmp_path / "wind", wind_inputs)
    train = load_script("wind_train", "scripts/train/train_convex_kernel.py")
    evaluation = load_script("wind_evaluation", "scripts/evaluation/evaluate_cached_model.py")
    monkeypatch.setattr(train, "load_aq_sensor_cache", lambda _: aq)
    monkeypatch.setattr(evaluation, "load_aq_sensor_cache", lambda _: aq)
    run = tmp_path / f"run-{mode}"
    # TEST AQ need not exist during training or normalization.
    aq.values[:, 6] = np.nan
    aq.availability[:, 6] = False
    result = CliRunner().invoke(train.main, [
        "--cache", str(tmp_path), "--output", str(run), "--mode", mode,
        "--wind-mode", "directional", "--wind-cache", str(wind.path),
        "--test-sensor-ids", "6", "--validation-nodes", "2", "--seed", "7",
        "--epochs", "2", "--batch-size", "2", "--device", "cpu"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert not summary["test_evaluated"] and summary["epochs_completed"] == 2
    assert summary["wind_strength"] > 0
    assert config["wind_cache_fingerprint"] == wind.metadata["fingerprint"]
    checkpoint = torch.load(run / "best-model.pt", weights_only=True)
    expected_beta = torch.nn.functional.softplus(checkpoint["model_state_dict"]["raw_wind_strength"]).item()
    assert summary["wind_strength"] == expected_beta
    outputs = []
    for target in (30., 1000.):
        aq.values[:, 6] = target
        aq.availability[:, 6] = True
        output = tmp_path / f"test-{mode}-{int(target)}"
        result = CliRunner().invoke(evaluation.main, [
            "--cache", str(tmp_path), "--run-dir", str(run), "--output", str(output), "--device", "cpu"])
        assert result.exit_code == 0, result.output + repr(result.exception)
        evaluated = json.loads((output / "summary.json").read_text())
        assert evaluated["test_windows"] == 2 and evaluated["test_target_count"] == 2
        assert evaluated["method"].endswith("_WIND_DIRECTIONAL")
        outputs.append(pd.read_csv(output / "predictions.csv"))
    np.testing.assert_array_equal(outputs[0].prediction, outputs[1].prediction)
    # A different valid wind cache is still not the training covariate.
    alternate = make_wind(tmp_path / "alternate", wind_inputs, observed_only=True)
    result = CliRunner().invoke(evaluation.main, [
        "--cache", str(tmp_path), "--run-dir", str(run), "--output", str(tmp_path / "rejected-eval"),
        "--wind-cache", str(alternate.path)])
    assert result.exit_code != 0 and "identity differs" in result.output


def test_cache_builder_click_uses_only_local_inputs(wind_inputs, tmp_path, monkeypatch):
    aq, sensors, meteo = wind_inputs
    cli = load_script("wind_builder", "scripts/build_gnn_wind_cache.py")
    monkeypatch.setattr(cli, "load_aq_sensor_cache", lambda _: aq)
    catalogue = tmp_path / "sensors.csv"
    sensors.to_csv(catalogue, index=False)
    result = CliRunner().invoke(cli.main, [
        "--cache", str(tmp_path), "--meteorology-dir", str(meteo), "--sensors", str(catalogue),
        "--mapping", "idw", "--output", str(tmp_path / "built")])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "100.00%" in result.output


def test_station_mapping_keeps_existing_wind_exact_and_never_fills_neighbours(wind_inputs, tmp_path):
    aq, _, meteo = wind_inputs
    for source in meteo.glob("meteo_*.parquet"):
        frame = pd.read_parquet(source)
        frame["sensor_id"] = frame.sensor_id.replace({1000: 0, 1001: 6})
        frame.to_parquet(source, index=False)
    wind = make_wind(tmp_path / "direct", wind_inputs, mapping="station",
                     source_interpolation="Ordinary Kriging (user-provided)")
    assert wind.metadata["mapping"] == "station"
    assert wind.metadata["power"] is None
    assert wind.availability[:, [0, 6]].all()
    assert not wind.availability[:, 1:6].any()
    np.testing.assert_array_equal(wind.values[1, [0, 6], 0], [3., -1.])
    assert wind.availability[0, 0] and (wind.values[0, 0] == 0).all()
    observed = make_wind(tmp_path / "direct-observed", wind_inputs, mapping="station", observed_only=True)
    assert observed.availability[:, 0].all()
    assert not observed.availability[:, 1:].any()
    np.testing.assert_array_equal(np.load(wind.path / "station_count.npy"), 2)
    np.testing.assert_array_equal(np.load(wind.path / "observed_station_count.npy"), 1)
