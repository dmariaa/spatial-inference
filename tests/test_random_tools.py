import importlib
import sys
from types import SimpleNamespace

import pytest
import sqlalchemy


def import_random_tools(monkeypatch):
    class FakeEngine:
        def connect(self):
            return "fake_connection"

    def fake_create_engine(connection_string, echo=False):
        return FakeEngine()

    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)

    # Allow these tests to run even if torch is not installed.
    if "torch" not in sys.modules:
        fake_torch = SimpleNamespace(
            manual_seed=lambda *_: None,
            cuda=SimpleNamespace(manual_seed=lambda *_: None),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(deterministic=False, benchmark=False),
            ),
            use_deterministic_algorithms=lambda *_: None,
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    sys.modules.pop("metraq_dip.data.metraq_db", None)
    sys.modules.pop("metraq_dip.tools.random_tools", None)
    return importlib.import_module("metraq_dip.tools.random_tools")


def test_get_random_time_windows_returns_expected_shape_and_constraints(monkeypatch):
    module = import_random_tools(monkeypatch)

    windows_per_month = 10
    start_hours = (8, 8, 17, 22)
    windows = module.get_random_time_windows(
        year=2024,
        windows_per_month=windows_per_month,
        weekend_fraction=0.4,
        start_hours=start_hours,
    )

    assert isinstance(windows, list)
    assert len(windows) == 12 * windows_per_month
    assert windows == sorted(windows)
    assert len(set(windows)) == len(windows)
    assert {w.year for w in windows} == {2024}
    assert all(w.hour in {8, 17, 22} for w in windows)

    for month in range(1, 13):
        month_windows = [w for w in windows if w.month == month]
        assert len(month_windows) == windows_per_month


def test_get_random_time_windows_respects_weekend_ratio_when_feasible(monkeypatch):
    module = import_random_tools(monkeypatch)

    windows_per_month = 9
    weekend_fraction = 0.4
    expected_weekend_count = round(windows_per_month * weekend_fraction)
    windows = module.get_random_time_windows(
        year=2024,
        windows_per_month=windows_per_month,
        weekend_fraction=weekend_fraction,
        start_hours=(8,),
    )

    for month in range(1, 13):
        month_windows = [w for w in windows if w.month == month]
        weekend_count = sum(1 for w in month_windows if w.weekday() >= 5)
        assert weekend_count == expected_weekend_count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"windows_per_month": 0, "weekend_fraction": 0.4, "start_hours": (8, 17)},
        {"windows_per_month": 10, "weekend_fraction": -0.1, "start_hours": (8, 17)},
        {"windows_per_month": 10, "weekend_fraction": 1.1, "start_hours": (8, 17)},
        {"windows_per_month": 10, "weekend_fraction": 0.4, "start_hours": ()},
        {"windows_per_month": 10, "weekend_fraction": 0.4, "start_hours": (8, 24)},
        {"windows_per_month": 40, "weekend_fraction": 0.4, "start_hours": (8,)},
    ],
)
def test_get_random_time_windows_raises_for_invalid_or_impossible_inputs(monkeypatch, kwargs):
    module = import_random_tools(monkeypatch)

    with pytest.raises(ValueError):
        module.get_random_time_windows(year=2024, **kwargs)
