import importlib
import sys

import pandas as pd
import sqlalchemy
from pandas.api.types import is_integer_dtype


def import_metraq_db(monkeypatch):
    created = {}

    class FakeEngine:
        def __init__(self):
            self.connect_calls = 0

        def connect(self):
            self.connect_calls += 1
            return "fake_connection"

        def dispose(self):
            return None

    fake_engine = FakeEngine()

    def fake_create_engine(connection_string, **kwargs):
        created["connection_string"] = connection_string
        created.update(kwargs)
        return fake_engine

    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)
    module_name = "metraq_dip.data.metraq_db"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return module, created, fake_engine


def test_get_sensors_calls_read_sql_query_with_magnitudes(monkeypatch):
    module, created, fake_engine = import_metraq_db(monkeypatch)
    captured = {}

    def fake_read_sql_query(query, con, **kwargs):
        captured["query"] = str(query)
        captured["con"] = con
        captured["kwargs"] = kwargs
        return pd.DataFrame({"id": [1], "utm_x": [1.0], "utm_y": [2.0]})

    monkeypatch.setattr(module.pd, "read_sql_query", fake_read_sql_query)

    result = module.metraq_db.get_sensors(magnitudes=[3, 4], sensors=None)

    assert captured["con"] is fake_engine
    assert captured["kwargs"] == {}
    assert "SELECT id, utm_x, utm_y" in captured["query"]
    assert "magnitude_id IN (3,4)" in captured["query"]
    assert created["echo"] is False
    assert created["pool_pre_ping"] is True
    assert created["pool_recycle"] == 1800
    assert result["id"].tolist() == [1]


def test_get_sensors_casts_id_and_filters(monkeypatch):
    module, _, _ = import_metraq_db(monkeypatch)

    def fake_read_sql_query(query, con, **kwargs):
        return pd.DataFrame(
            {"id": ["1", "2", "3"], "utm_x": [1.0, 2.0, 3.0], "utm_y": [4.0, 5.0, 6.0]}
        )

    monkeypatch.setattr(module.pd, "read_sql_query", fake_read_sql_query)

    result = module.metraq_db.get_sensors(magnitudes=[1], sensors=[2])

    assert is_integer_dtype(result["id"])
    assert result["id"].tolist() == [2]


def test_get_sensors_no_filter_when_sensors_none(monkeypatch):
    module, _, _ = import_metraq_db(monkeypatch)

    def fake_read_sql_query(query, con, **kwargs):
        return pd.DataFrame({"id": ["1", "2"], "utm_x": [1.0, 2.0], "utm_y": [3.0, 4.0]})

    monkeypatch.setattr(module.pd, "read_sql_query", fake_read_sql_query)

    result = module.metraq_db.get_sensors(magnitudes=[5], sensors=None)

    assert result["id"].tolist() == [1, 2]
