import numpy as np
import pandas as pd

from metraq_dip.utils.sensor_error_analysis import (
    summarize_sensor_errors,
    validate_window_aggregates,
)


def test_sensor_error_summary_and_window_validation():
    predictions = pd.DataFrame(
        {
            "group": [1, 1, 1, 1],
            "sensor_group": ["10-20"] * 4,
            "window_end": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
            "sensor_id": [10, 20, 10, 20],
            "method": ["DIP"] * 4,
            "target": [2.0, 4.0, 2.0, 2.0],
            "error": [1.0, -1.0, 2.0, -2.0],
            "absolute_error": [1.0, 1.0, 2.0, 2.0],
            "squared_error": [1.0, 1.0, 4.0, 4.0],
        }
    )
    predictions["prediction"] = predictions["target"] + predictions["error"]
    expected = pd.DataFrame(
        {
            "sensor_group": ["10-20", "10-20"],
            "time_window": ["2024-01-01", "2024-01-02"],
            "DIP_L1Loss": [1.0, 2.0],
            "DIP_MSELoss": [1.0, 4.0],
        }
    )

    validate_window_aggregates(predictions, expected)
    metrics = summarize_sensor_errors(predictions).set_index("sensor_id")
    assert metrics.loc[10, "count"] == 2
    assert metrics.loc[10, "mae"] == 1.5
    assert np.isclose(metrics.loc[10, "rmse"], np.sqrt(2.5))
    assert metrics.loc[10, "bias"] == 1.5
