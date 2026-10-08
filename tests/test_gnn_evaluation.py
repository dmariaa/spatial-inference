import numpy as np
import pandas as pd

from metraq_gnn.evaluation import summarize_all_predictions, summarize_sensor_predictions


def test_prediction_summaries_preserve_sensor_metrics_and_global_aggregation():
    predictions = pd.DataFrame(
        {
            "sensor_id": [10, 10, 20],
            "magnitude_id": [8, 8, 8],
            "target": [2.0, 4.0, 2.0],
            "prediction": [3.0, 2.0, 5.0],
            "error": [1.0, -2.0, 3.0],
            "absolute_error": [1.0, 2.0, 3.0],
            "squared_error": [1.0, 4.0, 9.0],
        }
    )

    per_sensor = summarize_sensor_predictions(predictions).set_index("sensor_id")
    assert per_sensor.loc[10, "count"] == 2
    assert per_sensor.loc[10, "mae"] == 1.5
    assert np.isclose(per_sensor.loc[10, "rmse"], np.sqrt(2.5))
    assert per_sensor.loc[10, "wape"] == 0.5
    assert per_sensor.loc[10, "bias"] == -0.5
    assert per_sensor.loc[20, "mae"] == 3.0

    aggregate = summarize_all_predictions(predictions)
    assert aggregate["test_target_count"] == 3
    assert aggregate["test_mae_physical"] == 2.0
    assert np.isclose(aggregate["test_rmse_physical"], np.sqrt(14.0 / 3.0))
    assert aggregate["test_wape"] == 0.75
