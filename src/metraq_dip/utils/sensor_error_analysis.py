from __future__ import annotations

import numpy as np
import pandas as pd


METHOD_COLUMNS = {
    "DIP": ("DIP_L1Loss", "DIP_MSELoss"),
    "KRG": ("KRG_L1Loss", "KRG_MSELoss"),
    "IDW": ("IDW_L1Loss", "IDW_MSELoss"),
}


def summarize_sensor_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["group", "sensor_group", "sensor_id", "method"]
    for key, frame in predictions.groupby(keys, sort=True):
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "count": len(frame),
                "mae": frame["absolute_error"].mean(),
                "rmse": np.sqrt(frame["squared_error"].mean()),
                "wape": frame["absolute_error"].sum() / frame["target"].abs().sum(),
                "bias": frame["error"].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def validate_window_aggregates(predictions: pd.DataFrame, expected: pd.DataFrame) -> None:
    actual = (
        predictions.groupby(["sensor_group", "window_end", "method"])
        .agg(mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
        .reset_index()
    )
    expected = expected.copy()
    expected["window_end"] = pd.to_datetime(expected["time_window"])
    expected = expected.set_index(["sensor_group", "window_end"])
    for row in actual.itertuples(index=False):
        l1_column, mse_column = METHOD_COLUMNS[row.method]
        reference = expected.loc[(row.sensor_group, row.window_end)]
        if not np.isclose(row.mae, float(reference[l1_column]), rtol=2e-5, atol=2e-5):
            raise RuntimeError(
                f"{row.method} MAE mismatch for {row.sensor_group} at {row.window_end}: "
                f"{row.mae} != {reference[l1_column]}"
            )
        if not np.isclose(row.mse, float(reference[mse_column]), rtol=2e-5, atol=2e-5):
            raise RuntimeError(
                f"{row.method} MSE mismatch for {row.sensor_group} at {row.window_end}: "
                f"{row.mse} != {reference[mse_column]}"
            )


__all__ = ["summarize_sensor_errors", "validate_window_aggregates"]
