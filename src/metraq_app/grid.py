"""Aggregate valid observations without treating missing data as zero."""
import numpy as np
from metraq_dip.tools.grid import map_sensor_ids_to_grid, prepare_grid_context


def make_grid(sensors, cell_size=1000):
    return prepare_grid_context(sensors, cell_size_m=cell_size, margin_m_x=3000, margin_m_y=2000)


def aggregate_cells(ctx, sensors, measurements):
    valid = measurements.loc[np.isfinite(measurements.value)].copy()
    # Average duplicates within a sensor first so each sensor has equal weight.
    observations = valid.groupby("sensor_id", as_index=False).agg(value=("value", "mean"))
    observations = observations.merge(sensors, left_on="sensor_id", right_on="id", how="inner")
    rows, cols, mapped = map_sensor_ids_to_grid(ctx, sensors, observations.sensor_id.to_numpy())
    observations["row"] = rows
    observations["col"] = cols
    observations = observations.loc[mapped].copy()
    observations["detail"] = [f"{name} ({int(sid)}): {value:.2f}" for name, sid, value in
                              zip(observations.name, observations.sensor_id, observations.value)]
    cells = observations.groupby(["row", "col"], as_index=False).agg(
        value=("value", "mean"), count=("sensor_id", "size"),
        stations=("detail", lambda values: "<br>".join(values)))
    cells["cell_id"] = cells.row.astype(str) + ":" + cells.col.astype(str)
    return cells, observations
