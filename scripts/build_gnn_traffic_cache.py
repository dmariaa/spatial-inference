from __future__ import annotations

import json
from pathlib import Path

import click

from metraq_dip.data.aq_backends import get_aq_backend
from metraq_dip.data.metraq_db import metraq_db
from metraq_gnn.data import load_aq_sensor_cache
from metraq_gnn.data.traffic_cache import build_traffic_grid_cache


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option(
    "--aq-cache",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--start", default="2015-01-01 00:00", help="Inclusive, hour-aligned timestamp.")
@click.option("--end", default="2024-12-31 23:00", help="Inclusive, hour-aligned timestamp.")
@click.option("--sensor-batch-size", type=click.IntRange(min=1, max=512), default=64)
@click.option("--statement-timeout-seconds", type=click.IntRange(min=1), default=120)
def main(
    aq_cache: Path,
    output: Path,
    start: str,
    end: str,
    sensor_batch_size: int,
    statement_timeout_seconds: int,
) -> None:
    """Build or resume a partition-bounded hourly traffic grid cache."""
    cache = load_aq_sensor_cache(aq_cache)
    backend = get_aq_backend(
        dataset=cache.metadata["dataset"],
        backend=cache.metadata["backend"],
    )
    with metraq_db.engine.connect() as connection:
        traffic_cache = build_traffic_grid_cache(
            path=output,
            aq_cache=cache,
            aq_backend=backend,
            connection=connection,
            start=start,
            end=end,
            sensor_batch_size=sensor_batch_size,
            statement_timeout_seconds=statement_timeout_seconds,
            progress_callback=lambda month, index, total, elapsed: click.echo(
                f"{index}/{total} {month} completed in {elapsed:.1f} s",
                err=True,
            ),
        )
    click.echo(
        json.dumps(
            {
                "path": str(traffic_cache.path.resolve()),
                "shape": list(traffic_cache.values.shape),
                "grid_shape": list(traffic_cache.grid_shape),
                "completed_months": traffic_cache.metadata["completed_months"],
                "available_cells": int(traffic_cache.availability.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
