"""Build physical wind vectors using existing admitted monthly meteorology."""
from pathlib import Path

import click
import pandas as pd

from metraq_gnn.data.cache import load_aq_sensor_cache
from metraq_gnn.data.wind_cache import build_wind_source_cache


@click.command(context_settings={"show_default": True})
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--meteorology-dir", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sensors", required=True, type=click.Path(exists=True, path_type=Path),
              help="CSV/parquet catalogue: id, utm_x, utm_y in EPSG:25830; AQ and weather stations.")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--start", help="First hour; defaults to AQ cache start.")
@click.option("--end", help="Last hour; defaults to AQ cache end.")
@click.option("--observed-only/--include-interpolated", default=False)
@click.option("--allow-missing-months", is_flag=True,
              help="Explicitly keep missing months unavailable; never drop AQ windows.")
@click.option("--mapping", type=click.Choice(["station", "idw"]), default="station",
              help="Assign existing AQ-station wind directly; IDW is an explicit optional alternative.")
@click.option("--source-interpolation", help="Document the interpolation already applied upstream.")
@click.option("--power", default=2., type=click.FloatRange(min=0., min_open=True))
@click.option("--neighbours", default=8, type=click.IntRange(min=1))
@click.option("--max-radius-m", type=click.FloatRange(min=0., min_open=True))
def main(cache, meteorology_dir, sensors, output, start, end, observed_only,
         allow_missing_months, power, neighbours, max_radius_m, mapping, source_interpolation):
    """No database access, new u/v conversion, or AQ target values are needed."""
    catalogue = pd.read_parquet(sensors) if sensors.suffix.lower() == ".parquet" else pd.read_csv(sensors)
    try:
        result = build_wind_source_cache(
            path=output, aq_cache=load_aq_sensor_cache(cache),
            meteorology_dirs=meteorology_dir, sensors=catalogue, start=start, end=end,
            observed_only=observed_only, allow_missing_months=allow_missing_months,
            power=power, neighbours=neighbours, max_radius_m=max_radius_m,
            mapping=mapping, source_interpolation=source_interpolation,
            progress_callback=lambda month, hours: click.echo(f"{month}: {hours} hours with paired wind"))
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wind cache: {result.path}; available={result.metadata['available_fraction']:.2%}")
    click.echo(f"Identity: {result.metadata['fingerprint']}")


if __name__ == "__main__":
    main()
