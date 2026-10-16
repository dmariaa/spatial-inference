"""Dataset-wide meteorology admission using the live catalog; never updates the DB."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ADMISSION = "d.is_valid=1 AND d.value>=m.min_value AND d.value<=m.max_value"


def derive_wind(data):
    """Convert only complete admitted speed/direction pairs; no clipping."""
    wind = data[data.magnitude_id.isin([81, 82])]
    keys = ["sensor_id", "entry_date"]
    values = wind.pivot(index=keys, columns="magnitude_id", values="value").reindex(columns=[81, 82]).dropna()
    provenance = wind.pivot(index=keys, columns="magnitude_id", values="is_interpolated").reindex(index=values.index, columns=[81, 82])
    results = []
    for magnitude, component in [(811, -values[81] * np.sin(np.deg2rad(values[82]))),
                                 (812, -values[81] * np.cos(np.deg2rad(values[82])))]:
        frame = component.rename("value").reset_index()
        frame["magnitude_id"] = magnitude
        frame["is_interpolated"] = np.where(provenance.eq(1).any(axis=1), 1,
                                            np.where(provenance.eq(0).all(axis=1), 0, np.nan))
        frame["derived"] = True
        results.append(frame)
    return pd.concat(results, ignore_index=True)


@click.command()
@click.option("--start", default="2019-01", help="First month, inclusive.")
@click.option("--end", default="2024-12", help="Last month, inclusive.")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--statement-timeout", type=click.IntRange(1, 60), default=20)
def main(start, end, output, statement_timeout):
    """Export admitted meteorology and exclusions for all sensors, without splits."""
    try:
        start_period, end_period = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    except ValueError:
        raise click.ClickException("Use YYYY-MM dates.") from None
    if start_period > end_period:
        raise click.ClickException("Start must not exceed end.")
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "completed_months.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[record["month"]] = record
    from metraq_dip.data.metraq_db_legacy import metraq_db
    with metraq_db.engine.connect() as connection:
        connection.execute(text(f"SET SESSION max_statement_time={statement_timeout}"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        catalog = [dict(row) for row in connection.execute(text("SELECT id,name,units,min_value,max_value FROM aq_magnitudes WHERE type='meteo' ORDER BY id")).mappings()]
        scope = dict(start=str(start_period), end=str(end_period), catalog=catalog,
                     selection="All dataset sensors, meteorology only; no splits",
                     admission=ADMISSION, clipping=False, database_updates=False,
                     query_timeout=statement_timeout,
                     wind="u=-VV*sin(DV), v=-VV*cos(DV); both source components admitted; observed only when both observed",
                     quality="is_valid contains municipal flag plus project corrections; original flag not reconstructed",
                     timestamp="Database timestamps preserved; no timezone reinterpretation")
        scope_path = output / "scope.json"
        if scope_path.exists() and json.loads(scope_path.read_text(encoding="utf-8")) != scope:
            raise click.ClickException("Catalog or scope changed; use a new output directory.")
        scope_path.write_text(json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8")
        connection.rollback()
        for index, month in enumerate(pd.period_range(start_period, end_period, freq="M"), 1):
            key = str(month)
            if key in done:
                continue
            params = dict(start=month.start_time.to_pydatetime(), end=(month + 1).start_time.to_pydatetime())
            connection.execute(text("START TRANSACTION READ ONLY"))
            try:
                stats_sql = f"""SELECT d.magnitude_id,d.is_interpolated,COUNT(*) AS total_records,
                    SUM(CASE WHEN d.is_valid=1 THEN 1 ELSE 0 END) AS flag_valid_records,
                    SUM(CASE WHEN {ADMISSION} THEN 1 ELSE 0 END) AS admitted_records,
                    SUM(CASE WHEN d.is_valid=1 AND m.min_value IS NOT NULL AND m.max_value IS NOT NULL AND (d.value<m.min_value OR d.value>m.max_value) THEN 1 ELSE 0 END) AS excluded_bounds,
                    SUM(CASE WHEN d.is_valid IS NULL OR d.is_valid<>1 THEN 1 ELSE 0 END) AS excluded_flag,
                    SUM(CASE WHEN d.is_valid=1 AND (d.value IS NULL OR m.min_value IS NULL OR m.max_value IS NULL) THEN 1 ELSE 0 END) AS excluded_missing
                    FROM MAD_merged_aq_data d FORCE INDEX(idx_entry_date)
                    JOIN aq_magnitudes m ON m.id=d.magnitude_id
                    WHERE m.type='meteo' AND d.entry_date>=:start AND d.entry_date<:end
                    GROUP BY d.magnitude_id,d.is_interpolated"""
                stats = [dict(row) for row in connection.execute(text(stats_sql), params).mappings()]
                values_sql = f"""SELECT d.sensor_id,d.entry_date,d.magnitude_id,d.value,d.is_interpolated
                    FROM MAD_merged_aq_data d FORCE INDEX(idx_entry_date)
                    JOIN aq_magnitudes m ON m.id=d.magnitude_id
                    WHERE m.type='meteo' AND d.entry_date>=:start AND d.entry_date<:end AND {ADMISSION}"""
                data = pd.DataFrame(connection.execute(text(values_sql), params).fetchall(),
                                    columns=["sensor_id", "entry_date", "magnitude_id", "value", "is_interpolated"])
                connection.rollback()
            except Exception as exc:
                connection.rollback()
                code = getattr(getattr(exc, "orig", None), "args", (None,))[0]
                raise click.ClickException(f"Query failed in {key}: {type(exc).__name__}, code {code}; checkpoints retained.") from None
            assert len(data) == sum(int(row["admitted_records"]) for row in stats)
            assert not data.duplicated(["sensor_id", "entry_date", "magnitude_id"]).any()
            data["derived"] = False
            wind = derive_wind(data)
            exported = pd.concat([data, wind], ignore_index=True)
            assert np.isfinite(exported.value).all()
            exported.to_parquet(output / f"meteo_{key}.parquet", index=False)
            item = dict(month=key, counts=[{k: int(v) if v is not None else None for k,v in row.items()} for row in stats],
                        derived_records=len(wind), admitted_records=len(data))
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            done[key] = item
            click.echo(f"{index}/{len(pd.period_range(start_period,end_period,freq='M'))} {key}: {len(data):,} admitted, {sum(int(r['excluded_bounds']) for r in stats):,} bounds exclusions", err=True)
    monthly = pd.DataFrame([dict(month=key, **row) for key,item in done.items() for row in item["counts"]])
    counts = ["total_records", "flag_valid_records", "admitted_records", "excluded_bounds", "excluded_flag", "excluded_missing"]
    assert (monthly.total_records == monthly.admitted_records + monthly.excluded_bounds + monthly.excluded_flag + monthly.excluded_missing).all()
    summary = monthly.groupby(["magnitude_id", "is_interpolated"], dropna=False)[counts].sum().reset_index().merge(pd.DataFrame(catalog), left_on="magnitude_id", right_on="id")
    monthly.to_csv(output / "monthly_admission.csv", index=False)
    summary.to_csv(output / "admission_summary.csv", index=False)
    totals = summary[counts].sum()
    (output / "report.md").write_text(
        "# Meteorologia admitida para el diagnostico\n\n"
        f"Periodo {start_period} a {end_period}; dataset completo, sin splits.\n\n"
        "Regla: is_valid=TRUE y min_value <= value <= max_value, con limites vigentes de AQ_magnitudes. "
        "No se modifica la BBDD ni se recortan valores. No se reconstruye la bandera municipal original.\n\n"
        + totals.to_string() + "\n\n"
        "Los parquets mensuales contienen solo valores admitidos; ausencia de fila significa ausencia "
        "para el analisis, no cero. No se rellenan ausencias. Los componentes u/v solo existen si ambos "
        "componentes originales pasan el filtro. derived distingue los componentes calculados. "
        "Los componentes son observados solo cuando ambos originales lo son.\n\n"
        "Se conservan unidades y timestamps de origen. No hay normalizacion, PCA, correlaciones o entrenamiento. "
        "El catalogo aplicado queda guardado en scope.json. La extraccion usa transacciones mensuales; "
        "no es un snapshot simultaneo de todo el dataset.\n\n"
        "Reproducir en un directorio nuevo o reanudar el mismo:\n\n```powershell\n"
        f".venv\\Scripts\\python.exe -B scripts/prepare_meteorology_dataset.py --start {start_period} --end {end_period} "
        f"--output {output.as_posix()} --statement-timeout {statement_timeout}\n```\n", encoding="utf-8")
    click.echo(f"COMPLETE: {int(totals.admitted_records):,} admitted; {int(totals.excluded_bounds):,} excluded by bounds")


if __name__ == "__main__":
    main()
