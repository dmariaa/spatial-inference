"""TRAIN-only auxiliary coverage; monthly, bounded, read-only database queries."""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METEO = {81: "wind_speed", 82: "wind_direction", 83: "temperature",
         86: "humidity", 87: "pressure", 88: "radiation", 89: "precipitation"}
TRAFFIC = ["traffic_intensity", "sensor_occupancy", "avg_speed"]


def traffic_bits(value):
    if value is None:
        return None
    value = str(value)
    if not 1 <= len(value) <= 3 or set(value) - {"0", "1"}:
        return None
    return value.zfill(3)


def cell(x, y):
    return math.floor(float(x) / 1000), math.floor(float(y) / 1000)


def query(connection, sql, ids):
    return connection.execute(text(sql).bindparams(bindparam("ids", expanding=True)),
                              {"ids": list(ids)}).fetchall()


def aq_coverage(connection, train, start, end, expected):
    sql = f"""SELECT sensor_id, entry_date, magnitude_id, value, is_valid, is_interpolated
        FROM MAD_merged_aq_data WHERE sensor_id IN :ids
        AND magnitude_id IN (12,81,82,83,86,87,88,89)
        AND entry_date >= '{start}' AND entry_date < '{end}'"""
    rows = query(connection, sql, train)
    groups = {(sid, mag): [] for sid in train for mag in [12, *METEO]}
    for row in rows:
        groups[(row[0], row[2])].append(row)
    out = []
    for (sid, mag), records in groups.items():
        available, observed, interpolated = set(), set(), set()
        invalid = unknown = off_hour = 0
        for _, ts, _, value, valid, provenance in records:
            if ts.minute or ts.second:
                off_hour += 1
                continue
            if valid != 1 or value is None or not np.isfinite(value):
                invalid += 1
                continue
            available.add(ts)
            if provenance == 0:
                observed.add(ts)
            elif provenance == 1:
                interpolated.add(ts)
            else:
                unknown += 1
        out.append(dict(sensor_id=sid, variable=METEO.get(mag, "NOX"), source="aq",
                        expected_hours=expected, available_hours=len(available),
                        observed_hours=len(observed), interpolated_hours=len(interpolated),
                        invalid_records=invalid, unknown_provenance_records=unknown,
                        off_hour_records=off_hour, source_sensors=1))
    # u/v require speed and direction at the exact same sensor/timestamp.
    for sid in train:
        valid_sets, observed_sets = [], []
        for mag in [81, 82]:
            records = groups[(sid, mag)]
            valid_sets.append({r[1] for r in records if r[4] == 1 and r[3] is not None
                               and np.isfinite(r[3]) and not (r[1].minute or r[1].second)})
            observed_sets.append({r[1] for r in records if r[1] in valid_sets[-1] and r[5] == 0})
        for name in ["wind_u", "wind_v"]:
            out.append(dict(sensor_id=sid, variable=name, source="derived_aq",
                            expected_hours=expected, available_hours=len(set.intersection(*valid_sets)),
                            observed_hours=len(set.intersection(*observed_sets)),
                            interpolated_hours=None, source_sensors=1))
    return out


def traffic_coverage(connection, mapping, aq_cells, start, end, partition, expected):
    out = []
    support = {(sid, j): [set(), set(), set()] for sid in aq_cells for j in range(3)}
    sensor_support = {(sid, j): set() for sid in aq_cells for j in range(3)}
    cell_to_aq = {location: sid for sid, location in aq_cells.items()}
    diagnostics = {"unknown_flag_records": 0, "excluded_quality_records": 0}
    if mapping:
        sql = f"""SELECT sensor_id, DATE(entry_date), HOUR(entry_date), is_interpolated,
            read_error, COUNT(*), COUNT(traffic_intensity), COUNT(sensor_occupancy),
            COUNT(CASE WHEN element_type='M30' THEN avg_speed END)
            FROM traffic_data PARTITION({partition}) FORCE INDEX(PRIMARY)
            WHERE sensor_id IN :ids AND entry_date >= '{start}' AND entry_date < '{end}'
            GROUP BY sensor_id, DATE(entry_date), HOUR(entry_date), is_interpolated, read_error"""
        for sensor, day, hour, flag, error, count, *nonnull in query(connection, sql, mapping):
            bits = traffic_bits(flag)
            if bits is None:
                diagnostics["unknown_flag_records"] += count
            if error != "N":
                diagnostics["excluded_quality_records"] += count
                continue
            sid = cell_to_aq[mapping[sensor]]
            for j in range(3):
                if not nonnull[j]:
                    continue
                sets = support[(sid, j)]
                sets[0].add((day, hour))
                sensor_support[(sid, j)].add(sensor)
                if bits is not None:
                    sets[1 if bits[j] == "0" else 2].add((day, hour))
    for sid in aq_cells:
        for j, name in enumerate(TRAFFIC):
            sets = support[(sid, j)]
            out.append(dict(sensor_id=sid, variable=name, source="traffic_cell",
                            expected_hours=expected, available_hours=len(sets[0]),
                            observed_hours=len(sets[1]), interpolated_hours=len(sets[2]),
                            source_sensors=len(sensor_support[(sid, j)])))
    return out, diagnostics


@click.command()
@click.option("--year", type=click.IntRange(2000, 2100), default=2024)
@click.option("--months", default="1,4,7,10", help="Comma-separated months.")
@click.option("--test-group", type=click.IntRange(0, 9), default=0)
@click.option("--statement-timeout", type=click.IntRange(1, 60), default=20)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(year, months, test_group, statement_timeout, output):
    """Audit coverage without reading TEST/VALIDATION measurements or training."""
    months = [int(m) for m in months.split(",")]
    if not months or len(set(months)) != len(months) or any(m not in range(1, 13) for m in months):
        raise click.ClickException("Use distinct months from 1 to 12.")
    if output.exists():
        raise click.ClickException("Choose a new output directory; existing reports are preserved.")
    historical = ROOT / "output/experiments/basic/metraq_nox_add24_ks_355/data.npz"
    with np.load(historical, allow_pickle=False) as data:
        test_ids = data["test_sensors"][test_group].astype(int).tolist()
    from metraq_dip.data.metraq_db_legacy import metraq_db
    rows, diagnostics = [], []
    with metraq_db.engine.connect() as connection:
        connection.execute(text(f"SET SESSION max_statement_time={statement_timeout}"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        sensors = list(connection.execute(text("SELECT id,utm_x,utm_y FROM merged_sensors WHERE has_NO2=1 ORDER BY id")))
        validation = [r[0] for r in sensors if r[0] not in test_ids][:4]
        train = [r[0] for r in sensors if r[0] not in test_ids + validation]
        aq_cells = {r[0]: cell(r[1], r[2]) for r in sensors if r[0] in train}
        if len(set(aq_cells.values())) != len(aq_cells):
            raise click.ClickException("Multiple AQ sensors share a cell; explicit aggregation is required.")
        traffic = list(connection.execute(text("SELECT id,utm_x,utm_y FROM traffic_sensors WHERE utm_x IS NOT NULL AND utm_y IS NOT NULL")))
        mapping = {r[0]: cell(r[1], r[2]) for r in traffic if cell(r[1], r[2]) in set(aq_cells.values())}
        for month in months:
            start = datetime(year, month, 1)
            end = datetime(year + (month == 12), month % 12 + 1, 1)
            expected = calendar.monthrange(year, month)[1] * 24
            partition = f"p{year}{month:02d}"
            exists = connection.execute(text("SELECT COUNT(*) FROM information_schema.partitions WHERE table_schema=DATABASE() AND table_name='traffic_data' AND partition_name=:p"), {"p": partition}).scalar()
            if not exists:
                raise click.ClickException(f"Missing partition {partition}.")
            click.echo(f"Auditing {year}-{month:02d} (TRAIN only, {partition})", err=True)
            current = aq_coverage(connection, train, start, end, expected)
            tr_rows, diag = traffic_coverage(connection, mapping, aq_cells, start, end, partition, expected)
            for row in current + tr_rows:
                row["month"] = f"{year}-{month:02d}"
            rows.extend(current + tr_rows)
            diagnostics.append(dict(month=f"{year}-{month:02d}", **diag))
        connection.rollback()
    frame = pd.DataFrame(rows)
    frame["observed_fraction"] = frame.observed_hours / frame.expected_hours
    output.mkdir(parents=True)
    frame.to_csv(output / "coverage.csv", index=False)
    metadata = dict(train=train, validation=validation, test=test_ids, test_group=test_group,
                    historical_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),
                    months=months, year=year, grid_m=1000, traffic_bits=TRAFFIC,
                    traffic_quality="read_error = N; null quality excluded",
                    speed_capability="avg_speed only for traffic_data.element_type=M30; urban zeros are unmeasured",
                    aq_quality="is_valid supplied by Ayuntamiento; retained as source quality flag, not independent physical validation",
                    diagnostics=diagnostics, statement_timeout=statement_timeout,
                    notes=["Hours use database naive timestamps; timezone unresolved.",
                           "Observed and interpolated traffic coverage can overlap within a cell/hour.",
                           "Coverage is presence, not independence or full intra-hour support.",
                           "No raw measures, correlations, PCA, or model selection exported."])
    (output / "scope.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = frame.groupby(["month", "variable"])[["expected_hours", "observed_hours", "available_hours"]].sum()
    summary.to_csv(output / "summary.csv")
    percentages = (100 * summary.observed_hours / summary.expected_hours).unstack("month").round(2)
    (output / "report.md").write_text(
        "# Cobertura de auxiliares en TRAIN\n\n"
        "Reparto: 16 estaciones TRAIN; cuatro VALIDATION por orden de identificador; "
        "TEST histórico excluido antes de consultar medidas. Véase scope.json para los IDs exactos.\n\n"
        "Tráfico: presencia de al menos una medida no nula por celda/hora con read_error=N; "
        "procedencia por campo. Velocidad solo en detectores M30; ceros urbanos no son medidas de velocidad. "
        "Los demás ceros se conservan. Esto no valida calidad física ni cobertura intrahoraria.\n\n"
        "is_valid de contaminación/meteorología es la bandera proporcionada por el Ayuntamiento.\n\n"
        "Los valores interpolados no son fuentes independientes. El número de sensores fuente "
        "de meteorología refiere a registros de la tabla, no a estaciones originales de interpolación.\n\n"
        "La zona horaria, unidades de tráfico y origen de interpolación siguen pendientes. "
        "No se ha ajustado ninguna transformación ni utilizado TEST para selección.\n\n"
        "Reproducir en un directorio nuevo:\n\n```powershell\n"
        f".venv\\Scripts\\python.exe -B scripts/audit_auxiliary_coverage.py --year {year} "
        f"--months {','.join(map(str, months))} --test-group {test_group} "
        f"--statement-timeout {statement_timeout} --output output/auxiliary_coverage_repeat\n```\n\n"
        "Porcentaje de horas/estación con soporte observado:\n\n```\n"
        + percentages.to_string() + "\n```\n\n"
        "Resumen (horas sumadas entre estaciones; no muestras independientes):\n\n```\n"
        + summary.to_string() + "\n```\n", encoding="utf-8")
    click.echo(f"Saved {len(frame)} coverage rows to {output}")


if __name__ == "__main__":
    main()
