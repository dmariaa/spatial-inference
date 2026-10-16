"""Build a TRAIN-only hourly table and audit support, constants and spatial equality."""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sqlalchemy import text

from audit_auxiliary_coverage import ROOT, METEO, TRAFFIC, cell, query


def aggregate_traffic(frame, mapping, aq_cells, mode):
    """Equal weight per contributing sensor after its own hourly mean."""
    records = []
    cell_to_aq = {location: sid for sid, location in aq_cells.items()}
    for j, variable in enumerate(TRAFFIC):
        columns = ["sensor", "time", f"v{j}_{mode}", f"n{j}_{mode}"]
        data = frame[columns].copy()
        data.columns = ["traffic_sensor", "time", "value", "records"]
        data["value"] = pd.to_numeric(data.value)
        data["observed_records"] = pd.to_numeric(frame[f"n{j}_observed"])
        data = data[(data.records > 0) & np.isfinite(data.value)].copy()
        data["sensor_id"] = data.traffic_sensor.map(lambda sid: cell_to_aq[mapping[sid]])
        grouped = data.groupby(["sensor_id", "time"]).agg(
            value=("value", "mean"), raw_records=("records", "sum"),
            observed_raw_records=("observed_records", "sum"),
            source_sensors=("traffic_sensor", "nunique"))
        grouped = grouped.reset_index()
        grouped["variable"] = variable
        grouped["mode"] = mode
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def prepare_aq(frame, mode):
    valid = frame.valid.eq(1) & np.isfinite(frame.value)
    valid &= frame.time.dt.minute.eq(0) & frame.time.dt.second.eq(0)
    valid &= frame.provenance.isin([0, 1])
    if mode == "observed":
        valid &= frame.provenance.eq(0)
    data = frame.loc[valid].copy()
    data["variable"] = data.magnitude.map({12: "NOX", **METEO})
    data["mode"] = mode
    data["raw_records"] = 1
    data["observed_raw_records"] = data.provenance.eq(0).astype(int)
    data["source_sensors"] = 1
    keys = ["sensor_id", "time"]
    wind = data[data.magnitude.isin([81, 82])].pivot(index=keys, columns="magnitude", values="value")
    wind = wind.reindex(columns=[81, 82]).dropna()
    wind_provenance = data[data.magnitude.isin([81, 82])].pivot(
        index=keys, columns="magnitude", values="provenance").reindex(index=wind.index, columns=[81, 82])
    derived = []
    for variable, component in [("wind_u", -wind[81] * np.sin(np.deg2rad(wind[82]))),
                                ("wind_v", -wind[81] * np.cos(np.deg2rad(wind[82])))]:
        result = component.rename("value").reset_index()
        result["variable"] = variable
        result["mode"] = mode
        result["raw_records"] = 2
        result["observed_raw_records"] = wind_provenance.eq(0).sum(axis=1).to_numpy()
        result["source_sensors"] = 1
        derived.append(result)
    columns = keys + ["variable", "mode", "value", "raw_records", "observed_raw_records", "source_sensors"]
    return pd.concat([data[columns], *derived], ignore_index=True)


def spatial_equalities(table):
    results = []
    for (month, variable, mode), part in table.groupby(["month", "variable", "mode"]):
        if variable == "NOX":
            continue
        wide = part.pivot(index="time", columns="sensor_id", values="value")
        for a, b in itertools.combinations(wide.columns, 2):
            pair = wide[[a, b]].dropna()
            if pair.empty:
                continue
            equal = int((pair[a] == pair[b]).sum())
            results.append(dict(month=month, variable=variable, mode=mode, sensor_a=a,
                                sensor_b=b, valid_pairs=len(pair), exact_equal=equal,
                                equal_fraction=equal / len(pair),
                                series_equal=equal == len(pair),
                                unique_a=pair[a].nunique(), unique_b=pair[b].nunique()))
    return pd.DataFrame(results)


def audit_findings(stats, pairs):
    ranges = stats.groupby(["variable", "mode"])[["minimum", "maximum"]].agg(
        {"minimum": "min", "maximum": "max"})
    no_variation = stats[(stats["mode"] == "observed") & stats.unique_values.eq(1)]
    counts = no_variation.groupby(["month", "variable"]).size()
    return (
        "\n## Hallazgos para revisar antes de correlaciones\n\n"
        "Grupos observados sin variacion (incluye series con una sola hora, que no permiten evaluar constancia):\n\n```\n"
        + counts.to_string() + "\n```\n\n"
        "Rangos de valores validos segun la BBDD, conservados sin filtro fisico:\n\n```\n"
        + ranges.to_string() + "\n```\n\n"
        "Velocidad: solo detectores M30; los ceros urbanos no se consideran medidas. "
        "is_valid es la bandera de calidad proporcionada por el Ayuntamiento. "
        "Los valores meteorologicos extremos que esa bandera admite requieren una revision separada antes de estandarizar. "
        "No se han eliminado ni corregido.\n\n"
        "Las coincidencias exactas se calculan sobre soporte comun, informado por pareja; "
        "no equivalen a identificar fuentes originales ni descartan dependencia espacial no exacta.\n")


@click.command()
@click.option("--coverage-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--statement-timeout", type=click.IntRange(1, 60), default=20)
@click.option("--sensor-batch", type=click.IntRange(1, 128), default=64)
def main(coverage_dir, output, statement_timeout, sensor_batch):
    """Keep original units; no imputation, normalization, correlation or PCA."""
    if output.exists():
        raise click.ClickException("Choose a new output directory.")
    scope = json.loads((coverage_dir / "scope.json").read_text(encoding="utf-8"))
    train = scope["train"]
    if set(train) & set(scope["test"] + scope["validation"]):
        raise click.ClickException("Overlapping partitions.")
    year = int(scope["year"])
    months = list(map(int, scope["months"]))
    if not 2000 <= year <= 2100 or any(m not in range(1, 13) for m in months):
        raise click.ClickException("Invalid period.")
    from metraq_dip.data.metraq_db_legacy import metraq_db
    tables, integrations = [], []
    with metraq_db.engine.connect() as connection:
        connection.execute(text(f"SET SESSION max_statement_time={statement_timeout}"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        sensors = query(connection, "SELECT id,utm_x,utm_y FROM merged_sensors WHERE id IN :ids", train)
        aq_cells = {r[0]: cell(r[1], r[2]) for r in sensors}
        if set(aq_cells) != set(train) or len(set(aq_cells.values())) != len(train):
            raise click.ClickException("Missing sensors or shared AQ cells.")
        tr = connection.execute(text("SELECT id,utm_x,utm_y FROM traffic_sensors WHERE utm_x IS NOT NULL AND utm_y IS NOT NULL")).fetchall()
        mapping = {r[0]: cell(r[1], r[2]) for r in tr if cell(r[1], r[2]) in set(aq_cells.values())}
        for month in months:
            start = datetime(year, month, 1)
            end = datetime(year + (month == 12), month % 12 + 1, 1)
            partition = f"p{year}{month:02d}"
            click.echo(f"Preparing {year}-{month:02d}: TRAIN only, {partition}", err=True)
            aq_sql = f"""SELECT sensor_id,entry_date,magnitude_id,value,is_valid,is_interpolated
                FROM MAD_merged_aq_data WHERE sensor_id IN :ids AND magnitude_id IN (12,81,82,83,86,87,88,89)
                AND entry_date >= '{start}' AND entry_date < '{end}'"""
            aq = pd.DataFrame(query(connection, aq_sql, train),
                              columns=["sensor_id", "time", "magnitude", "value", "valid", "provenance"])
            aq["time"] = pd.to_datetime(aq.time)
            expressions, names = [], []
            for j, variable in enumerate(TRAFFIC):
                for mode in ["all", "observed"]:
                    condition = "1=1" if mode == "all" else f"SUBSTRING(LPAD(is_interpolated,3,'0'),{j+1},1)='0'"
                    if variable == "avg_speed":
                        condition += " AND element_type='M30'"
                    expressions.extend([f"AVG(CASE WHEN {condition} THEN {variable} END)",
                                        f"COUNT(CASE WHEN {condition} THEN {variable} END)"])
                    names.extend([f"v{j}_{mode}", f"n{j}_{mode}"])
            traffic_sql = f"""SELECT sensor_id, DATE_FORMAT(entry_date,'%Y-%m-%d %H:00:00'),
                {','.join(expressions)}, MIN(integration_period), MAX(integration_period)
                FROM traffic_data PARTITION({partition}) FORCE INDEX(PRIMARY)
                WHERE sensor_id IN :ids AND entry_date >= '{start}' AND entry_date < '{end}'
                AND read_error='N' AND LPAD(is_interpolated,3,'0') IN ('000','001','010','011','100','101','110','111')
                GROUP BY sensor_id, DATE_FORMAT(entry_date,'%Y-%m-%d %H:00:00')"""
            traffic_rows = []
            traffic_ids = sorted(mapping)
            for offset in range(0, len(traffic_ids), sensor_batch):
                traffic_rows.extend(query(connection, traffic_sql, traffic_ids[offset:offset + sensor_batch]))
            traffic = pd.DataFrame(traffic_rows,
                                   columns=["sensor", "time", *names, "period_min", "period_max"])
            traffic["time"] = pd.to_datetime(traffic.time)
            integrations.append(dict(month=f"{year}-{month:02d}", sensor_hours=len(traffic),
                                     period_min=float(traffic.period_min.min()),
                                     period_max=float(traffic.period_max.max()),
                                     sensor_hours_with_variable_period=int((traffic.period_min != traffic.period_max).sum())))
            for mode in ["all", "observed"]:
                data = pd.concat([prepare_aq(aq, mode), aggregate_traffic(traffic, mapping, aq_cells, mode)])
                index = pd.MultiIndex.from_product([train, pd.date_range(start, end, freq="h", inclusive="left"),
                                                    ["NOX", *METEO.values(), "wind_u", "wind_v", *TRAFFIC]],
                                                   names=["sensor_id", "time", "variable"])
                data = data.set_index(["sensor_id", "time", "variable"]).reindex(index).reset_index()
                data["mode"] = mode
                data["available"] = data.value.notna()
                data["raw_records"] = data.raw_records.fillna(0).astype(int)
                data["observed_raw_records"] = data.observed_raw_records.fillna(0).astype(int)
                data["interpolated_raw_records"] = data.raw_records - data.observed_raw_records
                data["source_sensors"] = data.source_sensors.fillna(0).astype(int)
                data["month"] = f"{year}-{month:02d}"
                data["partition"] = "TRAIN"
                tables.append(data)
            del traffic, aq
            del traffic_rows
        connection.rollback()
    table = pd.concat(tables, ignore_index=True)
    table["cell_x"] = table.sensor_id.map(lambda sid: aq_cells[sid][0])
    table["cell_y"] = table.sensor_id.map(lambda sid: aq_cells[sid][1])
    assert set(table.sensor_id) == set(train)
    assert not table.duplicated(["sensor_id", "time", "variable", "mode"]).any()
    stats = table.groupby(["month", "sensor_id", "variable", "mode"]).agg(
        expected_hours=("time", "size"), available_hours=("value", "count"),
        unique_values=("value", "nunique"), minimum=("value", "min"), maximum=("value", "max"),
        std=("value", "std"), zero_hours=("value", lambda x: int(x.eq(0).sum())),
        median_source_sensors=("source_sensors", "median"), median_raw_records=("raw_records", "median"))
    stats["constant"] = stats.unique_values.eq(1)
    stats["insufficient_support"] = stats.available_hours.lt(2)
    stats["empty"] = stats.available_hours.eq(0)
    pairs = spatial_equalities(table)
    output.mkdir(parents=True)
    table.to_parquet(output / "hourly_train.parquet", index=False)
    stats.to_csv(output / "variable_audit.csv")
    pairs.to_csv(output / "spatial_equalities.csv", index=False)
    scope.update(aggregation="Arithmetic mean within sensor/hour, then equal-weight sensor means in cell; per-field provenance. No time weighting.",
                 modes={"observed": "valid observed components only", "all": "valid observed plus interpolated components"},
                 traffic_quality="read_error=N; null quality and unknown flags excluded",
                 speed_capability="avg_speed only for traffic_data.element_type=M30; urban zeros unmeasured",
                 aq_quality="is_valid supplied by Ayuntamiento; no extra physical filter applied",
                 integration_diagnostics=integrations, statement_timeout=statement_timeout,
                 sensor_batch=sensor_batch)
    (output / "scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    counts = stats.reset_index().groupby(["variable", "mode"])[["constant", "empty"]].sum()
    exact = pairs[pairs.series_equal & pairs.unique_a.gt(1)]
    report = (
        "# Tabla horaria y auditoria de auxiliares TRAIN\n\n"
        f"{len(table):,} filas; meses {months}, ano {year}. TEST y VALIDATION excluidos antes de consultar medidas.\n\n"
        "Unidad: hora y sensor AQ/celda de 1 km. Ausencias: NaN y available=False; ceros conservados. "
        "Modos observed y all en filas distintas. NOX es objetivo separado. No se ajustan transformaciones.\n\n"
        "Trafico: media aritmetica por sensor/hora y luego media entre sensores de la celda; "
        "no se mezclan campos interpolados en observed. No es una media ponderada por duracion, carriles o capacidad. "
        "Velocidad solo agrega detectores M30; los ceros de detectores urbanos quedan fuera de su soporte. "
        "raw_records y source_sensors permiten auditar soporte. No validan cobertura temporal completa.\n\n"
        "Periodo de integracion: unidades pendientes; diferencias entre registros impiden asumir duracion uniforme.\n\n"
        "Constantes y grupos sin datos (cada grupo = estacion, mes, variable y modo):\n\n```\n"
        + counts.to_string() + "\n```\n\n"
        f"{len(exact)} parejas de series auxiliares no constantes coinciden exactamente sobre su soporte comun. "
        "Ver spatial_equalities.csv: son coincidencias numericas, no prueba de origen compartido. "
        "Los ceros de precipitacion pueden producir coincidencias extensas sin ser duplicados de fuente.\n\n"
        "wind_u/v derivan de velocidad y direccion con la conversion existente. "
        "No incluir ambas representaciones del viento simultaneamente en PCA.\n\n"
        "Pendientes: zona horaria, unidades y semantica de calidad de trafico, origen de interpolacion. "
        "No hay correlaciones, PCA ni inferencia causal en esta fase.\n\n"
        "Reproducir en un directorio nuevo:\n\n```powershell\n"
        ".venv\\Scripts\\python.exe -B scripts/prepare_auxiliary_table.py "
        f"--coverage-dir {coverage_dir.as_posix()} --sensor-batch {sensor_batch} "
        f"--statement-timeout {statement_timeout} --output output/auxiliary_table_repeat\n```\n")
    (output / "report.md").write_text(report + audit_findings(stats.reset_index(), pairs), encoding="utf-8")
    click.echo(f"Saved table and audits to {output}")


if __name__ == "__main__":
    main()
