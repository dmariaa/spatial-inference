"""Resumable, read-only audit of every MAD_merged_aq_data row against its catalog."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SQL = """SELECT d.magnitude_id, d.is_valid, d.is_interpolated,
 COUNT(*) AS records,
 SUM(CASE WHEN d.value < m.min_value THEN 1 ELSE 0 END) AS below_min,
 SUM(CASE WHEN d.value > m.max_value THEN 1 ELSE 0 END) AS above_max,
 SUM(CASE WHEN d.value < m.min_value OR d.value > m.max_value THEN 1 ELSE 0 END) AS outside_bounds,
 SUM(CASE WHEN m.id IS NULL THEN 1 ELSE 0 END) AS missing_catalog,
 SUM(CASE WHEN d.value IS NULL THEN 1 ELSE 0 END) AS null_value,
 SUM(CASE WHEN m.id IS NOT NULL AND (m.min_value IS NULL OR m.max_value IS NULL) THEN 1 ELSE 0 END) AS missing_bounds,
 MIN(CASE WHEN d.value < m.min_value OR d.value > m.max_value THEN d.value END) AS outside_min_value,
 MAX(CASE WHEN d.value < m.min_value OR d.value > m.max_value THEN d.value END) AS outside_max_value
 FROM MAD_merged_aq_data d FORCE INDEX(idx_entry_date)
 LEFT JOIN aq_magnitudes m ON m.id=d.magnitude_id
 WHERE d.entry_date >= :start AND d.entry_date < :end
 AND (m.type IS NULL OR m.type <> 'traffic')
 GROUP BY d.magnitude_id,d.is_valid,d.is_interpolated"""


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return float(value)


def next_month(value):
    return datetime(value.year + (value.month == 12), value.month % 12 + 1, 1)


def report_breakdown(summary):
    data = summary.copy()
    data["outside_flagged_valid"] = data.outside_bounds.where(data.is_valid.eq(1), 0)
    types = data.groupby("type", dropna=False)[["records", "outside_bounds", "outside_flagged_valid"]].sum()
    meteo = data[data.type.eq("meteo")].groupby("name")[["below_min", "above_max", "outside_bounds", "outside_flagged_valid"]].sum()
    return ("\n## Desglose por tipo de magnitud\n\n```\n" + types.to_string()
            + "\n```\n\n## Meteorologia\n\n```\n" + meteo.to_string()
            + "\n```\n\nLas magnitudes de tipo traffic quedan excluidas del recuento y sus denominadores.\n")


@click.command()
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--statement-timeout", type=click.IntRange(1, 60), default=20)
def main(output, statement_timeout):
    """Audit the full table; existing checkpoints allow safe resumption."""
    from metraq_dip.data.metraq_db_legacy import metraq_db
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "completed_months.jsonl"
    complete = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            complete[item["month"]] = item
    scope_path = output / "scope.json"
    with metraq_db.engine.connect() as connection:
        connection.execute(text(f"SET SESSION max_statement_time={statement_timeout}"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        if scope_path.exists():
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
        else:
            lower = connection.execute(text("SELECT entry_date FROM MAD_merged_aq_data FORCE INDEX(idx_entry_date) ORDER BY entry_date LIMIT 1")).scalar()
            upper = connection.execute(text("SELECT entry_date FROM MAD_merged_aq_data FORCE INDEX(idx_entry_date) ORDER BY entry_date DESC LIMIT 1")).scalar()
            if lower is None or upper is None:
                raise click.ClickException("Empty table.")
            scope = dict(table="MAD_merged_aq_data", start=lower.isoformat(), last_timestamp=upper.isoformat(),
                         selection="All sensors; traffic magnitudes excluded; no split filters", created_utc=datetime.utcnow().isoformat(),
                         query_timeout=statement_timeout, consistency="Read-only transaction per query; not one whole-table snapshot")
            catalog = connection.execute(text("SELECT id,name,units,type,min_value,max_value FROM aq_magnitudes"))
            scope["catalog"] = [{k: plain(v) for k, v in row.items()} for row in catalog.mappings()]
            scope_path.write_text(json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8")
        connection.rollback()
        lower = datetime.fromisoformat(scope["start"])
        upper = datetime.fromisoformat(scope["last_timestamp"])
        month = datetime(lower.year, lower.month, 1)
        stop = next_month(datetime(upper.year, upper.month, 1))
        periods = []
        while month < stop:
            periods.append((month, next_month(month)))
            month = next_month(month)
        for index, (start, end) in enumerate(periods, 1):
            key = start.strftime("%Y-%m")
            if key in complete:
                continue
            begin = time.monotonic()
            def fetch(a, b):
                connection.execute(text("START TRANSACTION READ ONLY"))
                try:
                    result = connection.execute(text(SQL), {"start": a, "end": b})
                    return [{k: plain(v) for k, v in row.items()} for row in result.mappings()]
                finally:
                    connection.rollback()
            try:
                records = fetch(start, end)
                batches = 1
            except Exception as exc:
                code = getattr(getattr(exc, "orig", None), "args", (None,))[0]
                if code != 1969:
                    raise click.ClickException(f"Database query failed ({type(exc).__name__}, code {code}); checkpoints retained.") from None
                click.echo(f"{key}: monthly timeout; switching to daily queries", err=True)
                records, batches = [], 0
                day = start
                while day < end:
                    following = min(day + timedelta(days=1), end)
                    records.extend(fetch(day, following))
                    batches += 1
                    day = following
            item = dict(month=key, rows=records, batches=batches, elapsed_seconds=round(time.monotonic()-begin, 3))
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            complete[key] = item
            outside = int(sum(r["outside_bounds"] for r in records))
            count = int(sum(r["records"] for r in records))
            click.echo(f"{index}/{len(periods)} {key}: {count:,} records, {outside:,} outside ({item['elapsed_seconds']} s)", err=True)
            (output / "progress.json").write_text(json.dumps(dict(completed_months=len(complete),total_months=len(periods),last_month=key),indent=2),encoding="utf-8")
    rows = [dict(month=month, **row) for month, item in complete.items() for row in item["rows"]]
    data = pd.DataFrame(rows)
    excluded_ids = {item["id"] for item in scope["catalog"] if item["type"] == "traffic"}
    data = data[~data.magnitude_id.isin(excluded_ids)].copy()
    scope["selection"] = "All sensors; traffic magnitudes excluded; no split filters"
    scope["excluded_magnitude_type"] = "traffic"
    scope_path.write_text(json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["magnitude_id", "is_valid", "is_interpolated"]
    counts = ["records", "below_min", "above_max", "outside_bounds", "missing_catalog", "null_value", "missing_bounds"]
    catalog = pd.DataFrame(scope["catalog"])
    summary = data.groupby(keys, dropna=False)[counts].sum().reset_index().merge(catalog,left_on="magnitude_id",right_on="id",how="left")
    summary.to_csv(output / "summary.csv", index=False)
    data.to_csv(output / "monthly_counts.csv", index=False)
    violations = summary[summary.outside_bounds > 0]
    violations.to_csv(output / "violations.csv", index=False)
    total = int(summary.records.sum())
    outside = int(summary.outside_bounds.sum())
    flagged_valid = int(summary.loc[summary.is_valid.eq(1),"outside_bounds"].sum())
    (output / "report.md").write_text(
        "# Auditoria completa de limites del catalogo\n\n"
        f"Tabla: MAD_merged_aq_data. Periodo: {scope['start']} a {scope['last_timestamp']}. "
        "Todos los sensores; magnitudes de trafico excluidas; sin filtros de splits. BBDD consultada en solo lectura.\n\n"
        f"Registros examinados: {total:,}. Fuera de limites: {outside:,}. "
        f"Fuera de limites con is_valid=TRUE: {flagged_valid:,}.\n\n"
        "Se compara estrictamente value < min_value o value > max_value; los extremos iguales se admiten. "
        "is_valid es la bandera suministrada por el Ayuntamiento. Los incumplimientos son respecto al catalogo "
        "actual, no una validacion fisica independiente. Magnitudes sinteticas o limites de referencia pueden "
        "requerir una interpretacion distinta. Se informa is_interpolated tal como se almacena.\n\n"
        "Nulls, magnitudes sin catalogo y limites incompletos se cuentan por separado. "
        "No se ha corregido ni eliminado ningun registro. Los periodos son contiguos y disjuntos. "
        "Cada consulta tiene su transaccion de solo lectura; no es un snapshot global simultaneo.\n\n"
        "Archivos: summary.csv (magnitud/calidad/procedencia), monthly_counts.csv (periodos), "
        "violations.csv (solo grupos con incumplimientos), scope.json (limites y alcance), "
        "completed_months.jsonl (reanudacion).\n\n"
        "Reproducir en un directorio nuevo, o reanudar el mismo:\n\n```powershell\n"
        f".venv\\Scripts\\python.exe -B scripts/audit_catalog_bounds.py --output {output.as_posix()} "
        f"--statement-timeout {statement_timeout}\n```\n" + report_breakdown(summary),encoding="utf-8")
    click.echo(f"COMPLETE: {total:,} records, {outside:,} outside, {flagged_valid:,} outside and valid")


if __name__ == "__main__":
    main()
