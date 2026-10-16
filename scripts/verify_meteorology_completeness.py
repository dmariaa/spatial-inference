"""Check actual database rows against the complete station/magnitude/hour grid."""
from pathlib import Path
import json
import sys

import click
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@click.command()
@click.option("--start", default="2019-01")
@click.option("--end", default="2024-12")
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(start, end, output):
    """No is_valid, provenance or range filters: this tests row existence."""
    if output.exists():
        raise click.ClickException("Choose a new output directory.")
    periods = pd.period_range(start, end, freq="M")
    if not len(periods):
        raise click.ClickException("Empty period.")
    from metraq_dip.data.metraq_db_legacy import metraq_db
    results, unexpected = [], []
    with metraq_db.engine.connect() as connection:
        connection.execute(text("SET SESSION max_statement_time=15"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        sensors = [r[0] for r in connection.execute(text("SELECT id FROM merged_sensors WHERE has_NO2=1 ORDER BY id"))]
        magnitudes = [r[0] for r in connection.execute(text("SELECT id FROM aq_magnitudes WHERE type='meteo' ORDER BY id"))]
        connection.rollback()
        for index, period in enumerate(periods, 1):
            params = {"start": period.start_time.to_pydatetime(), "end": (period + 1).start_time.to_pydatetime()}
            expected = len(pd.date_range(period.start_time, (period+1).start_time, freq="h", inclusive="left"))
            connection.execute(text("START TRANSACTION READ ONLY"))
            sql = """SELECT d.sensor_id,d.magnitude_id,COUNT(*) AS rows_found,
                COUNT(DISTINCT CASE WHEN MINUTE(d.entry_date)=0 AND SECOND(d.entry_date)=0 THEN d.entry_date END) AS distinct_hours,
                SUM(CASE WHEN MINUTE(d.entry_date)<>0 OR SECOND(d.entry_date)<>0 THEN 1 ELSE 0 END) AS off_hour_rows,
                SUM(CASE WHEN d.value IS NULL THEN 1 ELSE 0 END) AS null_values
                FROM MAD_merged_aq_data d FORCE INDEX(idx_entry_date)
                JOIN aq_magnitudes m ON m.id=d.magnitude_id
                WHERE m.type='meteo' AND d.entry_date>=:start AND d.entry_date<:end
                GROUP BY d.sensor_id,d.magnitude_id"""
            found = {(r[0],r[1]): r for r in connection.execute(text(sql), params)}
            connection.rollback()
            for sid in sensors:
                for mag in magnitudes:
                    r = found.pop((sid,mag), (sid,mag,0,0,0,0))
                    results.append(dict(month=str(period),sensor_id=sid,magnitude_id=mag,
                                        expected_hours=expected,rows_found=int(r[2]),distinct_hours=int(r[3]),
                                        missing_hours=expected-int(r[3]),off_hour_rows=int(r[4]),null_values=int(r[5])))
            unexpected.extend(dict(month=str(period),sensor_id=r[0],magnitude_id=r[1],rows_found=int(r[2])) for r in found.values())
            if index % 12 == 0:
                click.echo(f"Verified {index}/{len(periods)} months through {period}",err=True)
    frame = pd.DataFrame(results)
    assert (frame.missing_hours>=0).all()
    output.mkdir(parents=True)
    frame.to_csv(output/"completeness.csv",index=False)
    anomalies = frame[(frame.missing_hours>0)|(frame.off_hour_rows>0)|(frame.null_values>0)|frame.rows_found.ne(frame.distinct_hours)]
    anomalies.to_csv(output/"anomalies.csv",index=False)
    summary = dict(start=start,end=end,sensors=sensors,magnitudes=magnitudes,
                   expected_rows=int(frame.expected_hours.sum()),actual_rows=int(frame.rows_found.sum()),
                   missing_hours=int(frame.missing_hours.sum()),null_values=int(frame.null_values.sum()),
                   off_hour_rows=int(frame.off_hour_rows.sum()),unexpected_groups=unexpected,
                   filters="No validity, provenance, range or split filters",
                   interpretation="Existence of rows, not observed coverage or quality")
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output/"report.md").write_text(
        "# Completitud meteorologica en la BBDD\n\n"
        f"Periodo {start} a {end}, {len(sensors)} estaciones AQ y {len(magnitudes)} magnitudes meteorologicas.\n\n"
        f"Registros esperados: {summary['expected_rows']:,}; encontrados: {summary['actual_rows']:,}. "
        f"Combinaciones estacion/magnitud/hora ausentes: {summary['missing_hours']:,}. "
        f"Valores nulos: {summary['null_values']:,}. Registros fuera de hora exacta: {summary['off_hour_rows']:,}.\n\n"
        "Se cuenta cada hora distinta de cada estacion y magnitud en intervalos mensuales disjuntos; "
        "no se filtra is_valid, is_interpolated ni los limites. La existencia no equivale a calidad ni a observacion original. "
        "Se conserva el calendario horario de timestamps de la BBDD, sin reinterpretar zona horaria.\n\n"
        "Comprobacion reproducible:\n\n```powershell\n"
        f".venv\\Scripts\\python.exe -B scripts/verify_meteorology_completeness.py --start {start} --end {end} "
        "--output output/meteorology_completeness_repeat\n```\n",encoding="utf-8")
    click.echo(json.dumps({k:summary[k] for k in ['expected_rows','actual_rows','missing_hours','null_values','off_hour_rows']}))


if __name__ == "__main__":
    main()
