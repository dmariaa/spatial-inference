"""Source metadata; measurements and sensors use the existing AQBackend."""
import numpy as np
import pandas as pd
from sqlalchemy import text

from metraq_dip.data.aq_backends import get_aq_backend

# Madrid air-quality magnitude codes; meteorology is intentionally excluded.
POLLUTANTS = {
    1: ("SO₂", "µg/m³"), 6: ("CO", "mg/m³"), 7: ("NO", "µg/m³"),
    8: ("NO₂", "µg/m³"), 9: ("PM2.5", "µg/m³"), 10: ("PM10", "µg/m³"),
    12: ("NOx", "µg/m³"), 14: ("O₃", "µg/m³"), 20: ("Tolueno", "µg/m³"),
    30: ("Benceno", "µg/m³"), 35: ("Etilbenceno", "µg/m³"),
}


def load_catalog(source):
    """Actual coverage and fixed observed range, one row per pollutant."""
    if source == "db":
        from metraq_dip.data.metraq_db_legacy import metraq_db
        ids = ",".join(str(i) for i in POLLUTANTS)
        return pd.read_sql_query(text(f"""
            SELECT magnitude_id, MIN(entry_date) AS first_date,
                   MAX(entry_date) AS last_date, MIN(value) AS low, MAX(value) AS high
            FROM MAD_merged_aq_data
            WHERE is_valid AND magnitude_id IN ({ids})
            GROUP BY magnitude_id
        """), metraq_db.engine, parse_dates=["first_date", "last_date"])
    if source != "files":
        raise ValueError("Unknown data source")
    from metraq_dip.data.metraq_files import metraq_files
    # Stream columns in batches instead of loading the full dataset into memory.
    dataset = metraq_files._dataset_for_files(metraq_files._resolve_files())
    summaries = []
    for batch in dataset.scanner(columns=["magnitude_id", "entry_date", "value"],
                                filter=metraq_files._build_filter(magnitudes=list(POLLUTANTS))).to_batches():
        frame = batch.to_pandas()
        frame = frame.loc[np.isfinite(frame.value)]
        summaries.append(frame.groupby("magnitude_id").agg(
            first_date=("entry_date", "min"), last_date=("entry_date", "max"),
            low=("value", "min"), high=("value", "max")))
    if not summaries:
        return pd.DataFrame(columns=["magnitude_id", "first_date", "last_date", "low", "high"])
    return pd.concat(summaries).groupby(level=0).agg(
        {"first_date": "min", "last_date": "max", "low": "min", "high": "max"}).reset_index()


def load_sensors(source, magnitudes):
    return get_aq_backend(dataset="metraq", backend=source).get_sensors(magnitudes=list(magnitudes))


def load_measurements(source, magnitude, timestamp):
    return get_aq_backend(dataset="metraq", backend=source).get_measurements(
        start_date=timestamp, end_date=timestamp, magnitudes=[int(magnitude)])
