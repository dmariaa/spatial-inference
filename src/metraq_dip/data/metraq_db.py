from typing import List

import pandas as pd
from sqlalchemy import create_engine, text


def _metraq_db_generator():
    class MetraqDB:
        def __init__(self):
            self.connection_string = "mariadb+pymysql://root:capo.2024$@capo-escobar.etsii.urjc.es:3306/traffic-database?collation=utf8mb4_general_ci"
            self.engine = create_engine(self.connection_string, echo=False)
            self.connection = self.engine.connect()

        def get_sensors(self, *, magnitudes: List[int], sensors: List[int] = None) -> pd.DataFrame:
            aq_sensors_query = (f"SELECT id, utm_x, utm_y"
                                f" FROM merged_sensors "
                                f" WHERE id IN (SELECT DISTINCT sensor_id "
                                f"             FROM MAD_merged_aq_data "
                                f"             WHERE magnitude_id IN ({','.join(map(str, magnitudes))}))")

            df = pd.read_sql_query(aq_sensors_query, con=self.connection, parse_dates=['entry_date'])
            df["id"] = df["id"].astype(int)

            if sensors is not None:
                df = df[df["id"].isin(sensors)]

            return df

        def execute(self, sql: str, data: dict = None):
            if data is not None:
                results = self.connection.execution_options(stream_results=True).execute(text(sql), data)
            else:
                results = self.connection.execution_options(stream_results=True).execute(text(sql))

            if results is not None and results.returns_rows:
                return results

            return None

    return MetraqDB()

metraq_db = _metraq_db_generator()