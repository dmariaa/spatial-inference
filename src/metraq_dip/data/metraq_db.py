import os
from threading import Lock
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text


def _metraq_db_generator():
    class MetraqDB:
        def __init__(self):
            self.connection_string = "mariadb+pymysql://root:capo.2024$@capo-escobar.etsii.urjc.es:3306/traffic-database?collation=utf8mb4_general_ci"
            self._engine = None
            self._engine_pid = None
            self._connection_lock = Lock()

        @property
        def engine(self):
            current_pid = os.getpid()

            with self._connection_lock:
                if (
                    self._engine is None
                    or self._engine_pid != current_pid
                ):
                    if self._engine is not None:
                        self._engine.dispose()

                    self._engine = create_engine(
                        self.connection_string,
                        echo=False,
                        pool_pre_ping=True,
                        pool_recycle=1800,
                    )
                    self._engine_pid = current_pid

                return self._engine

        @property
        def connection(self):
            # Backward-compatible alias: return an SQLAlchemy Connectable.
            return self.engine

        def get_sensors(self, *, magnitudes: List[int], sensors: List[int] = None) -> pd.DataFrame:
            aq_sensors_query = (f"SELECT id, utm_x, utm_y"
                                f" FROM merged_sensors "
                                f" WHERE id IN (SELECT DISTINCT sensor_id "
                                f"             FROM MAD_merged_aq_data "
                                f"             WHERE magnitude_id IN ({','.join(map(str, magnitudes))}))")

            df = pd.read_sql_query(text(aq_sensors_query), con=self.engine)
            df["id"] = df["id"].astype(int)

            if sensors is not None:
                df = df[df["id"].isin(sensors)]

            return df

        def execute(self, sql: str, data: dict = None):
            with self.engine.connect() as connection:
                if data is not None:
                    results = connection.execute(text(sql), data)
                else:
                    results = connection.execute(text(sql))

                if results is not None and results.returns_rows:
                    return results.fetchall()

                connection.commit()
                return None

    return MetraqDB()

metraq_db = _metraq_db_generator()
