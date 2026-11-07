"""
Módulo de persistencia y gestión de base de datos analítica local en DuckDB.
"""
from contextlib import contextmanager
from datetime import date
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import duckdb
import pandas as pd

from src.config import config

logger = logging.getLogger("DatabaseManager")


class DuckDBManager:
    """
    Gestor de base de datos DuckDB para almacenamiento y consulta de series de mercado.
    Implementa transacciones atómicas, inicialización de esquemas e inserciones idempotentes.
    """

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        # Asegurar existencia del directorio data/
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager para asegurar el cierre de conexiones en DuckDB."""
        conn = duckdb.connect(database=str(self.db_path), read_only=False)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Crea la tabla ecb_fx_rates si aún no existe, con su clave primaria e índices."""
        ddl = """
        CREATE TABLE IF NOT EXISTS ecb_fx_rates (
            rate_date DATE PRIMARY KEY,
            base_currency VARCHAR(3) DEFAULT 'EUR',
            quote_currency VARCHAR(3) DEFAULT 'USD',
            spot_rate DOUBLE NOT NULL,
            ingested_at TIMESTAMP NOT NULL
        );
        """
        with self.get_connection() as conn:
            conn.execute(ddl)
            logger.info(f"Esquema inicializado en base de datos: {self.db_path}")

    def get_latest_rate_date(self) -> Optional[date]:
        """
        Consulta la fecha más reciente almacenada en la tabla.

        Returns:
            datetime.date con la fecha máxima o None si la tabla está vacía.
        """
        query = "SELECT MAX(rate_date) FROM ecb_fx_rates;"
        with self.get_connection() as conn:
            result = conn.execute(query).fetchone()
            if result and result[0] is not None:
                return result[0]
            return None

    def upsert_fx_rates(self, df: pd.DataFrame) -> int:
        """
        Inserta registros de forma idempotente. Si la fecha ya existe, actualiza el valor
        y la marca de tiempo de ingestión.

        Args:
            df: DataFrame con las columnas [rate_date, base_currency, quote_currency, spot_rate, ingested_at].

        Returns:
            int: Número de registros procesados.
        """
        if df.empty:
            logger.info("DataFrame vacío recibido para persistencia. No se realizaron cambios.")
            return 0

        upsert_query = """
        INSERT OR REPLACE INTO ecb_fx_rates (
            rate_date, base_currency, quote_currency, spot_rate, ingested_at
        )
        SELECT 
            rate_date, base_currency, quote_currency, spot_rate, ingested_at
        FROM df_staging;
        """

        with self.get_connection() as conn:
            conn.register("df_staging", df)
            conn.execute(upsert_query)
            conn.unregister("df_staging")

        logger.info(f"Persistencia completada: {len(df)} registros insertados/actualizados de forma idempotente.")
        return len(df)

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la tabla para auditoría y comprobación de estado."""
        query = """
        SELECT 
            COUNT(*) AS total_records,
            MIN(rate_date) AS min_date,
            MAX(rate_date) AS max_date,
            (SELECT spot_rate FROM ecb_fx_rates ORDER BY rate_date DESC LIMIT 1) AS latest_rate
        FROM ecb_fx_rates;
        """
        with self.get_connection() as conn:
            row = conn.execute(query).fetchone()
            return {
                "total_records": row[0] if row else 0,
                "min_date": row[1] if row else None,
                "max_date": row[2] if row else None,
                "latest_rate": row[3] if row else None,
            }

