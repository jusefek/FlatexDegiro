"""
Módulo de validación de calidad de datos (Data Quality Checks) para FX Rates.
"""
from datetime import datetime, timezone
import logging
from typing import Tuple
import pandas as pd

from src.config import config

logger = logging.getLogger("DataValidator")


class DataQualityError(Exception):
    """Excepción lanzada cuando los datos no superan los filtros de integridad o coherencia económica."""
    pass


class FXDataValidator:
    """
    Validador financiero y estructural para series de tipos de cambio del BCE.
    Garantiza consistencia de esquemas, ausencia de nulos y sanidad económica de los precios.
    """

    @staticmethod
    def validate_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Valida, transforma y limpia el DataFrame bruto recibido del BCE.

        Args:
            df_raw: DataFrame original con las columnas del estándar SDMX-CSV.

        Returns:
            pd.DataFrame limpio y validado listo para la ingestión en DuckDB.

        Raises:
            DataQualityError: Si faltan columnas esenciales o se violan restricciones críticas.
        """
        if df_raw is None or df_raw.empty:
            raise DataQualityError("El conjunto de datos recibido está vacío o es nulo.")

        # 1. Comprobación de presencia de columnas obligatorias
        required_cols = {"TIME_PERIOD", "OBS_VALUE"}
        missing_cols = required_cols - set(df_raw.columns)
        if missing_cols:
            raise DataQualityError(f"Faltan columnas obligatorias del BCE: {missing_cols}")

        df = df_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()

        # 2. Parseo y tipado estricto
        # Parseo de fechas (formato ISO YYYY-MM-DD)
        df["rate_date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m-%d", errors="coerce").dt.date
        invalid_dates = df["rate_date"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Se encontraron {invalid_dates} registros con formato de fecha inválido. Descartando...")
            df = df.dropna(subset=["rate_date"])

        # Parseo de tipos de cambio (float numérico)
        df["spot_rate"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        invalid_rates = df["spot_rate"].isna().sum()
        if invalid_rates > 0:
            logger.warning(f"Se encontraron {invalid_rates} observaciones con valor numérico no válido. Descartando...")
            df = df.dropna(subset=["spot_rate"])

        if df.empty:
            raise DataQualityError("No quedan registros válidos tras el parseo de fechas y tipos de cambio.")

        # 3. Validaciones de integridad y coherencia económica (Sanity Checks)
        # a) Estricta positividad
        non_positive = (df["spot_rate"] <= 0).sum()
        if non_positive > 0:
            raise DataQualityError(f"Error crítico de datos: se encontraron {non_positive} tipos de cambio <= 0.")

        # b) Rango económico razonable para EUR/USD (ej. [0.5, 2.5])
        out_of_bounds = df[
            (df["spot_rate"] < config.MIN_SPOT_RATE) | (df["spot_rate"] > config.MAX_SPOT_RATE)
        ]
        if not out_of_bounds.empty:
            anomaly_sample = out_of_bounds[["rate_date", "spot_rate"]].to_dict(orient="records")
            raise DataQualityError(
                f"Valores de tipo de cambio fuera de rango admisible [{config.MIN_SPOT_RATE}, {config.MAX_SPOT_RATE}]: "
                f"{anomaly_sample}"
            )

        # 4. Deduplicación y ordenación
        # Si hubiera duplicados por fecha en la serie, conservar la última ocurrencia
        duplicates_count = df.duplicated(subset=["rate_date"]).sum()
        if duplicates_count > 0:
            logger.warning(f"Se detectaron {duplicates_count} fechas duplicadas. Manteniendo última observación.")
            df = df.drop_duplicates(subset=["rate_date"], keep="last")

        df = df.sort_values(by="rate_date", ascending=True).reset_index(drop=True)

        # 5. Enriquecimiento con metadatos para auditoría y linaje
        now_utc = datetime.now(timezone.utc)
        df["base_currency"] = config.BASE_CURRENCY
        df["quote_currency"] = config.QUOTE_CURRENCY
        df["ingested_at"] = now_utc

        # Seleccionar y ordenar columnas para el esquema de la tabla
        final_df = df[["rate_date", "base_currency", "quote_currency", "spot_rate", "ingested_at"]]

        logger.info(
            f"Validación de calidad superada exitosamente para {len(final_df)} registros. "
            f"Rango temporal: [{final_df['rate_date'].min()} a {final_df['rate_date'].max()}], "
            f"Rango de tipos: [{final_df['spot_rate'].min():.4f} - {final_df['spot_rate'].max():.4f}]."
        )
        return final_df

