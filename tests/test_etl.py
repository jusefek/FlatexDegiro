"""
Suite de pruebas unitarias y de integración para el pipeline ETL de FX del BCE.
"""
from datetime import date
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.validator import FXDataValidator, DataQualityError
from src.database import DuckDBManager


class TestFXDataValidator:
    """Pruebas de validación y control de calidad de datos."""

    def test_valid_data_passes_validation(self):
        df_raw = pd.DataFrame({
            "TIME_PERIOD": ["2026-10-30", "2026-11-02"],
            "OBS_VALUE": ["1.1205", "1.1216"],
        })
        clean_df = FXDataValidator.validate_and_clean(df_raw)
        assert len(clean_df) == 2
        assert list(clean_df.columns) == ["rate_date", "base_currency", "quote_currency", "spot_rate", "ingested_at"]
        assert clean_df.iloc[0]["spot_rate"] == 1.1205
        assert clean_df.iloc[0]["rate_date"] == date(2026, 10, 30)

    def test_missing_required_columns_raises_error(self):
        df_invalid = pd.DataFrame({"TIME_PERIOD": ["2026-11-02"]})
        with pytest.raises(DataQualityError, match="Faltan columnas obligatorias"):
            FXDataValidator.validate_and_clean(df_invalid)

    def test_negative_or_zero_rates_raises_error(self):
        df_negative = pd.DataFrame({
            "TIME_PERIOD": ["2026-11-02"],
            "OBS_VALUE": ["-1.05"],
        })
        with pytest.raises(DataQualityError, match="<= 0"):
            FXDataValidator.validate_and_clean(df_negative)

    def test_out_of_bounds_rates_raises_error(self):
        df_out_of_bounds = pd.DataFrame({
            "TIME_PERIOD": ["2026-11-02"],
            "OBS_VALUE": ["4.50"],  # Absurdo para EUR/USD
        })
        with pytest.raises(DataQualityError, match="fuera de rango admisible"):
            FXDataValidator.validate_and_clean(df_out_of_bounds)

    def test_invalid_dates_and_non_numerics_handled(self):
        df_dirty = pd.DataFrame({
            "TIME_PERIOD": ["2026-11-02", "INVALID-DATE", "2026-11-03"],
            "OBS_VALUE": ["1.1216", "1.1220", "NOT_A_NUMBER"],
        })
        clean_df = FXDataValidator.validate_and_clean(df_dirty)
        # Sólo la primera fila es válida
        assert len(clean_df) == 1
        assert clean_df.iloc[0]["rate_date"] == date(2026, 11, 2)


class TestDuckDBManager:
    """Pruebas de persistencia e idempotencia en DuckDB."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_market_data.duckdb"
            manager = DuckDBManager(db_path=db_path)
            manager.initialize_schema()
            yield manager

    def test_schema_initialization(self, temp_db):
        summary = temp_db.get_metadata_summary()
        assert summary["total_records"] == 0
        assert temp_db.get_latest_rate_date() is None

    def test_idempotent_upsert(self, temp_db):
        df = pd.DataFrame({
            "rate_date": [date(2026, 11, 1), date(2026, 11, 2)],
            "base_currency": ["EUR", "EUR"],
            "quote_currency": ["USD", "USD"],
            "spot_rate": [1.1200, 1.1216],
            "ingested_at": [pd.Timestamp.now(), pd.Timestamp.now()],
        })

        # Primera inserción
        count_1 = temp_db.upsert_fx_rates(df)
        assert count_1 == 2
        assert temp_db.get_metadata_summary()["total_records"] == 2
        assert temp_db.get_latest_rate_date() == date(2026, 11, 2)

        # Segunda inserción (idénticos datos, debe actualizar sin duplicar)
        count_2 = temp_db.upsert_fx_rates(df)
        assert count_2 == 2
        assert temp_db.get_metadata_summary()["total_records"] == 2

        # Actualización con valor corregido
        df_updated = df.copy()
        df_updated.loc[df_updated["rate_date"] == date(2026, 11, 2), "spot_rate"] = 1.1250
        temp_db.upsert_fx_rates(df_updated)
        assert temp_db.get_metadata_summary()["total_records"] == 2
        assert temp_db.get_metadata_summary()["latest_rate"] == 1.1250

