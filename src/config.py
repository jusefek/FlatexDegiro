"""
Configuración centralizada para el pipeline ETL de FX del Banco Central Europeo.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    # Rutas de almacenamiento
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "market_data.duckdb"

    # Especificaciones de la serie del Banco Central Europeo (BCE / ECB Data Portal)
    # EXR: Exchange Rates
    # D: Daily frequency
    # USD: Currency quoted against EUR
    # EUR: Currency denomination (Base: 1 EUR = X USD)
    # SP00: Foreign exchange reference rate
    # A: Average/standard
    ECB_SERIES_KEY: str = "EXR.D.USD.EUR.SP00.A"
    ECB_BASE_URL: str = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
    
    # Parámetros por defecto
    INITIAL_START_DATE: str = "2015-01-01"
    BASE_CURRENCY: str = "EUR"
    QUOTE_CURRENCY: str = "USD"

    # Umbrales de calidad del dato (Economic Sanity Checks)
    MIN_SPOT_RATE: float = 0.50
    MAX_SPOT_RATE: float = 2.50

    # Resiliencia de red y reintentos HTTP
    REQUEST_TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 5
    BACKOFF_FACTOR: float = 1.5
    STATUS_FORCELIST: tuple = (429, 500, 502, 503, 504)


config = PipelineConfig()

