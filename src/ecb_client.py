"""
Cliente HTTP resiliente para el ECB Data Portal REST API (SDMX-CSV).
"""
import io
import logging
from typing import Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import config

logger = logging.getLogger("ECBClient")


class ECBClient:
    """
    Cliente robusto para la extracción de series temporales del Banco Central Europeo.
    Implementa reintentos con backoff exponencial y gestión adecuada de días no hábiles.
    """

    def __init__(
        self,
        base_url: str = config.ECB_BASE_URL,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        backoff_factor: float = config.BACKOFF_FACTOR,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

        # Configurar estrategia de reintentos para errores transitorios y rate limiting
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=config.STATUS_FORCELIST,
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_fx_data(self, start_period: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Descarga datos de tipo de cambio en formato CSV estructurado.

        Args:
            start_period: Fecha de inicio en formato ISO 'YYYY-MM-DD'. Si es None, descarga todo el histórico disponible.

        Returns:
            pd.DataFrame con los datos brutos recibidos, o None si no hay cotizaciones publicadas.
        """
        params = {"format": "csvdata"}
        if start_period:
            params["startPeriod"] = start_period

        logger.info(
            f"Solicitando datos a la API del BCE (URL: {self.base_url}, startPeriod: {start_period or 'Histórico completo'})..."
        )

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={"Accept": "text/csv, application/vnd.sdmx.data+csv;version=1.0.0"},
            )

            # Manejo específico de respuestas del BCE:
            # 1. Código 404: El BCE retorna 404 cuando no existen observaciones para el rango (p.ej. fines de semana / festivos)
            if response.status_code == 404:
                logger.info(
                    f"Respuesta HTTP 404 recibida para startPeriod={start_period}. "
                    "Interpretado como día inhábil bancario (TARGET2) o sin observaciones publicadas."
                )
                return None

            # 2. Código 400: Puede ocurrir si el periodo solicitado es estrictamente futuro
            if response.status_code == 400:
                logger.info(
                    f"Respuesta HTTP 400 recibida para startPeriod={start_period}. "
                    "El rango solicitado excede la última fecha hábil publicada por el BCE."
                )
                return None

            # 3. Validar otros códigos de error HTTP
            response.raise_for_status()

            content = response.text.strip()
            if not content:
                logger.info(
                    f"Respuesta con cuerpo vacío para startPeriod={start_period}. "
                    "No hay nuevas observaciones registradas."
                )
                return None

            # Parsear CSV
            df = pd.read_csv(io.StringIO(content))

            if df.empty:
                logger.info("El CSV devuelto por la API del BCE no contiene registros de datos.")
                return None

            logger.info(f"Descarga exitosa del BCE: {len(df)} filas brutas recibidas.")
            return df

        except requests.exceptions.RequestException as exc:
            logger.error(f"Fallo crítico de red o protocolo al conectar con el ECB Data Portal: {exc}")
            raise

