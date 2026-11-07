#!/usr/bin/env python3
"""
Pipeline ETL de Ingesta de Tipos de Cambio del Banco Central Europeo (ECB Data Portal).
Caso Práctico FlatexDEGIRO - Tarea 1a: Financial Risk Data Engineering.

Uso:
    python etl_ecb.py --mode initial [--start-date 2015-01-01]
    python etl_ecb.py --mode daily
"""
import argparse
from datetime import datetime, timedelta, timezone, date
import logging
import sys
import time

from src.config import config
from src.ecb_client import ECBClient
from src.validator import FXDataValidator, DataQualityError
from src.database import DuckDBManager


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configura el logging estructurado en consola."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("ETL-Orchestrator")


def run_initial_load(client: ECBClient, db: DuckDBManager, start_date: str, logger: logging.Logger) -> None:
    """
    Ejecuta la carga inicial histórica de tipos de cambio USD/EUR.
    """
    logger.info("=" * 70)
    logger.info(f"INICIANDO CARGA HISTÓRICA INICIAL (startPeriod={start_date})")
    logger.info("=" * 70)
    start_time = time.time()

    # 1. Asegurar esquema en DuckDB
    db.initialize_schema()

    # 2. Extracción desde la API del BCE
    df_raw = client.fetch_fx_data(start_period=start_date)
    if df_raw is None or df_raw.empty:
        logger.error("No se recibieron datos del BCE para la carga inicial. Proceso cancelado.")
        sys.exit(1)

    # 3. Validación y Control de Calidad
    try:
        df_clean = FXDataValidator.validate_and_clean(df_raw)
    except DataQualityError as err:
        logger.error(f"Fallo en control de calidad de datos durante la carga inicial: {err}")
        sys.exit(1)

    # 4. Ingestión idempotente en DuckDB
    inserted_count = db.upsert_fx_rates(df_clean)

    # 5. Resumen final de auditoría
    elapsed = time.time() - start_time
    summary = db.get_metadata_summary()
    logger.info("-" * 70)
    logger.info("RESUMEN DE CARGA INICIAL:")
    logger.info(f" - Registros procesados en esta ejecución: {inserted_count:,}")
    logger.info(f" - Total registros en tabla ecb_fx_rates:  {summary['total_records']:,}")
    logger.info(f" - Rango temporal almacenado:              {summary['min_date']} a {summary['max_date']}")
    logger.info(f" - Último tipo de cambio registrado:       1 EUR = {summary['latest_rate']:.4f} USD")
    logger.info(f" - Tiempo de ejecución:                    {elapsed:.2f} segundos")
    logger.info("=" * 70)


def run_daily_update(client: ECBClient, db: DuckDBManager, logger: logging.Logger) -> None:
    """
    Ejecuta la actualización incremental diaria de tipos de cambio.
    Detecta la última fecha almacenada y solicita únicamente nuevos registros.
    """
    logger.info("=" * 70)
    logger.info("INICIANDO ACTUALIZACIÓN INCREMENTAL DIARIA (DAILY UPDATE)")
    logger.info("=" * 70)
    start_time = time.time()

    # 1. Asegurar esquema en DuckDB
    db.initialize_schema()

    # 2. Consultar última fecha registrada
    latest_stored_date = db.get_latest_rate_date()

    if latest_stored_date is None:
        logger.warning(
            "La base de datos está vacía. No existe histórico previo para actualización incremental. "
            f"Ejecutando carga inicial desde {config.INITIAL_START_DATE} por defecto..."
        )
        run_initial_load(client, db, config.INITIAL_START_DATE, logger)
        return

    logger.info(f"Última fecha registrada en base de datos local: {latest_stored_date}")

    # 3. Determinar fecha de consulta
    next_date = latest_stored_date + timedelta(days=1)
    today_utc = datetime.now(timezone.utc).date()

    if next_date > today_utc:
        logger.info(
            f"El almacén local ya está completamente al día ({latest_stored_date}). "
            "No existen observaciones posteriores que solicitar."
        )
        logger.info("=" * 70)
        return

    logger.info(f"Solicitando observaciones del BCE a partir de: {next_date.isoformat()}")

    # 4. Extracción desde API del BCE
    df_raw = client.fetch_fx_data(start_period=next_date.isoformat())

    if df_raw is None or df_raw.empty:
        logger.info(
            f"No hay nuevas cotizaciones publicadas por el BCE desde {next_date}. "
            "Causa esperada: fin de semana, festivo bancario (TARGET2) o publicación pendiente (habitual ~16:00 CET)."
        )
        logger.info("Estado de la base de datos: SIN CAMBIOS (consistente y al día).")
        logger.info("=" * 70)
        return

    # 5. Validación y Control de Calidad
    try:
        df_clean = FXDataValidator.validate_and_clean(df_raw)
    except DataQualityError as err:
        logger.error(f"Fallo en control de calidad en la actualización diaria: {err}")
        sys.exit(1)

    # 6. Ingestión idempotente en DuckDB
    inserted_count = db.upsert_fx_rates(df_clean)

    # 7. Resumen final de auditoría
    elapsed = time.time() - start_time
    summary = db.get_metadata_summary()
    logger.info("-" * 70)
    logger.info("RESUMEN DE ACTUALIZACIÓN DIARIA:")
    logger.info(f" - Nuevos registros incorporados:          {inserted_count}")
    logger.info(f" - Total registros en tabla ecb_fx_rates:  {summary['total_records']:,}")
    logger.info(f" - Nueva fecha máxima registrada:          {summary['max_date']}")
    logger.info(f" - Último tipo de cambio registrado:       1 EUR = {summary['latest_rate']:.4f} USD")
    logger.info(f" - Tiempo de ejecución:                    {elapsed:.2f} segundos")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ETL de ingesta de tipos de cambio del BCE (FlatexDEGIRO Case Study - Tarea 1a)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "daily"],
        required=True,
        help="Modo de ejecución: 'initial' para carga histórica amplia, 'daily' para actualización incremental.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=config.INITIAL_START_DATE,
        help="Fecha de inicio en formato 'YYYY-MM-DD' (usada exclusivamente en modo 'initial').",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(config.DB_PATH),
        help="Ruta al archivo DuckDB de almacenamiento analítico local.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar salida detallada de depuración (DEBUG).",
    )

    args = parser.parse_args()
    logger = setup_logging(verbose=args.verbose)

    # Instanciar componentes modulares
    client = ECBClient()
    db = DuckDBManager(db_path=args.db_path)

    try:
        if args.mode == "initial":
            run_initial_load(client=client, db=db, start_date=args.start_date, logger=logger)
        elif args.mode == "daily":
            run_daily_update(client=client, db=db, logger=logger)
    except Exception as exc:
        logger.exception(f"Error inesperado durante la ejecución del pipeline: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

