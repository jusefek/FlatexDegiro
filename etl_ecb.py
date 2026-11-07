#!/usr/bin/env python3
"""
ECB Foreign Exchange Reference Rates ETL Pipeline (ECB Data Portal).
FlatexDEGIRO Case Study - Task 1a: Financial Risk Data Engineering.

Usage:
    python etl_ecb.py --mode initial [--start-date 2015-01-01]
    python etl_ecb.py --mode daily
"""
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
import time

from src.config import config
from src.ecb_client import ECBClient
from src.validator import FXDataValidator, DataQualityError
from src.database import DuckDBManager


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configures structured console logging."""
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
    Executes the initial historical bulk load of USD/EUR exchange rates.
    """
    logger.info("=" * 70)
    logger.info(f"STARTING INITIAL HISTORICAL BACKFILL (startPeriod={start_date})")
    logger.info("=" * 70)
    start_time = time.time()

    # 1. Ensure DuckDB table schema
    db.initialize_schema()

    # 2. Extract from ECB SDMX API
    df_raw = client.fetch_fx_data(start_period=start_date)
    if df_raw is None or df_raw.empty:
        logger.error("No data received from ECB for initial load. Aborting.")
        sys.exit(1)

    # 3. Data Quality Validation
    try:
        df_clean = FXDataValidator.validate_and_clean(df_raw)
    except DataQualityError as err:
        logger.error(f"Data quality check failed during initial load: {err}")
        sys.exit(1)

    # 4. Idempotent upsert into DuckDB
    inserted_count = db.upsert_fx_rates(df_clean)

    # 5. Final audit summary
    elapsed = time.time() - start_time
    summary = db.get_metadata_summary()
    logger.info("-" * 70)
    logger.info("INITIAL LOAD SUMMARY:")
    logger.info(f" - Records processed in this run: {inserted_count:,}")
    logger.info(f" - Total records in ecb_fx_rates: {summary['total_records']:,}")
    logger.info(f" - Stored date range:             {summary['min_date']} to {summary['max_date']}")
    logger.info(f" - Latest stored spot rate:       1 EUR = {summary['latest_rate']:.4f} USD")
    logger.info(f" - Execution time:                {elapsed:.2f} seconds")
    logger.info("=" * 70)


def run_daily_update(client: ECBClient, db: DuckDBManager, logger: logging.Logger) -> None:
    """
    Executes the incremental daily update of USD/EUR exchange rates.
    Detects the latest stored date and requests only subsequent observations.
    """
    logger.info("=" * 70)
    logger.info("STARTING INCREMENTAL DAILY UPDATE")
    logger.info("=" * 70)
    start_time = time.time()

    # 1. Ensure DuckDB table schema
    db.initialize_schema()

    # 2. Query latest stored date
    latest_stored_date = db.get_latest_rate_date()

    if latest_stored_date is None:
        logger.warning(
            "Local database is empty. No prior history found for incremental update. "
            f"Falling back to initial backfill from {config.INITIAL_START_DATE}..."
        )
        run_initial_load(client, db, config.INITIAL_START_DATE, logger)
        return

    logger.info(f"Latest rate_date stored in local DuckDB: {latest_stored_date}")

    # 3. Determine next query date
    next_date = latest_stored_date + timedelta(days=1)
    today_utc = datetime.now(timezone.utc).date()

    if next_date > today_utc:
        logger.info(
            f"Local warehouse is already up to date ({latest_stored_date}). "
            "No newer observations to request."
        )
        logger.info("=" * 70)
        return

    logger.info(f"Requesting ECB observations starting from: {next_date.isoformat()}")

    # 4. Extract from ECB SDMX API
    df_raw = client.fetch_fx_data(start_period=next_date.isoformat())

    if df_raw is None or df_raw.empty:
        logger.info(
            f"No new rates published by the ECB since {next_date}. "
            "Expected cause: weekend, TARGET2 bank holiday, or pending publication (~16:00 CET)."
        )
        logger.info("Database status: UNCHANGED (consistent and up to date).")
        logger.info("=" * 70)
        return

    # 5. Data Quality Validation
    try:
        df_clean = FXDataValidator.validate_and_clean(df_raw)
    except DataQualityError as err:
        logger.error(f"Data quality check failed during daily update: {err}")
        sys.exit(1)

    # 6. Idempotent upsert into DuckDB
    inserted_count = db.upsert_fx_rates(df_clean)

    # 7. Final audit summary
    elapsed = time.time() - start_time
    summary = db.get_metadata_summary()
    logger.info("-" * 70)
    logger.info("DAILY UPDATE SUMMARY:")
    logger.info(f" - New records upserted:          {inserted_count}")
    logger.info(f" - Total records in ecb_fx_rates: {summary['total_records']:,}")
    logger.info(f" - New maximum date stored:       {summary['max_date']}")
    logger.info(f" - Latest stored spot rate:       1 EUR = {summary['latest_rate']:.4f} USD")
    logger.info(f" - Execution time:                {elapsed:.2f} seconds")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ECB FX Reference Rate ETL Pipeline (FlatexDEGIRO Case Study - Task 1a)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "daily"],
        required=True,
        help="Execution mode: 'initial' for historical backfill, 'daily' for incremental delta update.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=config.INITIAL_START_DATE,
        help="Start date in 'YYYY-MM-DD' format (used in 'initial' mode).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(config.DB_PATH),
        help="Path to local analytical DuckDB database file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed DEBUG logging output.",
    )

    args = parser.parse_args()
    logger = setup_logging(verbose=args.verbose)

    client = ECBClient()
    db = DuckDBManager(db_path=args.db_path)

    try:
        if args.mode == "initial":
            run_initial_load(client=client, db=db, start_date=args.start_date, logger=logger)
        elif args.mode == "daily":
            run_daily_update(client=client, db=db, logger=logger)
    except Exception as exc:
        logger.exception(f"Unexpected error during ETL pipeline execution: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
