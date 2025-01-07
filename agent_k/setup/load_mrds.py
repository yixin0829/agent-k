"""
Load MRDS data filtered for current commodity into DuckDB database.
"""

import os

import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.utils.db_utils import DuckDBWrapper


def load_mrds_to_duckdb():
    success = False

    try:
        # 1. Read MRDS CSV file (filtered for current commodity)
        mrds_table_name = f"mrds_{config_general.COMMODITY}"
        mrds_file_path = os.path.join(config_general.MRDS_DIR, f"{mrds_table_name}.csv")
        logger.info(f"Reading MRDS data from {mrds_file_path}")
        df = pd.read_csv(mrds_file_path, dtype=config_general.MRDS_DTYPE)
        # 2. Use DuckDB with context manager and specified database file
        with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
            # 3. Create table and insert data
            db.create_table_from_df(mrds_table_name, df)
            logger.info(f"Created MRDS table {mrds_table_name} successfully")
            logger.info(f"Inserted {len(df)} rows into MRDS table {mrds_table_name}")

            # 4. Verify data
            success, message, result_df = db.run_query(
                f"SELECT COUNT(*) FROM {mrds_table_name}"
            )
            if not success:
                raise ValueError(f"Failed to verify data: {message}")

            if result_df.iloc[0, 0] != len(df):
                raise ValueError(
                    f"Row count mismatch: {result_df.iloc[0, 0]} in database vs {len(df)} in source"
                )

            logger.info(f"Successfully loaded {len(df)} rows into MRDS table")
            success = True

    except FileNotFoundError:
        logger.error(f"MRDS file not found at {mrds_file_path}")
    except pd.errors.EmptyDataError:
        logger.error("MRDS file is empty")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return success


if __name__ == "__main__":
    load_mrds_to_duckdb()
