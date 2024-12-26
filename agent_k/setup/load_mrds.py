"""
Load MRDS data filtered for current commodity into PostgreSQL database.
"""

import os

import pandas as pd
from loguru import logger
from psycopg2 import Error as PostgresError

import agent_k.config.general as config_general
from agent_k.utils.db_utils import PostgresDB


def load_mrds_to_postgres():
    success = False

    try:
        # 1. Read MRDS CSV file (filtered for current commodity)
        mrds_file = os.path.join(
            config_general.MRDS_DIR, f"mrds_{config_general.COMMODITY}.csv"
        )
        logger.info(f"Reading MRDS data from {mrds_file}")
        df = pd.read_csv(mrds_file, dtype=config_general.mrds_dtype)

        # 2. Use PostgresDB with context manager
        with PostgresDB() as db:
            # 3. Create DDL table schema
            dtype_mapping = {
                "category": "TEXT",
                "object": "TEXT",
                "int64": "BIGINT",
                "float64": "DOUBLE PRECISION",
                "bool": "BOOLEAN",
                "datetime64[ns]": "TIMESTAMP",
            }

            columns = []
            for col, dtype in df.dtypes.items():
                pg_type = dtype_mapping.get(str(dtype), "TEXT")
                columns.append(f'"{col}" {pg_type}')

            create_table_query = f"""
            DROP TABLE IF EXISTS mrds_{config_general.COMMODITY};
            CREATE TABLE mrds_{config_general.COMMODITY} (
                {','.join(columns)}
            );
            """

            success, message, _ = db.run_query(create_table_query)
            if not success:
                raise PostgresError(f"Failed to create table: {message}")
            logger.info("Created MRDS table successfully")

            # 4. Insert data
            db.insert_df(f"mrds_{config_general.COMMODITY}", df, chunk_size=1000)
            logger.info("Inserted MRDS data successfully")

            # 5. Verify data
            test_query = f"SELECT COUNT(*) FROM mrds_{config_general.COMMODITY}"
            success, message, result_df = db.run_query(test_query)
            if not success:
                raise PostgresError(f"Failed to verify data: {message}")

            if result_df.iloc[0, 0] != len(df):
                raise ValueError(
                    f"Row count mismatch: {result_df.iloc[0, 0]} in database vs {len(df)} in source"
                )

            logger.info(f"Successfully loaded {len(df)} rows into MRDS table")
            success = True

    except FileNotFoundError:
        logger.error(f"MRDS file not found at {mrds_file}")
    except pd.errors.EmptyDataError:
        logger.error("MRDS file is empty")
    except PostgresError as e:
        logger.error(f"Database error: {e}")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return success


if __name__ == "__main__":
    load_mrds_to_postgres()
