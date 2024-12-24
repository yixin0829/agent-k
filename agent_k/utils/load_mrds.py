import os

import pandas as pd
from loguru import logger
from psycopg2 import Error as PostgresError

from agent_k.config.general import MRDS_DIR, mrds_dtype
from agent_k.utils.db_utils import get_db_connection, insert_df, run_query


def load_mrds_to_postgres():
    """Load MRDS data into PostgreSQL database."""
    conn = None
    success = False

    try:
        # 1. Read MRDS CSV file
        mrds_file = os.path.join(MRDS_DIR, "mrds.csv")
        logger.info(f"Reading MRDS data from {mrds_file}")
        df = pd.read_csv(mrds_file, dtype=mrds_dtype)

        # 2. Connect to PostgreSQL
        conn = get_db_connection()

        # 3. Create table schema
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
        DROP TABLE IF EXISTS mrds;
        CREATE TABLE mrds (
            {','.join(columns)}
        );
        """

        success, message, _ = run_query(conn, create_table_query)
        if not success:
            raise PostgresError(f"Failed to create table: {message}")
        logger.info("Created MRDS table successfully")

        # 4. Insert data
        insert_df(conn, "mrds", df, chunk_size=10000)
        logger.info("Inserted MRDS data successfully")

        # 5. Verify data
        test_query = "SELECT COUNT(*) FROM mrds"
        success, message, result_df = run_query(conn, test_query)
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
    finally:
        if conn:
            conn.close()
    return success


if __name__ == "__main__":
    load_mrds_to_postgres()
