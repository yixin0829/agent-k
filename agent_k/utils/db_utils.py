"""Database utility functions for DuckDB operations.

# Method 1: Regular instantiation
db = DuckDB(database="my_db.duckdb")
tables = db.list_tables()
db.close()

# Method 2: Using context manager with default parameters (recommended)
with DuckDB() as db:
    tables = db.list_tables()
    # connection automatically closes after the with block
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger


class DuckDBWrapper:
    """A class to manage DuckDB database operations."""

    def __init__(
        self,
        database: str = ":memory:",  # Use in-memory database by default
    ):
        """Initialize database connection."""
        self.conn = self._get_db_connection(database)

    def _get_db_connection(
        self,
        database: str,
    ) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB database connection."""
        try:
            conn = duckdb.connect(database=database)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables in a given schema."""
        result = self.conn.sql("SHOW TABLES").fetchall()
        return [row[0] for row in result]

    def list_columns(self, table: str) -> List[Dict]:
        """List all columns and their types for a given table."""
        result = self.conn.sql(f"DESCRIBE {table}").df()
        return result.to_dict("records")

    def list_column_unique_values(self, table: str, column: str) -> List[str]:
        """List all unique values for a given column."""
        result = self.conn.sql(f"SELECT DISTINCT {column} FROM {table}").fetchall()
        return [row[0] for row in result]

    def list_columns_with_details(self, table: str) -> str:
        """Mock function to return a string description of the table from markdown files."""
        all_file_paths = {}
        for root, _dirs, files in os.walk(os.path.join(config_general.ALL_SOURCES_DIR)):
            for f in files:
                all_file_paths[f] = os.path.join(root, f)

        if f"{table}.md" in all_file_paths:
            with open(all_file_paths[f"{table}.md"], "r") as file:
                return file.read()
        else:
            return f"No description found for {table}."

    def create_table_from_df(self, table: str, df: pd.DataFrame) -> None:
        """Create a table from a pandas DataFrame. Drop table if it exists."""
        try:
            # DuckDB has efficient DataFrame insertion
            self.conn.sql(f"DROP TABLE IF EXISTS {table}")
            self.conn.sql(f"CREATE TABLE {table} AS SELECT * FROM df")
            logger.debug(f"Inserted {len(df)} rows into {table}")
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            raise

    def run_query(
        self, query: str, output_dir: Optional[str] = config_general.DB_AGENT_CACHE_DIR
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Execute a SQL query and optionally save results to a JSON file.

        Returns:
            Tuple containing:
            - success: bool indicating if query executed successfully
            - message: error message if failed, success message if passed
            - df: pandas DataFrame with results if query was a SELECT
        """
        try:
            if query.strip().upper().startswith("SELECT"):
                df = self.conn.sql(query).df()

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                    file_path = os.path.join(output_dir, f"{timestamp}.json")
                    df.to_json(file_path, orient="values")
                    return (
                        True,
                        f"Query executed successfully. Results saved to {file_path}",
                        df,
                    )
                return True, "Query executed successfully", df
            else:
                self.conn.sql(query)
                return True, "Query executed successfully", None

        except Exception as e:
            return False, str(e), None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
