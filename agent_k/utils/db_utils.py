"""Database utility functions for PostgreSQL operations.

# Method 1: Regular instantiation
db = PostgresDB(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
)
tables = db.list_tables()
print(tables)
db.close()

# Method 2: Using context manager with default parameters (recommended)
with PostgresDB() as db:
    tables = db.list_tables()
    schemas = db.list_schemas()
    print(tables)
    print(schemas)
    # connection automatically closes after the with block
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from loguru import logger
from psycopg2.extensions import connection
from psycopg2.extras import RealDictCursor

import agent_k.config.general as config_general

load_dotenv()


class PostgresDB:
    """A class to manage PostgreSQL database operations."""

    def __init__(
        self,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "postgres",
        host: str = "localhost",
        port: str = "5432",
    ):
        """Initialize database connection."""
        self.conn = self._get_db_connection(dbname, user, password, host, port)

    def _get_db_connection(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str,
        port: str,
    ) -> connection:
        """Create a PostgreSQL database connection."""
        try:
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
            )
            return conn
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

    def list_schemas(self) -> List[str]:
        """List all schemas in the database."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
                """
            )
            return [row[0] for row in cur.fetchall()]

    def list_tables(self, schema: str = "public") -> List[str]:
        """List all tables in a given schema."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                """,
                (schema,),
            )
            return [row[0] for row in cur.fetchall()]

    def list_columns(self, table: str, schema: str = "public") -> List[Dict[str, str]]:
        """List all columns and their types for a given table."""
        # RealDictCursor is a special cursor factory in psycopg2 that returns query
        # results as dictionaries instead of tuples.
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            return cur.fetchall()

    def list_column_unique_values(
        self, column: str, table: str, schema: str = "public"
    ) -> List[str]:
        """List all unique values for a given column."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT {column} FROM {schema}.{table}")
            return [row[0] for row in cur.fetchall()]

    def list_columns_with_details(self, table: str, schema: str = "public") -> str:
        """Return a string description of the table. Mock method for now."""
        # Read all files in a directory recursively
        all_file_paths = {}
        for root, _dirs, files in os.walk(os.path.join(config_general.ALL_SOURCES_DIR)):
            for file in files:
                all_file_paths[file] = os.path.join(root, file)

        # Match the file path to the table name "{table}.md"
        if f"{table}.md" in all_file_paths:
            with open(all_file_paths[f"{table}.md"], "r") as file:
                return file.read()
        else:
            return f"No description found for {schema}.{table}."

    def insert_df(self, table: str, df: pd.DataFrame, chunk_size: int = 1000) -> None:
        """Insert data from a pandas DataFrame into a table."""
        # TODO: refactor to use sqlalchemy and df.to_sql() method and handle nan values
        total_chunks = len(df) // chunk_size

        with self.conn.cursor() as cur:
            # Process the dataframe in chunks
            for i in range(0, len(df), chunk_size):
                df_chunk = df.iloc[i : i + chunk_size]
                values = ",".join(
                    cur.mogrify(
                        f"({','.join(['%s'] * len(df_chunk.columns))})", tuple(x)
                    ).decode("utf-8")
                    for x in df_chunk.values
                )
                insert_query = f"""
                INSERT INTO {table} VALUES {values}
                """
                # Convert nan values to None to be compatible with PostgreSQL NULL values
                # TODO: Temp fix to handle nan values
                insert_query = insert_query.replace("'NaN'", "NULL")
                cur.execute(insert_query)
                self.conn.commit()  # Commit each chunk
                logger.debug(
                    f"Inserted chunk {i//chunk_size + 1} / {total_chunks} ({i} to {i+len(df_chunk)} rows)"
                )

    def run_query(
        self, query: str, output_dir: Optional[str] = config_general.AGENT_CACHE_DIR
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
            with self.conn.cursor() as cur:
                cur.execute(query)

                if query.strip().upper().startswith("SELECT"):
                    # Fetch column names
                    cols = [desc[0] for desc in cur.description]
                    # Fetch data
                    data = cur.fetchall()
                    df = pd.DataFrame(data, columns=cols)

                    if output_dir:
                        # Create directory if it doesn't exist
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        file_path = os.path.join(output_dir, f"{timestamp}.json")
                        df.to_json(file_path, orient="values")
                        return (
                            True,
                            f"Query executed successfully. Results saved to {file_path}",
                            df,
                        )
                    return True, "Query executed successfully", df

                self.conn.commit()
                return (
                    True,
                    f"Query executed successfully. {cur.rowcount} rows affected.",
                    None,
                )

        except Exception as e:
            self.conn.rollback()
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


# Test the class
# if __name__ == "__main__":
#     with PostgresDB() as db:
#         print(db.list_tables())
#         print("-"*100)
#         print(db.list_schemas())
#         print("-"*100)
#         print(db.list_columns("mrds_nickel"))
#         print("-"*100)
#         print(db.list_column_unique_values("country", "mrds_nickel"))
#         print("-"*100)
#         print(db.list_columns_with_details("mrds_nickel"))
