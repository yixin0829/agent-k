"""Database utility functions for PostgreSQL operations."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from loguru import logger
from psycopg2.extensions import connection
from psycopg2.extras import RealDictCursor


def get_db_connection(
    dbname: str = "postgres",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: str = "5432",
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


def list_schemas(conn: connection) -> List[str]:
    """List all schemas in the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
            """
        )
        return [row[0] for row in cur.fetchall()]


def list_tables(conn: connection, schema: str = "public") -> List[str]:
    """List all tables in a given schema."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            """,
            (schema,),
        )
        return [row[0] for row in cur.fetchall()]


def list_columns(
    conn: connection, table: str, schema: str = "public"
) -> List[Dict[str, str]]:
    """List all columns and their types for a given table."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
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


def insert_df(
    conn: connection, table: str, df: pd.DataFrame, chunk_size: int = 1000
) -> None:
    """Insert data from a pandas DataFrame into a table."""
    total_chunks = len(df) // chunk_size
    with conn.cursor() as cur:
        # Process the dataframe in chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            values = ",".join(
                cur.mogrify(
                    f"({','.join(['%s'] * len(chunk.columns))})", tuple(x)
                ).decode("utf-8")
                for x in chunk.values
            )
            insert_query = f"""
            INSERT INTO {table} VALUES {values}
            """
            cur.execute(insert_query)
            conn.commit()  # Commit each chunk
            logger.debug(
                f"Inserted chunk {i//chunk_size + 1} / {total_chunks} ({i} to {i+len(chunk)} rows)"
            )


def run_query(
    conn: connection, query: str, output_path: Optional[str] = None
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Execute a SQL query and optionally save results to CSV.

    Returns:
        Tuple containing:
        - success: bool indicating if query executed successfully
        - message: error message if failed, success message if passed
        - df: pandas DataFrame with results if query was a SELECT
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)

            if query.strip().upper().startswith("SELECT"):
                # Fetch column names
                cols = [desc[0] for desc in cur.description]
                # Fetch data
                data = cur.fetchall()
                df = pd.DataFrame(data, columns=cols)

                if output_path:
                    df.to_csv(output_path, index=False)
                    return (
                        True,
                        f"Query executed successfully. Results saved to {output_path}",
                        df,
                    )
                return True, "Query executed successfully", df

            conn.commit()
            return (
                True,
                f"Query executed successfully. {cur.rowcount} rows affected.",
                None,
            )

    except Exception as e:
        conn.rollback()
        return False, str(e), None
