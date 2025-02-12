import json
import os

import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.utils.db_utils import DuckDBWrapper


def load_eval_set(eval_set_version: str = "v3"):
    # Read the eval dataset
    with open(
        os.path.join(
            config_general.EVAL_DIR,
            config_general.eval_set_matched_based_file(
                config_general.COMMODITY, eval_set_version
            ),
        ),
        "r",
    ) as f:
        eval_set = [json.loads(line) for line in f]
        logger.info(f"Eval set loaded: {len(eval_set)} questions")

    return eval_set


def load_latest_pdf_extraction() -> pd.DataFrame:
    """
    Load the latest extraction results from the cache directory.
    """
    csv_files = [
        f
        for f in os.listdir(config_general.PDF_AGENT_CACHE_DIR)
        if f.startswith("pdf_agent_extraction_") and f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No extraction CSV files found in {config_general.PDF_AGENT_CACHE_DIR}"
        )

    latest_file = os.path.join(
        config_general.PDF_AGENT_CACHE_DIR,
        max(
            csv_files,
            key=lambda f: os.path.getctime(
                os.path.join(config_general.PDF_AGENT_CACHE_DIR, f)
            ),
        ),
    )

    if not os.path.exists(latest_file):
        raise FileNotFoundError(
            f"No extraction results found in {config_general.PDF_AGENT_CACHE_DIR}"
        )

    # Ask user to confirm the file
    logger.info(f"Latest PDF agent extraction file: {latest_file}")
    # user_input = input("Do you want to load the file? (y/n) ")
    # if user_input != "y":
    #     raise ValueError("User did not confirm the file")

    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} records from {latest_file}")

    return df


def load_latest_extraction_to_duckdb():
    """
    Load the latest extraction results to duckdb
    """
    df = load_latest_pdf_extraction()

    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        db.create_table_from_df("ni_43_101", df)
        logger.info(f"Loaded {len(df)} rows into 43_101_extraction table")
