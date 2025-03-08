"""
Master setup script that orchestrates all data download and loading steps.

This script runs the following steps in sequence:
1. Download MRDS data and filter for current commodity
2. Load MRDS data into DuckDB
3. Download and enrich MinMod Hyper data (ground truth)
4. Download 43-101 reports
5. Construct eval set (matched-based)
"""

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.setup.construct_db_eval import construct_eval_set_matched_based
from agent_k.setup.download_43_101 import download_reports_main
from agent_k.setup.download_hyper import download_minmod_hyper_csv, enrich_minmod_hyper
from agent_k.setup.download_mrds import download_mrds_main
from agent_k.setup.load_mrds import load_mrds_to_duckdb


def setup_all():
    """Run all setup steps in sequence."""
    logger.info(f"Starting setup for commodity: {config_general.COMMODITY}")

    # Step 1: Download MRDS data
    logger.info("Step 1: Downloading MRDS data...")
    download_mrds_main()

    # Step 2: Load MRDS data into DuckDB
    logger.info("Step 2: Loading MRDS data into DuckDB...")
    success = load_mrds_to_duckdb()

    # Step 3: Download and enrich MinMod Hyper data
    logger.info("Step 3: Downloading and enriching MinMod Hyper data...")
    download_minmod_hyper_csv()
    enrich_minmod_hyper()

    # Step 4: Download 43-101 reports
    logger.info("Step 4: Downloading 43-101 reports...")
    download_reports_main()

    # Step 5: Construct eval set
    logger.info("Step 5: Constructing eval set...")
    construct_eval_set_matched_based()

    if success:
        logger.info("All setup steps completed successfully!")
    else:
        logger.error("Setup failed during MRDS data loading step")


if __name__ == "__main__":
    setup_all()
