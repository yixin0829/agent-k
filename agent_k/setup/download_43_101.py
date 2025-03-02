"""
1. Fetch deduplicated mineral site entities for the current commodity from the MinMod API.
2. Download 43-101 PDF reports from the CDR API.
"""

import asyncio
import os
import warnings

import httpx
import pandas as pd
from tqdm import tqdm

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, MinModHyperCols

warnings.filterwarnings("ignore")
tqdm.pandas()


async def download_report(record_id: str, save_path: str, semaphore: asyncio.Semaphore):
    """
    Downloads a single PDF report from the CDR API using the provided record ID.

    Args:
        record_id: Unique identifier for the report to download
        save_path: Directory path where the PDF should be saved
        semaphore: Semaphore to control concurrent downloads

    Returns:
        bool: True if download was successful, False otherwise
    """
    url = (
        config_general.API_CDR_LAND_URL
        + config_general.DOCUMENT_BY_ID_ENDPOINT.format(doc_id=record_id)
    )
    async with semaphore:  # Limit concurrent requests
        try:
            async with httpx.AsyncClient() as client:
                # Download PDF with authentication headers and write to file in save_path
                res = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {os.getenv('API_CDR_AUTH_TOKEN')}"
                    },
                    follow_redirects=True,
                )
                res.raise_for_status()
                logger.info(f"Downloaded {record_id} successfully")

                os.makedirs(save_path, exist_ok=True)

                with open(os.path.join(save_path, f"{record_id}.pdf"), "wb") as f:
                    f.write(res.content)
                return True
        except Exception:
            logger.error(f"Error downloading {record_id}: {res.status_code}")
            return False


async def download_all_reports(
    unique_43_101_record_ids: list[str], max_concurrent_requests: int
):
    """
    Downloads all PDF reports referenced in the provided DataFrame concurrently.

    Args:
        unique_43_101_record_ids: List of unique 43-101 report record IDs
        max_concurrent_requests: Maximum number of concurrent downloads

    Returns:
        List of tuples containing (record ID (str), download result (bool))
    """
    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks = []

    # Create download tasks for records with 43-101 data source and not yet downloaded
    for record_id in unique_43_101_record_ids:
        task = asyncio.create_task(
            download_report(
                record_id,
                save_path=config_general.CDR_REPORTS_DIR,
                semaphore=semaphore,
            )
        )
        tasks.append((task, record_id))

    # Wait for all downloads to complete
    results = []
    for task, record_id in tasks:
        result: bool = await task
        results.append((record_id, result))

    return results


def download_reports_main(max_concurrent_requests: int = 10):
    """
    Main function to orchestrate downloading all reports.
    """
    # Load report metadata
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )

    # Filter for unique and not yet downloaded 43-101 report record IDs
    df_hyper_43_101 = df_hyper[
        (df_hyper[MinModHyperCols.DATA_SOURCE.value] == DataSource.API_CDR_LAND.value)
        & (~df_hyper[MinModHyperCols.DOWNLOADED_PDF.value])
    ]
    unique_43_101_record_ids = df_hyper_43_101[
        MinModHyperCols.RECORD_VALUE.value
    ].unique()

    # Download all reports concurrently
    results = asyncio.run(
        download_all_reports(unique_43_101_record_ids, max_concurrent_requests)
    )

    # Filter for successful downloads record IDs
    successful_downloads = [record_id for record_id, success in results if success]

    # Update DataFrame with successful downloads
    df_hyper.loc[
        df_hyper[MinModHyperCols.RECORD_VALUE.value].isin(successful_downloads),
        MinModHyperCols.DOWNLOADED_PDF.value,
    ] = True

    # logger.info summary and save updated DataFrame
    logger.info(
        "Summarize unique 43-101 reports downloaded:\n"
        f"{df_hyper.drop_duplicates(subset=[MinModHyperCols.RECORD_VALUE.value])[MinModHyperCols.DOWNLOADED_PDF.value].value_counts()}"
    )
    df_hyper.to_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )


if __name__ == "__main__":
    download_reports_main()
